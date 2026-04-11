import os
import re
import torch
import random
import numpy as np
import transformers
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from collections import Counter
from unsloth import FastLanguageModel
from datasets import Dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from sklearn.metrics import accuracy_score, confusion_matrix
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments, EarlyStoppingCallback

# =========================================================
# 1. Setup & Configuration
# =========================================================
transformers.logging.set_verbosity_error()

dataset_path = "dataset/dataset.csv"      
benchmark_path = "dataset/benchmark.csv"  
dir_path = "./"
try_name = "fine_tune_4"

# =========================================================
# 2. Load Model & Tokenizer
# =========================================================
max_seq_length = 1024
dtype = None 
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 256,
    lora_alpha = 512,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout = 0.03,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    modules_to_save = ["embed_tokens", "lm_head"],
)

# =========================================================
# 3. Prompt Template Definition
# =========================================================
prompt_template = """System: You are an elite, board-certified expert in Medical Pathology. Your task is to evaluate clinical and pathological multiple-choice questions with absolute medical accuracy.

### strict_rules:
1. Analyze the question and the four options carefully.
2. Output ONLY the single correct letter (A, B, C, or D).
3. DO NOT output any explanations, rationale, punctuation, or preamble.
4. Failure to adhere to these format constraints will result in system failure.
5. Think step-by-step through the pathological mechanisms before answering.

---
### demonstration_examples:

[Question]:
RBC contains?
[Options]:
A. Iron
B. Folic acid
C. Vitamin C
D. Biotin
[Answer]:
A

[Question]:
Fibrosis is due to -
[Options]:
A. TNF -α
B. TGF-β
C. IL-7
D. IL-10
[Answer]:
B

[Question]:
Ability of stem cells to cross barrier of differentiation to transform into a cell of another lineage expressing the molecular characteristics of different cell type with the ability to perform the function of the new cell type is referred as:
[Options]:
A. De differentiation
B. Re differentiation
C. Trans-differentiation
D. Sub differentiation
[Answer]:
C

[Question]:
Severe hypovolemic shock occurs when blood volume less is -
[Options]:
A. > 10 %
B. > 20 %
C. > 30 %
D. > 40%
[Answer]:
D

[Question]:
Neurologic abnormalities have been noted in about one-third of patients with AIDS. Which of the following is NOT seen in HIV involvement of CNS?
[Options]:
A. Perivascular giant cell
B. Vacuolar degeneration of post column
C. Microglial nodule formation
D. Inclusion bodies
[Answer]:
D

[Question]:
The worst prognosis for renal cell carcinoma is
[Options]:
A. Vascular invasion
B. Associated with hypercalcemia
C. Presence of Hematuria
D. Size more than 5 cm.
[Answer]:
A

---
### current_task:
[Question]:
{question}
[Options]:
A. {opa}
B. {opb}
C. {opc}
D. {opd}
[Answer]:
"""

# =========================================================
# 4. Data Loading, Splitting & Augmentation
# =========================================================

# Map numerical answers (0, 1, 2, 3) to letters (A, B, C, D)
num_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

print(f"Reading dataset: {dataset_path}")
df = pd.read_csv(dataset_path)
df['ans_letter'] = df['ans'].map(num_to_letter)

# Split dataset (80% Train, 10% Val, 10% Test)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)
print(f"Original train set: {len(train_df)} | Validation set: {len(val_df)} | Test set: {len(test_df)}")

# Create a global option pool for data augmentation
all_options_series = pd.concat([train_df['opa'], train_df['opb'], train_df['opc'], train_df['opd']])
global_option_pool = [str(opt).strip() for opt in all_options_series.dropna().unique() if str(opt).strip() != '']
print(f"Option pool created successfully, collected {len(global_option_pool)} unique medical terms!")

# Convert Pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

def data_augment(example):
    original_options = [example['opa'], example['opb'], example['opc'], example['opd']]
    correct_text = str(original_options[example['ans']]).strip()

    new_distractors = random.sample([opt for opt in global_option_pool if opt != correct_text], 3)
    new_options = [correct_text] + new_distractors
    random.shuffle(new_options)
    new_ans_idx = new_options.index(correct_text)

    example['opa'], example['opb'], example['opc'], example['opd'] = new_options
    example['ans'] = new_ans_idx
    example['ans_letter'] = num_to_letter[new_ans_idx] 
    return example

def format_prompt_hf(example, is_training=True):
    prompt = prompt_template.format(
        question=example["question"],
        opa=example["opa"],
        opb=example["opb"],
        opc=example["opc"],
        opd=example["opd"]
    )

    if is_training:
        raw_ans = example.get("ans")
        ans_letter = num_to_letter.get(int(raw_ans))
        eos = tokenizer.eos_token if tokenizer else "<|end_of_text|>"
        example["text"] = prompt + ans_letter + eos
    else:
        example["text"] = prompt

    return example

# Apply augmentation to the training dataset and concatenate
augmented_dataset = train_dataset.map(data_augment)
train_dataset = concatenate_datasets([train_dataset, augmented_dataset]).shuffle(seed=42)
print(f"After augmentation -> Final train set size: {len(train_dataset)}")

print("Formatting Datasets...")
train_dataset = train_dataset.map(lambda x: format_prompt_hf(x, is_training=True), batched=False)
val_dataset = val_dataset.map(lambda x: format_prompt_hf(x, is_training=True), batched=False)
test_dataset = test_dataset.map(lambda x: format_prompt_hf(x, is_training=False), batched=False)

print("Data processing complete!")




# =========================================================
# 4.5 Visualize the distribution the dataset
# =========================================================
sns.set_theme(style="whitegrid")
train_ans_counts = pd.Series(train_dataset['ans_letter']).value_counts().reindex(['A', 'B', 'C', 'D'], fill_value=0)
val_ans_counts = pd.Series(val_dataset['ans_letter']).value_counts().reindex(['A', 'B', 'C', 'D'], fill_value=0)
test_ans_counts = pd.Series(test_dataset['ans_letter']).value_counts().reindex(['A', 'B', 'C', 'D'], fill_value=0)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Training set distribution
sns.barplot(x=train_ans_counts.index, y=train_ans_counts.values, palette='viridis', hue=train_ans_counts.index, legend=False, ax=axes[0])
axes[0].set_title('Training Set Answer Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Answer Choice', fontsize=12)
axes[0].set_ylabel('Number of Questions', fontsize=12)
for index, value in enumerate(train_ans_counts.values):
    axes[0].text(index, value + 0.5, str(value), ha='center', va='bottom', fontsize=10, fontweight='bold')

# Validation set distribution
sns.barplot(x=val_ans_counts.index, y=val_ans_counts.values, palette='viridis', hue=val_ans_counts.index, legend=False, ax=axes[1])
axes[1].set_title('Validation Set Answer Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Answer Choice', fontsize=12)
axes[1].set_ylabel('') # Share the Y-axis, so the label is omitted here
for index, value in enumerate(val_ans_counts.values):
    axes[1].text(index, value + 0.5, str(value), ha='center', va='bottom', fontsize=10, fontweight='bold')

# Test set distribution
sns.barplot(x=test_ans_counts.index, y=test_ans_counts.values, palette='viridis', hue=test_ans_counts.index, legend=False, ax=axes[2])
axes[2].set_title('Test Set Answer Distribution', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Answer Choice', fontsize=12)
axes[2].set_ylabel('') # Share the Y-axis, so the label is omitted here
for index, value in enumerate(test_ans_counts.values):
    axes[2].text(index, value + 0.5, str(value), ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Distribution of Correct Answers Across Datasets', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to leave room for the main title
plt.show()

# Save the figure
plots_dir = os.path.join(dir_path, try_name, "plots")
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, "answer_distribution.png"))
plt.close(fig)

transformers.logging.set_verbosity_info()

# Tell the model to compute loss only for tokens that appear after "[Answer]:\n"
# This string must exactly match the prefix that appears before the answer in prompt_template
response_template = "[Answer]:\n"
collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

# =================================================================
# 5. Training
# =================================================================
training_args = TrainingArguments(
    output_dir = os.path.join(plots_dir, try_name, "logs/training_logs"),
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 16,
    warmup_steps = 20,
    num_train_epochs = 2,
    learning_rate = 3e-5,
    max_grad_norm = 0.3,           # Gradient clipping to prevent sudden loss spikes
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 5,
    optim = "adamw_8bit",
    weight_decay = 0.05,
    lr_scheduler_type = "cosine",
    seed = 3707,

    # Early stopping settings
    eval_strategy = "steps",
    eval_steps = 20,
    # save_strategy = "steps",       # Must match the evaluation step interval
    # save_steps = 20,
    # save_total_limit = 3,          # Keep only a few checkpoints to save disk space
    # load_best_model_at_end = True, # Automatically restore the best model after training
    # metric_for_best_model = "eval_loss", # SFT usually tracks validation loss, where lower is better
    # greater_is_better = False,     # Loss should be minimized
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    dataset_text_field = "text",
    max_seq_length = 1024,
    dataset_num_proc = 2,
    packing = False,
    args = training_args,
    data_collator = collator,
    neftune_noise_alpha = 5,  # Helps reduce overfitting
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer_stats = trainer.train()

# Save LoRA model weights
output_model_path = os.path.join(dir_path, try_name, "saved_models")
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)
print(f"Model saved to: {output_model_path}")


# Set plotting style
sns.set_theme(style="whitegrid")

# Extract training logs from the trainer
log_history = trainer.state.log_history

train_steps, train_loss = [], []
eval_steps, eval_loss = [], []
lr_steps, lr_values = [], []

for log in log_history:
    if "loss" in log and "step" in log:
        train_steps.append(log["step"])
        train_loss.append(log["loss"])
        lr_steps.append(log["step"])
        lr_values.append(log["learning_rate"])
    elif "eval_loss" in log and "step" in log:
        eval_steps.append(log["step"])
        eval_loss.append(log["eval_loss"])

# Create a 1x2 figure canvas
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Loss curves
axes[0].plot(train_steps, train_loss, label="Training Loss", color="blue", marker="o", markersize=4)
if eval_loss:
    axes[0].plot(eval_steps, eval_loss, label="Validation Loss", color="red", marker="s", markersize=5)
axes[0].set_title("Training & Validation Loss Over Steps", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Training Steps", fontsize=12)
axes[0].set_ylabel("Loss", fontsize=12)
axes[0].legend()

# Plot 2: Learning rate curve
axes[1].plot(lr_steps, lr_values, label="Learning Rate", color="green", linestyle="--")
axes[1].set_title("Learning Rate Schedule", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Training Steps", fontsize=12)
axes[1].set_ylabel("Learning Rate", fontsize=12)
axes[1].legend()

plt.tight_layout()
plt.show()

# Save the figure
plots_dir = os.path.join(dir_path, try_name, "plots")
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, "training_metrics.png"))
plt.close(fig)


# =================================================================
# 6. Validation
# =================================================================
transformers.logging.set_verbosity_error()

# Enable Unsloth inference acceleration
FastLanguageModel.for_inference(model)

true_labels = []
pred_labels = []

validation_records = []

print("Starting validation inference...")

# test_df = test_df[:len(test_df) // 5]

# Iterate over the validation set
for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating (Test)"):
    # Fill in the question and answer choices
    prompt = prompt_template.format(
        question=row['question'],
        opa=row['opa'],
        opb=row['opb'],
        opc=row['opc'],
        opd=row['opd']
    )

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    # Generate the answer
    # Since the model only needs to emit one letter among A, B, C, or D, 5 tokens are enough
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        use_cache=False, # Changed to False to prevent RuntimeError
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode only the newly generated tokens so the regex does not match answers from the in-context examples
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Find the first A, B, C, or D in the model output with a regular expression
    # print(f"output: {output_text}")
    match = re.search(r'([ABCD])', output_text.upper())

    if match:
        pred = match.group(1)
    else:
        # Fallback in case the model fails to produce a valid answer letter
        pred = 'A'
        print("not match, change ans to A")

    pred_labels.append(pred)
    true_labels.append(row['ans_letter'].upper())
    # print(f"pred: {pred} / ans: {row['ans_letter'].upper()}")

    is_correct = (pred == row['ans_letter'].upper())
    if not is_correct:
      validation_records.append({
          'raw_output': output_text,
          'parsed_prediction': pred,
          'ground_truth': row['ans_letter'].upper(),
          'opa': row['opa'],
          'opb': row['opb'],
          'opc': row['opc'],
          'opd': row['opd'],
          'question': row['question'],
          'question_id': row.get('question_id', index), # Use the row index if question_id is unavailable
          'is_correct': is_correct,                # Boolean correctness flag
      })

# Compute final accuracy
acc = accuracy_score(true_labels, pred_labels)
print(f"\n Validation Accuracy: {acc*100:.2f}%")


error_report_path = os.path.join(dir_path, try_name, "val_error_report.csv")
error_df = pd.DataFrame(validation_records)
error_df.to_csv(error_report_path, index=False)
print(f"Error report saved ({len(error_df)} questions): {error_report_path}")

# =========================================================
# 6. Validation Metrics
# =========================================================
labels_order = ['A', 'B', 'C', 'D']

# 1. Print the detailed classification report (precision, recall, F1-score)
print("📊 Validation Classification Report:")
print("-" * 50)
print(classification_report(true_labels, pred_labels, labels=labels_order, zero_division=0))
print("-" * 50)

# 2. Plot the confusion matrix heatmap
cm = confusion_matrix(true_labels, pred_labels, labels=labels_order)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=labels_order, yticklabels=labels_order)
axes[0].set_title("Confusion Matrix (Validation)", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Predicted Label", fontsize=12)
axes[0].set_ylabel("True Label", fontsize=12)

# 3. Plot the count distribution of true answers versus predicted answers
true_counts = [true_labels.count(l) for l in labels_order]
pred_counts = [pred_labels.count(l) for l in labels_order]

x = np.arange(len(labels_order))
width = 0.35

axes[1].bar(x - width/2, true_counts, width, label='True Answers', color='skyblue')
axes[1].bar(x + width/2, pred_counts, width, label='Predicted Answers', color='salmon')
axes[1].set_title("Distribution of Answers (Validation)", fontsize=14, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels_order)
axes[1].set_ylabel("Count", fontsize=12)
axes[1].legend()

plt.tight_layout()
plt.show()

# Save the figure
plots_dir = os.path.join(dir_path, try_name, "plots")
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, "val_metrics.png"))
plt.close(fig)

# =========================================================
# 7. ReTraining with Hard Example Mining
# =========================================================

print("Starting hard example mining on the training set (train_df)...")

# Ensure the model is in inference mode
FastLanguageModel.for_inference(model)

hard_records = []
easy_records = [] 

for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Mining Hard Examples"):
    # Note: Do not include the answer here, we are blind-testing it
    prompt = prompt_template.format(
        question=row['question'],
        opa=row['opa'],
        opb=row['opb'],
        opc=row['opc'],
        opd=row['opd']
    )

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        use_cache=False, 
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False # Disable randomness to see its true logic
    )

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Extract ABCD using regular expression
    match = re.search(r'([ABCD])', output_text.upper())
    pred = match.group(1) if match else 'C' 

    # Determine if the answer is incorrect
    is_correct = (pred == row['ans_letter'].upper())
    if not is_correct:
        hard_records.append(row.to_dict())
    else:
        easy_records.append(row.to_dict())

hard_df = pd.DataFrame(hard_records)
easy_df = pd.DataFrame(easy_records)

print(f"Mining complete!")
print(f"Hard examples (incorrect questions): {len(hard_df)}")
print(f"Easy examples (correct questions): {len(easy_df)}")

# =========================================================
# 6. Build Stage 2 Training Dataset (Anti-forgetting Buffer)
# =========================================================
# Duplicate hard examples twice to increase their weight
hard_df_upsampled = pd.concat([hard_df, hard_df], ignore_index=True)

# Sample 3x the number of hard examples from easy ones to prevent catastrophic forgetting
num_easy_samples = min(len(hard_df) * 3, len(easy_df))
easy_df_sampled = easy_df.sample(n=num_easy_samples, random_state=42)

# Merge and shuffle thoroughly
stage2_train_df = pd.concat([hard_df_upsampled, easy_df_sampled], ignore_index=True)
stage2_train_df = stage2_train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

print(f"Stage 2 Final Train Dataset Size: {len(stage2_train_df)}")

# Convert to HF Dataset and apply formatting
stage2_train_dataset = Dataset.from_pandas(stage2_train_df)
print("Formatting Stage 2 Train Dataset...")
stage2_train_dataset = stage2_train_dataset.map(lambda x: format_prompt_hf(x, is_training=True), batched=False)

# =========================================================
# 7. Stage 2 Training (Retraining with low learning rate)
# =========================================================
# Switch back to training mode
FastLanguageModel.for_training(model)

trainer_stage2 = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=stage2_train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length, 
    args=TrainingArguments(
        output_dir=os.path.join(plots_dir, try_name, "logs/retraining_logs"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        num_train_epochs=2, 
        learning_rate=5e-7, 
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        seed=42,
    ),
)

print("Starting Hard Example Retraining (Stage 2)...")
trainer_stage2.train()
print("Stage 2 training complete!")


# Define the path where the model was saved
model_path = "./saved_models"

# Parameters consistent with initial model loading
max_seq_length = max_seq_length
dtype = None # Auto-detect
load_in_4bit = True # 4-bit quantization

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path, # Load from local path
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)


# =========================================================
# 8. High-Temperature Majority Voting Inference (Benchmark)
# =========================================================
print(f"Reading benchmark set: {benchmark_path}")
benchmark_df = pd.read_csv(benchmark_path)

# Ensure the model is in inference mode
FastLanguageModel.for_inference(model)

num_votes=5
temperature=0.9

def get_majority_votes(question, opt_list):
    prompt = prompt_template.format(
        question=question,
        opa=opt_list[0], opb=opt_list[1], opc=opt_list[2], opd=opt_list[3]
    )
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # Generate multiple distinct answers in parallel
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        use_cache=False,       
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,                # Enable random sampling
        temperature=temperature,       # High temperature (0.7 is balanced for MCQs)
        num_return_sequences=num_votes # Output distinct responses at once
    )
    
    votes = []
    input_length = inputs["input_ids"].shape[1]
    
    # Parse the outputs
    for i in range(num_votes):
        generated_tokens = outputs[i][input_length:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        match = re.search(r'([ABCD])', output_text.upper())
        pred = match.group(1) if match else 'C' # Default guess is C
        votes.append(pred)
        
    return votes

predictions = []
print("Starting High-Temperature Majority Voting Prediction ...")

for _, row in tqdm(benchmark_df.iterrows(), total=len(benchmark_df), desc="Majority Voting"):
    q = row['question']
    orig_opts = [row['opa'], row['opb'], row['opc'], row['opd']]
    
    # Obtain 5 votes (e.g., ['A', 'A', 'B', 'A', 'C'])
    votes = get_majority_votes(q, orig_opts, num_votes=5, temperature=0.7)
    
    # Tally votes and pick the highest
    majority_ans_letter = Counter(votes).most_common(1)[0][0]
    
    idx_map_1 = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    final_ans = idx_map_1[majority_ans_letter]
    
    predictions.append(final_ans)

# Save submission file
submission_df = pd.DataFrame({
    'question_id': benchmark_df['question_id'] if 'question_id' in benchmark_df.columns else benchmark_df.index, 
    'ans': predictions
})

os.makedirs(os.path.join(dir_path, try_name), exist_ok=True)
submission_path = os.path.join(dir_path, try_name, "submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"Majority voting prediction complete! Results saved to: {submission_path}")

# torch.cuda.empty_cache()