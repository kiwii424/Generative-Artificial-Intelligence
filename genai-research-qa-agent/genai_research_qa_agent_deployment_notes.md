# GenAI Research QA Agent Deployment Notes

## 1. Project Overview

**GenAI Research QA Agent** is a cloud-deployed FastAPI backend service designed as the deployment foundation for an Agentic RAG research question-answering system.

The long-term goal of this project is to support the following workflow:

1. A user submits a research question and paper.
2. The agent calls a retrieval tool.
3. The system retrieves relevant evidence from research paper chunks.
4. The LLM generates a grounded answer based on the retrieved evidence.
5. The API returns the answer, evidence snippets, source metadata, and retrieval scores.
6. The service is deployed on Google Cloud Run as a public HTTPS API.

Current deployment URL:

```text
https://genai-research-qa-api-232454355491.asia-east1.run.app
```

Current implemented version:

- FastAPI backend
- Health check endpoint
- Query endpoint
- Evidence-based response format
- Dockerized backend
- Google Artifact Registry image hosting
- Google Cloud Run deployment

---

## 2. Why This Project Matters

This project demonstrates the transition from a local prototype to a cloud-native backend service.

Instead of only running a GenAI or RAG prototype in a local notebook, the system is packaged as a containerized backend API and deployed to Google Cloud Run.

This is relevant to GenAI engineering roles because it shows practical experience with:

- API development
- Docker containerization
- Cloud deployment
- Google Cloud Run
- Artifact Registry
- Deployment debugging
- Service accessibility through public HTTPS endpoints

---

## 3. System Architecture

```text
User
  ↓ HTTPS request
Cloud Run Service
  ↓ runs Docker container
FastAPI Backend
  ↓ handles API request
Query Endpoint
  ↓ returns structured JSON
Answer + Evidence
```

Deployment pipeline:

```text
Local Source Code
  ↓ Docker build / Docker buildx
Docker Image
  ↓ Docker push
Google Artifact Registry
  ↓ Cloud Run deploy
Public Cloud Run API
```

---

## 4. API Usage

### 4.1 Health Check

Endpoint:

```text
GET /
```

Example request:

```bash
curl https://genai-research-qa-api-232454355491.asia-east1.run.app/
```

Expected response:

```json
{
  "status": "ok",
  "service": "GenAI Research QA Agent"
}
```

---

### 4.2 Query Endpoint

Endpoint:

```text
POST /query
```

Example request:

```bash
curl -X POST https://genai-research-qa-api-232454355491.asia-east1.run.app/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

Expected response format:

```json
{
  "answer": "Based on retrieved evidence, your question is: What is RAG?",
  "evidence": [
    {
      "source": "sample_paper.pdf",
      "chunk_id": "chunk_001",
      "snippet": "Retrieval-Augmented Generation retrieves relevant context before generating an answer.",
      "score": 0.91
    }
  ]
}
```

---

## 5. How Other People Can Use This Project

Other users can interact with the deployed API in three ways.

### Option 1: Use curl

This is the simplest way to test the backend.

```bash
curl -X POST https://genai-research-qa-api-232454355491.asia-east1.run.app/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval augmented generation?"}'
```

### Option 2: Use Postman

Users can test the API with Postman:

- Method: `POST`
- URL: `https://genai-research-qa-api-232454355491.asia-east1.run.app/query`
- Headers:
  - `Content-Type: application/json`
- Body:

```json
{
  "question": "What is RAG?"
}
```

### Option 3: Use a Simple Frontend

A simple frontend can be built to make the demo easier for non-technical users.

The frontend only needs:

- A text input for the question
- A submit button
- A section to display the answer
- A section to display the retrieved evidence

The frontend can call the Cloud Run API through JavaScript `fetch()`.

---

## 6. Simple Frontend Demo

A minimal HTML frontend can be used to demo the API.

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>GenAI Research QA Agent</title>
</head>
<body>
  <h1>GenAI Research QA Agent</h1>

  <textarea id="question" rows="4" cols="60" placeholder="Ask a research question..."></textarea>
  <br />
  <button onclick="askQuestion()">Ask</button>

  <h2>Answer</h2>
  <pre id="answer"></pre>

  <h2>Evidence</h2>
  <pre id="evidence"></pre>

  <script>
    async function askQuestion() {
      const question = document.getElementById("question").value;

      const response = await fetch("https://genai-research-qa-api-232454355491.asia-east1.run.app/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ question })
      });

      const data = await response.json();

      document.getElementById("answer").textContent = data.answer;
      document.getElementById("evidence").textContent = JSON.stringify(data.evidence, null, 2);
    }
  </script>
</body>
</html>
```

For a more polished portfolio version, the frontend can be implemented with:

- React
- Vite
- Tailwind CSS
- Vercel deployment

---

## 7. Deployment Steps Completed

### Step 1: Set Google Cloud Project

```bash
gcloud config set project genai-research-qa-agent
```

Purpose:

This tells the `gcloud` CLI which Google Cloud project to operate on.

---

### Step 2: Enable Billing

The first deployment attempt failed because billing was not enabled.

Error message:

```text
Billing account for project is not found.
Billing must be enabled for activation of service(s) ...
```

Solution:

A Cloud Billing account was linked to the project.

Verification command:

```bash
gcloud billing projects describe genai-research-qa-agent
```

Successful result:

```yaml
billingEnabled: true
```

Purpose:

Cloud Run, Artifact Registry, and Cloud Build require billing to be enabled, even if the project stays within free-tier usage.

---

### Step 3: Enable Required Google Cloud APIs

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
```

Purpose:

This enables the required Google Cloud services:

- Cloud Run for running the backend API
- Artifact Registry for storing Docker images
- Cloud Build for container-related deployment support

---

### Step 4: Set Cloud Run Region

```bash
gcloud config set run/region asia-east1
```

Purpose:

The service was deployed in `asia-east1`, which is the Taiwan region. This is suitable for lower latency when demoing the service from Taiwan.

---

### Step 5: Create Artifact Registry Repository

```bash
gcloud artifacts repositories create genai-repo \
  --repository-format=docker \
  --location=asia-east1 \
  --description="Docker repository for GenAI Research QA Agent"
```

Purpose:

Artifact Registry stores the Docker image that Cloud Run will pull and execute.

---

### Step 6: Configure Docker Authentication

```bash
gcloud auth configure-docker asia-east1-docker.pkg.dev
```

Purpose:

This allows local Docker to push images to Google Artifact Registry.

---

### Step 7: Build Docker Image

Initial build command:

```bash
docker build -t asia-east1-docker.pkg.dev/genai-research-qa-agent/genai-repo/genai-research-qa-api:v1 .
```

Purpose:

This packages the FastAPI application, dependencies, and runtime environment into a Docker image.

---

### Step 8: Push Docker Image

```bash
docker push asia-east1-docker.pkg.dev/genai-research-qa-agent/genai-repo/genai-research-qa-api:v1
```

Purpose:

This uploads the local Docker image to Artifact Registry so Cloud Run can access it.

---

### Step 9: Resolve Cloud Run Architecture Issue

Deployment initially failed with:

```text
Cloud Run does not support image ...
Container manifest type 'application/vnd.oci.image.index.v1+json' must support amd64/linux.
```

Cause:

The image was built on a Mac environment, which may produce an ARM64 image. Cloud Run requires a Linux AMD64-compatible image.

Solution:

Rebuild and push the image with Docker buildx:

```bash
docker buildx build \
  --platform linux/amd64 \
  -t asia-east1-docker.pkg.dev/genai-research-qa-agent/genai-repo/genai-research-qa-api:v1-amd64 \
  --push .
```

Purpose:

This explicitly builds the image for the `linux/amd64` platform, which is compatible with Cloud Run.

---

### Step 10: Deploy to Cloud Run

```bash
gcloud run deploy genai-research-qa-api \
  --image asia-east1-docker.pkg.dev/genai-research-qa-agent/genai-repo/genai-research-qa-api:v1-amd64 \
  --platform managed \
  --region asia-east1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 1
```

Purpose:

This creates a public Cloud Run service from the Docker image.

Important deployment settings:

- `--allow-unauthenticated`: Allows public access to the API.
- `--memory 512Mi`: Allocates 512 MiB memory.
- `--cpu 1`: Allocates 1 vCPU.
- `--timeout 300`: Allows requests to run for up to 300 seconds.
- `--min-instances 0`: Scales down to zero when idle to reduce cost.
- `--max-instances 1`: Limits scaling to control cost.

Successful deployment result:

```text
Service URL: https://genai-research-qa-api-232454355491.asia-east1.run.app
```

---

## 8. Problems Encountered and Solutions

### Problem 1: Billing Was Not Enabled

Error:

```text
Billing account for project is not found.
```

Cause:

The project was not linked to a Cloud Billing account.

Solution:

Linked a billing account to the project and verified:

```bash
gcloud billing projects describe genai-research-qa-agent
```

---

### Problem 2: Docker Daemon Was Not Running

Error:

```text
Cannot connect to the Docker daemon
```

Cause:

Docker Desktop was not running.

Solution:

Started Docker Desktop and verified with:

```bash
docker info
```

---

### Problem 3: Dockerfile CMD Warning

Warning:

```text
JSONArgsRecommended: JSON arguments recommended for CMD
```

Cause:

The Dockerfile used shell-form `CMD`.

Solution:

Updated the Dockerfile to use JSON-form command:

```dockerfile
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
```

This improves signal handling behavior in containers.

---

### Problem 4: Image Not Found During Cloud Run Deployment

Error:

```text
Image ... not found.
```

Cause:

The image had not been pushed to Artifact Registry before deployment.

Solution:

Pushed the image:

```bash
docker push asia-east1-docker.pkg.dev/genai-research-qa-agent/genai-repo/genai-research-qa-api:v1
```

Verified image existence:

```bash
gcloud artifacts docker images list asia-east1-docker.pkg.dev/genai-research-qa-agent/genai-repo
```

---

### Problem 5: Cloud Run Architecture Mismatch

Error:

```text
Container manifest type ... must support amd64/linux.
```

Cause:

The Docker image was not built for the platform required by Cloud Run.

Solution:

Used Docker buildx with:

```bash
--platform linux/amd64
```

---

## 9. What This Project Currently Demonstrates

This project currently demonstrates:

- Building a FastAPI backend
- Defining REST API endpoints
- Returning structured JSON responses
- Designing an evidence-based answer format
- Containerizing a Python backend with Docker
- Authenticating Docker with Google Artifact Registry
- Pushing Docker images to Artifact Registry
- Deploying a public API to Google Cloud Run
- Debugging real deployment issues
- Handling platform mismatch between Mac local builds and Cloud Run runtime

---

## 10. Next Improvements

Planned improvements:

1. Add real paper chunks as a JSON dataset.
2. Implement BM25 retrieval.
3. Add FAISS vector retrieval.
4. Wrap retrieval as an agent tool.
5. Add LLM answer synthesis.
6. Return answer, evidence, source title, chunk ID, and retrieval score.
7. Add a simple React frontend.
8. Add README architecture diagram.
9. Add request logging and error handling.
10. Add environment variable management for API keys.

---

## 11. Resume Description

Possible resume title:

```text
GenAI Research QA Agent — Python, FastAPI, Docker, Google Cloud Run
```

Resume bullets:

```text
• Built and deployed a cloud-native FastAPI backend on Google Cloud Run, exposing public endpoints for research question answering with evidence-based responses.
• Containerized the backend with Docker, pushed the image to Google Artifact Registry, and deployed a linux/amd64-compatible service to Cloud Run.
• Resolved deployment issues including billing setup, Docker daemon configuration, Artifact Registry image publishing, and architecture mismatch between Mac ARM64 builds and Cloud Run AMD64 runtime.
```

Future version after adding retrieval and LLM:

```text
• Implemented an Agentic RAG workflow that retrieves evidence from research paper chunks and generates grounded answers with source metadata through a cloud-deployed API.
```

---

## 12. Interview Explanation

A concise interview explanation:

```text
I built a cloud-deployed FastAPI backend for a GenAI Research QA Agent. The system is designed as an Agentic RAG API where a user submits a research question, the retrieval tool finds relevant evidence from paper chunks, and the LLM generates a grounded answer with source metadata.

For deployment, I containerized the backend with Docker, pushed the image to Google Artifact Registry, and deployed it to Google Cloud Run as a public HTTPS API. During deployment, I encountered and resolved several real engineering issues, including missing billing setup, Docker daemon configuration, image-not-found errors, and an architecture mismatch caused by building on a Mac ARM environment while Cloud Run required a linux/amd64 image.
```
