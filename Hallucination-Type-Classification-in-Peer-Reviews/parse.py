import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


_SECTION_CATEGORIES = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "related_work", "related works": "related_work",
    "background": "related_work", "prior work": "related_work",
    "literature": "related_work", "literature review": "related_work",
    "preliminaries": "related_work", "preliminary": "related_work",
    "method": "methodology", "methods": "methodology",
    "methodology": "methodology", "approach": "methodology",
    "model": "methodology", "framework": "methodology",
    "system": "methodology", "proposed": "methodology",
    "proposed method": "methodology", "proposed approach": "methodology",
    "architecture": "methodology", "network": "methodology", "networks": "methodology",
    "experiment": "experiments", "experiments": "experiments",
    "experimental": "experiments", "experimental setup": "experiments",
    "setup": "experiments", "evaluation": "experiments",
    "implementation": "experiments", "implementation details": "experiments",
    "result": "results", "results": "results",
    "analysis": "results", "discussion": "results",
    "ablation": "results", "ablations": "results",
    "ablation study": "results", "finding": "results",
    "findings": "results", "performance": "results",
    "conclusion": "conclusion", "conclusions": "conclusion",
    "summary": "conclusion", "future work": "conclusion",
    "data": "data", "dataset": "data", "datasets": "data",
    "corpus": "data", "corpora": "data",
}

_SKIP_HEADERS = {
    "acknowledgment", "acknowledgments", "acknowledgement", "acknowledgements",
    "references", "reference", "bibliography",
    "appendix", "appendices", "supplementary",
    "supplementary material", "supplementary materials",
    "ethics statement", "broader impact", "broader impacts",
    "limitations", "limitation",
    "paper checklist", "neurips paper checklist", "checklist",
    "reproducibility checklist", "datasheet",
}

_NUM_HEAD_PAT = re.compile(r"^\s*(\d+(?:\.\d+){0,3})\.?\s+(.+?)\s*$")
_LETTER_HEAD_PAT = re.compile(r"^\s*([A-Z](?:\.\d+)*)\.?\s+(.+?)\s*$")
_PAGE_NUM_PAT = re.compile(r"^\s*\d+\s*$")
_FIG_TABLE_PAT = re.compile(r"^\s*(?:Figure|Fig\.|Table|Algorithm|Listing)\s*S?\d+", re.IGNORECASE)
_YEAR_PAT = re.compile(r"\b(?:19|20)\d{2}\b")

_TITLE_STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "in", "for", "to", "with",
    "on", "at", "by", "from", "as", "via", "vs", "is", "are", "be",
    "no", "not", "but", "into", "over", "under",
}

_SENTENCE_STARTERS = {
    "while", "if", "when", "where", "although", "since", "because",
    "however", "moreover", "furthermore", "thus", "therefore",
    "given", "let", "consider", "first", "next", "finally", "once",
    "even", "despite", "though", "after", "before", "during",
    "in", "on", "at", "by", "for", "with", "to", "from", "as",
    "this", "these", "those", "it", "we", "our", "they",
}

_UNNUMBERED_OK_HEADERS = {"abstract"}

_CORE_UNNUMBERED_MAJOR_HEADERS = {
    "abstract", "introduction",
    "background", "preliminaries", "preliminary",
    "related work", "related works", "prior work",
    "method", "methods", "methodology",
    "data", "dataset", "datasets",
    "experiment", "experiments", "evaluation",
    "result", "results", "discussion",
    "conclusion", "conclusions", "references",
    "acknowledgment", "acknowledgments", "acknowledgement", "acknowledgements",
}


def _strip_numbering(header: str) -> str:
    h = re.sub(r"^\s*(?:\d+(?:\.\d+){0,3}|[A-Z](?:\.\d+)*)\.?\s+", "", header).strip()
    return re.sub(r"\s+", " ", h)


def _is_checklist_header(header: str) -> bool:
    return "checklist" in _strip_numbering(header).lower()


def _classify_section(header: str, parent_category: str = "content") -> str:
    h = _strip_numbering(header).lower()
    if not h:
        return parent_category if parent_category != "content" else "content"
    if h in _SKIP_HEADERS:
        return "_skip"
    if h in _SECTION_CATEGORIES:
        return _SECTION_CATEGORIES[h]
    for kw, cat in _SECTION_CATEGORIES.items():
        if re.search(rf"\b{re.escape(kw)}\b", h):
            return cat
    for kw in _SKIP_HEADERS:
        if h.startswith(kw):
            return "_skip"
    return parent_category if parent_category != "content" else "content"


def _is_title_case(text: str) -> bool:
    tokens = text.split()
    n_cap, n_violation = 0, 0
    for tok in tokens:
        clean = re.sub(r"^[^A-Za-z]+", "", tok)
        if not clean:
            continue
        if clean[0].isupper():
            n_cap += 1
        elif clean.lower() in _TITLE_STOPWORDS:
            continue
        else:
            n_violation += 1
    return n_cap >= 1 and n_violation == 0


def _looks_like_header_text(text: str, short_len: int = 60) -> bool:
    s = text.strip()
    if not s or not s[0].isalpha() or not s[0].isupper():
        return False
    first_word = s.split()[0].lower().rstrip(",.;:")
    if first_word in _SENTENCE_STARTERS:
        return False
    if len(s) <= short_len:
        if "," in s and any(w in s.lower().split() for w in ["which", "that", "where", "when"]):
            return False
        return True
    return _is_title_case(s)


def _has_real_word(text: str, min_len: int = 4) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    return any(len(w) >= min_len and w.lower() not in _TITLE_STOPWORDS for w in words)


def _looks_like_bib_entry(text: str) -> bool:
    if "(cited on page" in text.lower():
        return True
    if _YEAR_PAT.search(text) and text.count(",") >= 2:
        return True
    return False


def _estimate_body_font_size(font_sizes):
    rounded = [round(float(s), 1) for s in font_sizes if s and float(s) > 0]
    prose_like = [s for s in rounded if 8.5 <= s <= 11.5]
    if prose_like:
        return Counter(prose_like).most_common(1)[0][0]
    return Counter(rounded).most_common(1)[0][0] if rounded else 10.0


def _has_heading_style(font_size, body_size, is_bold):
    return bool(is_bold) or (body_size > 0 and font_size >= body_size * 1.08)


def _has_major_heading_style(font_size, body_size, is_bold):
    return bool(is_bold) and body_size > 0 and font_size >= body_size * 1.15


def _has_split_heading_title_style(font_size, body_size, is_bold):
    return bool(is_bold) and body_size > 0 and font_size >= body_size * 1.02


def _is_unnumbered_major_header(text, font_size, body_size, is_bold):
    h_low = text.lower().rstrip(".").strip()
    if h_low in _UNNUMBERED_OK_HEADERS:
        return True
    if not _has_heading_style(font_size, body_size, is_bold):
        return False
    if h_low in _CORE_UNNUMBERED_MAJOR_HEADERS:
        return _has_major_heading_style(font_size, body_size, is_bold)
    if h_low in _SECTION_CATEGORIES or h_low in _SKIP_HEADERS:
        return _has_major_heading_style(font_size, body_size, is_bold)
    cls = _classify_section(text)
    return (cls != "content"
            and _has_major_heading_style(font_size, body_size, is_bold)
            and _looks_like_header_text(text, short_len=90))


def _merge_split_numbered_headings(all_lines, body_size):
    merged = []
    i = 0
    while i < len(all_lines):
        text, size, bold = all_lines[i]
        t = text.strip().rstrip(".")
        if (
            (
                (re.fullmatch(r"\d{1,2}", t) and 1 <= int(t) <= 30)
                or re.fullmatch(r"[A-Z]", t)
            )
            and _has_major_heading_style(size, body_size, bold)
            and i + 1 < len(all_lines)
        ):
            next_text, next_size, next_bold = all_lines[i + 1]
            nt = next_text.strip()
            if (
                nt
                and len(nt) <= 120
                and not nt.isdigit()
                and not _NUM_HEAD_PAT.match(nt)
                and _has_real_word(nt)
                and _has_split_heading_title_style(next_size, body_size, next_bold)
                and not _looks_like_bib_entry(nt)
            ):
                merged.append((f"{t}. {nt}", max(size, next_size), bool(bold or next_bold)))
                i += 2
                continue
        merged.append((text, size, bold))
        i += 1
    return merged


def _detect_header_kind(text, font_size, body_size, is_bold):
    t = text.strip()
    if not t or len(t) > 120:
        return None
    if _FIG_TABLE_PAT.match(t):
        return None
    if t.endswith((",", ";", "!", "?")):
        return None
    if t.endswith(".") and len(t) > 60:
        return None
    if len(t) <= 2 or t.isdigit():
        return None
    if any(ch in t for ch in "@∗†‡§"):
        return None
    if _looks_like_bib_entry(t):
        return None
    if re.search(r"[=<>≤≥≈≠]", t) or t.count(":") >= 2:
        return None

    if re.fullmatch(r"[A-Z]\.\d+(?:\.\d+)*\.?", t):
        return "subsection"

    m = _NUM_HEAD_PAT.match(t)
    if m:
        num_str, title = m.group(1), m.group(2).strip()
        if not title:
            return None
        if "." in num_str:
            if _has_real_word(title):
                return "subsection"
            return None
        try:
            main_num = int(num_str)
        except ValueError:
            return None
        if not (1 <= main_num <= 30):
            return None
        if not _has_real_word(title):
            return None
        title_cls = _classify_section(title)
        if re.search(r"\d", title) and title_cls == "content":
            return None
        if not (_is_title_case(title) or _looks_like_header_text(title, short_len=90)):
            return None
        return "major"

    lm = _LETTER_HEAD_PAT.match(t)
    if lm:
        letter_num, title = lm.group(1), lm.group(2).strip()
        if "." in letter_num:
            return "subsection" if _has_real_word(title) else None
        if not _has_major_heading_style(font_size, body_size, is_bold):
            return None
        if not _has_real_word(title):
            return None
        return "major"

    if _is_unnumbered_major_header(t, font_size, body_size, is_bold):
        return "major"
    return None


def parse_pdf(pdf_path: str) -> dict:
    doc = fitz.open(pdf_path)
    n_pages = doc.page_count

    all_lines = []
    font_sizes = []
    for i in range(n_pages):
        page = doc[i]
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", []) or []
                if not spans:
                    continue
                texts, sizes, bold_n = [], [], 0
                for sp in spans:
                    texts.append(sp.get("text", ""))
                    sizes.append(float(sp.get("size", 0)))
                    if int(sp.get("flags", 0)) & 16:
                        bold_n += 1
                line_text = "".join(texts).strip()
                if not line_text:
                    continue
                avg_size = sum(sizes) / len(sizes) if sizes else 0.0
                is_bold = bold_n >= max(1, len(spans) // 2)
                all_lines.append((line_text, avg_size, is_bold))
                font_sizes.append(round(avg_size, 1))
    doc.close()

    if not all_lines:
        return {"paper_id": Path(pdf_path).stem, "num_pages": n_pages, "sections": [], "total_chars": 0}

    body_size = _estimate_body_font_size(font_sizes)
    all_lines = _merge_split_numbered_headings(all_lines, body_size)

    sections, current = [], None
    skip_mode = False
    pre_abstract_mode = True
    last_section_num = 0
    checklist_mode = False

    for text, size, bold in all_lines:
        if _PAGE_NUM_PAT.match(text) and len(text) <= 4:
            continue
        if checklist_mode and current is not None:
            current["body"] = (current["body"] + " " + text).strip() if current["body"] else text
            continue

        kind = _detect_header_kind(text, size, body_size, bold)
        if kind == "skip":
            kind = "major"
        if kind in ("subsection", "sibling"):
            continue

        if kind == "major":
            cls = _classify_section(text)
            if cls == "_skip":
                cls = "content"
            m = _NUM_HEAD_PAT.match(text.strip())
            if m and "." not in m.group(1):
                try:
                    hn = int(m.group(1))
                except ValueError:
                    hn = None
                if hn is not None:
                    last_section_num = hn
            if pre_abstract_mode:
                if cls == "abstract" or (m and "." not in m.group(1)):
                    pre_abstract_mode = False
                else:
                    continue
            skip_mode = False
            checklist_mode = _is_checklist_header(text)
            current = {"category": cls, "header": text.strip(), "body": "", "level": 1}
            sections.append(current)
            continue

        if pre_abstract_mode or skip_mode or current is None:
            continue
        current["body"] = (current["body"] + " " + text).strip() if current["body"] else text

    if not sections:
        current = None
        skip_mode = False
        last_section_num = 0
        checklist_mode = False
        for text, size, bold in all_lines:
            if _PAGE_NUM_PAT.match(text) and len(text) <= 4:
                continue
            if checklist_mode and current is not None:
                current["body"] = (current["body"] + " " + text).strip() if current["body"] else text
                continue
            kind = _detect_header_kind(text, size, body_size, bold)
            if kind == "skip":
                kind = "major"
            if kind in ("subsection", "sibling"):
                continue
            if kind == "major":
                cls = _classify_section(text)
                if cls == "_skip":
                    cls = "content"
                skip_mode = False
                checklist_mode = _is_checklist_header(text)
                current = {"category": cls, "header": text.strip(), "body": "", "level": 1}
                sections.append(current)
                continue
            if skip_mode or current is None:
                continue
            current["body"] = (current["body"] + " " + text).strip() if current["body"] else text

    sections = [s for s in sections if s["body"].strip() and len(s["body"]) >= 60]
    return {
        "paper_id": Path(pdf_path).stem,
        "num_pages": n_pages,
        "sections": sections,
        "total_chars": sum(len(s["body"]) for s in sections),
    }


def split_sentences(text: str):
    raw = nltk.sent_tokenize(text)
    out, buf = [], ""
    for s in raw:
        s = s.strip()
        if not s:
            continue
        if buf:
            buf += " " + s
            if len(buf) >= 100:
                out.append(buf)
                buf = ""
        elif len(s) < 100:
            buf = s
        else:
            out.append(s)
    if buf:
        if out:
            out[-1] += " " + buf
        else:
            out.append(buf)
    return out


@dataclass
class Chunk:
    chunk_id: int
    text: str
    section: str


def merge_small_chunks(chunks, min_size: int = 400, max_merge: int = 1800):
    out = []
    for ch in chunks:
        if (out and out[-1].section == ch.section
                and len(out[-1].text) < min_size
                and len(out[-1].text) + 1 + len(ch.text) <= max_merge):
            out[-1] = Chunk(out[-1].chunk_id, out[-1].text + " " + ch.text, out[-1].section)
            continue
        out.append(ch)
    for i, c in enumerate(out):
        c.chunk_id = i
    return out


def build_chunks(parsed: dict, target_chars: int, max_chars: int, merge_min: int, sent_overlap: int = 2):
    chunks = []
    for sec in parsed["sections"]:
        sentences = split_sentences(sec["body"])
        if not sentences:
            continue
        cur = []
        for sent in sentences:
            cur_text = " ".join(cur + [sent])
            if cur and len(cur_text) > target_chars:
                chunks.append(Chunk(len(chunks), " ".join(cur), sec["category"]))
                ov = max(0, len(cur) - sent_overlap)
                cur = cur[ov:] + [sent]
            else:
                cur.append(sent)
        if cur:
            chunks.append(Chunk(len(chunks), " ".join(cur), sec["category"]))
    out = []
    for ch in chunks:
        if len(ch.text) <= max_chars:
            out.append(ch)
        else:
            t = ch.text
            mid = len(t) // 2
            sp = t.rfind(". ", mid - 100, mid + 100)
            sp = sp if sp != -1 else mid
            out.append(Chunk(len(out), t[:sp + 1].strip(), ch.section))
            out.append(Chunk(len(out), t[sp + 1:].strip(), ch.section))
    for i, ch in enumerate(out):
        ch.chunk_id = i
    out = merge_small_chunks(out, min_size=merge_min, max_merge=max(max_chars, merge_min * 2))
    return out
