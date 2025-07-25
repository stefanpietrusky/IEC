"""
title: IEC V2 [INDIVIDUAL EDUCATIONAL CHATBOT]
author: stefanpietrusky
author_url: https://downchurch.studio/
version: 1.0
"""

import os
import subprocess
import re
import requests
import fitz
import tiktoken
import logging
import asyncio
import uuid
import edge_tts
import json
import faiss
import numpy as np
from bs4 import BeautifulSoup
from collections import defaultdict
from duckduckgo_search import DDGS
from datetime import datetime
from readability import Document
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, Response, current_app, send_from_directory, url_for

index = None
metadatas = []
all_chunks = []
all_embeddings = None

OLLAMA_BASE      = "http://localhost:11434/api"
OLLAMA_EMBED_MOD = "nomic-embed-text"

app = Flask(__name__)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DATA_DIR = "extracted_texts"
os.makedirs(DATA_DIR, exist_ok=True)

CONV_ROOT = "conversations"
os.makedirs(CONV_ROOT, exist_ok=True)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

TOKEN_LIMIT = 131072
token_limit = TOKEN_LIMIT
enc = tiktoken.get_encoding("gpt2")

def load_faiss_index():
    global index, metadatas, all_chunks, all_embeddings
    if os.path.exists("rag_index.faiss") and os.path.exists("rag_meta.json"):
        index = faiss.read_index("rag_index.faiss")
        with open("rag_meta.json", "r", encoding="utf-8") as f:
            metadatas = json.load(f)
        with open("rag_chunks.json", "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        all_embeddings = index.reconstruct_n(0, index.ntotal)
    else:
        logging.warning("FAISS index or metadata not found!")

def answer_per_source(competence, src, content, question, selected_model):
    if competence == "Beginner":
        level_instr = "answer briefly and simply"
    elif competence == "Intermediate":
        level_instr = "answer in a balanced manner"
    else:
        level_instr = "explain in detail"

    prompt = (
        f"System: You are an intelligent assistant. {level_instr}. "
        f"Only use the following source code, and at the end, cite the source as (Source: {src}) to.\n\n"
        f"Source:\n{content}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return query_llama_via_cli(prompt, selected_model)


def clean_text_for_tts(text):
    import re
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  
    text = re.sub(r'\*([^*]+)\*', r'\1', text)    
    text = re.sub(r'__([^_]+)__', r'\1', text)  
    text = re.sub(r'_([^_]+)_', r'\1', text) 
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text) 
    text = re.sub(r'<[^>]+>', '', text)  
    text = re.sub(r'[#>\-]', '', text) 
    text = re.sub(r'[•●‣→⇒]', '', text)  
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_rag_prompt(retrieved, question, level_instr):
    sources_used = []
    for r in retrieved:
        src = r['metadata']['source']
        if src not in sources_used:
            sources_used.append(src)
    source_to_number = {src: idx+1 for idx, src in enumerate(sources_used)}

    context_parts = []
    for r in retrieved:
        src_number = source_to_number[r['metadata']['source']]
        context_parts.append(
            f"[Source {src_number}: {r['metadata']['source']}]\n{r['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    legend = "; ".join([f"Source {num} = {src}" for src, num in source_to_number.items()])

    beispiel = (
        "ANSWER FORMAT (please use this format exactly!):\n"
        "Section on the first source...\n"
        "(Source 1)\n\n"
        "Section on the second source...\n"
        "(Source 2)\n\n"
        "Section if both sources say the same thing...\n"
        "(Source 1, Source 2)\n\n"
        "Legende: Source 1 = extraction_20250725_160552.txt; Source 2 = extraction_20250725_160546.txt\n"
        "----\n"
        "Do NOT use any other format, NO continuous text without source references. Each paragraph MUST be followed by (source X). The legend MUST appear at the end."
    )

    instructions = (
        f"System: You are an assistant. {level_instr}.\n"
        "For each source, summarize the content in a separate paragraph and write the source at the END of the paragraph as (Source X). "
        "If different sources say the same thing, summarize them together and write both sources in parentheses. "
        "NO continuous text, ONLY individual paragraphs with source references. "
        "At the end: Write a caption. "
        "If no information is provided, respond precisely:\n"
        "'The sources contain no information on this question.'\n"
        + "\n\n" + beispiel
    )

    return (
        f"{instructions}\n\n"
        f"{context}\n\n"
        f"Frage: {question}\n"
        "Antwort:"
    )

def embed_via_ollama(texts: list[str]) -> list[list[float]]:
    payload = {"model": OLLAMA_EMBED_MOD, "input": texts}
    resp    = requests.post(f"{OLLAMA_BASE}/embed", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["embeddings"]

async def _generate_tts(text, voice, out_path):
    await edge_tts.Communicate(text=text, voice=voice).save(out_path)

def generate_tts_conv(text, conv_id, voice="en-GB-ThomasNeural"):
    text = clean_text_for_tts(text)
    conv_dir = os.path.join(CONV_ROOT, conv_id)
    os.makedirs(conv_dir, exist_ok=True)
    fname = f"{uuid.uuid4().hex}.mp3"
    out_path = os.path.join(conv_dir, fname)
    asyncio.run(_generate_tts(text, voice, out_path))
    return fname

def append_to_log(conv_id, entry: dict):
    conv_dir = os.path.join(CONV_ROOT, conv_id)
    log_path = os.path.join(conv_dir, "log.json")
    log = []
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
    log.append(entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def tokenize_text(text):
    return enc.encode(text)

def detokenize_text(tokens):
    return enc.decode(tokens)

def split_text_into_blocks(text, max_tokens):
    tokens = tokenize_text(text)
    for i in range(0, len(tokens), max_tokens):
        yield detokenize_text(tokens[i : i + max_tokens])

def get_readable_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        doc = Document(response.text)
        readable_html = doc.summary()
        soup = BeautifulSoup(readable_html, 'html.parser')
        readable_text = soup.get_text()
        return re.sub(r'\s+', ' ', readable_text).strip()
    except requests.exceptions.RequestException as e:
        logging.debug("Error fetching content from URL %s: %s", url, str(e))
        return f"Error fetching content: {str(e)}"

def build_faiss_index(extracted_contents_by_file: dict[str, str],
                      max_tokens_per_chunk: int = 1024) -> None:
    global index, metadatas, all_chunks, all_embeddings

    all_chunks, metadatas = [], []

    for fname, text in extracted_contents_by_file.items():
        if not text.strip():
            continue         
        chunks = list(split_text_into_blocks(text, max_tokens_per_chunk))
        if not chunks:
            continue     
        all_chunks.extend(chunks)
        metadatas.extend([{"source": fname} for _ in chunks])

    if not all_chunks:
        logging.warning("No chunks found – FAISS index will not be created.")
        index          = None
        all_embeddings = None
        return

    embeddings = embed_via_ollama(all_chunks)
    embs       = np.array(embeddings, dtype="float32")

    if embs.ndim == 1:
        embs = embs.reshape(1, -1)

    all_embeddings = embs

    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)

    faiss.write_index(index, "rag_index.faiss")
    with open("rag_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)

    with open("rag_chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

def extract_text_from_pdf(file):
    text = ""
    pdf_data = file.read()
    try:
        with fitz.open(stream=pdf_data, filetype="pdf") as pdf:
            for page in pdf:
                text += page.get_text()
    except Exception as e:
        logging.debug("Error reading PDF: %s", str(e))
        return f"Error reading PDF: {str(e)}"
    cleaned_text = re.sub(r'\s*\n\s*', ' ', text)
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
    return cleaned_text.strip()

def extract_content(url_input, pdf_files):
    all_content = ""
    if url_input:
        urls = [url.strip() for url in url_input.split(",")]
        for url in urls:
            if url.startswith("http"):
                all_content += get_readable_content(url) + "\n"
    if pdf_files:
        for pdf_file in pdf_files:
            pdf_text = extract_text_from_pdf(pdf_file)
            all_content += pdf_text + "\n"
    content = all_content.strip() or "No content extracted from the provided inputs."
    logging.debug("Extracted content: %s", content)
    return content

def check_internet_connection():
    try:
        requests.get("https://www.google.com", timeout=5)
        logging.debug("Internet connection available.")
        return True
    except requests.RequestException:
        logging.debug("No internet connection available.")
        return False

def perform_internet_search_multiple(query, num_results=3):
    logging.debug("Performing free DuckDuckGo search for query: %s", query)
    contents = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=10)
        logging.debug("Found %d search results", len(results) if results else 0)
        for i in range(1, min(num_results + 1, len(results))):
            result = results[i]
            link = result.get("href") or result.get("url", "")
            logging.debug("Using search result %d URL: %s", i + 1, link)
            content = get_readable_content(link)
            if content and len(content) > 100:
                contents.append(content)
                logging.debug("Extracted content from result %d (first 200 chars): %s", i + 1, content[:200])
            else:
                logging.debug("Content from result %d is too short or empty.", i + 1)
    except Exception as e:
        logging.debug("Exception during free DuckDuckGo search: %s", str(e))
    return contents

def summarise_text(text):
    prompt = f"Summarize the following text in a few sentences.:\n\n{text}"
    return query_llama_via_cli(prompt)

def generate_combined_response(competence_level, web_contents, user_question):
    if not competence_level or not user_question.strip():
        return "Please select a skill level and enter a question."
    
    summarized_contents = []
    for content in web_contents:
        if len(tokenize_text(content)) > 1000:  
            summary = summarise_text(content)
            summarized_contents.append(summary)
        else:
            summarized_contents.append(content)
    
    combined_content = "\n\n---\n\n".join(summarized_contents)
    prompt = f"System: You are an intelligent assistant. Please summarize the following information and then answer the question.\n\nInformations:\n{combined_content}\n\nQuestion: {user_question}"
    response = query_llama_via_cli(prompt)
    return response

def query_llama_via_cli(input_text, selected_model="llama3.2:latest"):
    try:
        result = subprocess.run(
            ["ollama", "run", selected_model],
            input=f"{input_text}\n",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            timeout=60 
        )
        if result.returncode != 0:
            logging.debug("Model error: %s", result.stderr.strip())
            return f"Error in the model request: {result.stderr.strip()}"
        response = re.sub(r'\x1b\[.*?m', '', result.stdout)
        return response.strip()
    except subprocess.TimeoutExpired:
        logging.debug("Model request timed out.")
        return "The model request timed out. Please try again."
    except Exception as e:
        logging.debug("Unexpected error during model request: %s", str(e))
        return f"An unexpected error has occurred: {str(e)}"

def select_relevant_chunks(chunks, question, threshold=0.5):
    relevant = []
    for i, chunk in enumerate(chunks):
        prompt = (
            f"System: You are a helper.\n"
            f"Question: {question}\n\n"
            f"Section #{i+1}:\n{chunk}\n\n"
            "Is this section RELEVANT to answering the question?"
            "Answer only with 'yes' or 'no'."
        )
        resp = query_llama_via_cli(prompt).lower()
        if resp.startswith("ja"):
            relevant.append(chunk)
    return relevant

def select_chunks_within_budget(chunks, max_tokens):
    selected = []
    total = 0
    for chunk in chunks:
        tok = len(tokenize_text(chunk))
        if total + tok > max_tokens:
            break
        selected.append(chunk)
        total += tok
    return selected

def generate_responses_from_blocks(competence_level, extracted_contents_by_file, user_question):
    if not competence_level or not user_question.strip():
        return "Please select a skill level and enter a question."

    chunk_size = 4096
    chunks = []
    for fname, text in extracted_contents_by_file.items():
        header = f"### Source: {fname}\n"
        for block in split_text_into_blocks(header + text, chunk_size):
            chunks.append(block)

    all_relevant = []
    for fname, text in extracted_contents_by_file.items():
        header = f"### source: {fname}\n"
        source_chunks = list(split_text_into_blocks(header + text, chunk_size))

        rel = select_relevant_chunks(source_chunks, user_question)
        if not rel:
            rel = [source_chunks[0]]
        all_relevant.extend(rel)

    selected_chunks = select_chunks_within_budget(all_relevant, token_limit)
    combined_context = "\n\n---\n\n".join(selected_chunks)

    if competence_level == "Beginner":
        level_instr = "answer briefly and simply"
    elif competence_level == "Intermediate":
        level_instr = "answer in a balanced manner at a moderate level"
    else:
        level_instr = "explain in detail at an advanced level"

    prompt = (
        f"System: You are an intelligent assistant. {level_instr}. "
        "Answer based solely on the following contexts. "
        "If there is no answer in the sources, answer exactly: "
        "'The sources contain no information on this question.'\n\n"
        f"Contexts:\n{combined_context}\n\n"
        f"Question: {user_question}\n"
        "Answer:"
    )

    response = query_llama_via_cli(prompt).strip()
    fallback = "The sources contain no information on this question."
    return fallback if fallback in response else response

def generate_from_multiple_sources(competence_level, extracted_contents_by_file, user_question):
    if not competence_level or not user_question.strip():
        return "Please select a skill level and enter a question."

    summaries = []
    for fname, text in extracted_contents_by_file.items():
        summary = summarise_text(text)
        summaries.append(f"### Summary {fname}:\n{summary}")

    merged = "\n\n---\n\n".join(summaries)

    if competence_level == "Beginner":
        level_instr = "answer briefly and simply"
    elif competence_level == "Intermediate":
        level_instr = "answer in a balanced manner at a moderate level"
    else:
        level_instr = "explain in detail at an advanced level"

    prompt = (
        f"System: You are an intelligent assistant. {level_instr}. "
        "Summarize the following source summaries and answer the question:\n\n"
        f"{merged}\n\n"
        f"Question: {user_question}\n"
        "Answer:"
    )

    return query_llama_via_cli(prompt)

def generate_response_from_extracted_content(competence_level, extracted_content, url_input, pdf_files, user_question):
    if not competence_level or not user_question.strip():
        return "Please select a skill level and enter a question."
    
    if extracted_content.strip() == "" or extracted_content.strip() == "No content extracted from the provided inputs.":
        logging.debug("No extracted content provided; using fallback mechanism.")
        if not url_input.strip() and (not pdf_files or len(pdf_files) == 0):
            if check_internet_connection():
                web_contents = perform_internet_search_multiple(user_question, num_results=3)
                if web_contents:
                    extracted_content = "\n\n---\n\n".join(web_contents)
                    logging.debug("Web search yielded content. Using it for the answer.")
                else:
                    logging.debug("Web search returned no useful content. Using internal model knowledge.")
                    return query_llama_via_cli(f"Frage: {user_question}")
            else:
                logging.debug("No internet connection. Using internal model knowledge.")
                return query_llama_via_cli(f"Frage: {user_question}")
        else:
            return "No extracted content available."
    
    return generate_responses_from_blocks(competence_level, extracted_content, user_question)

def clear_extracted_content():
    return ""

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <script src="https://cdn.jsdelivr.net/npm/markdown-it/dist/markdown-it.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/dompurify/dist/purify.min.js"></script>
  <title>IEC V2</title>
  <link rel="stylesheet" href="/styles.css" />
</head>
<body>
  <div class="container">
    <h1>IEC V2</h1>
    
    <div class="section">
      <div class="model-select-container" style="margin-top:18px;">
        <h2>Select model</h2>
        <select id="model-select" style="width: 100%; padding: 10px; border: 3px solid #262626; border-radius: 5px; font-size: 1em;">
          <option value="">Models are loading...</option>
        </select>
      </div>

      <h2>Extract content</h2>
      <label for="url-input">Enter URLs (separated by commas):</label>
      <input type="text" id="url-input" name="urls" placeholder="https://example.com" />
      
      <div class="pdf-upload-container" id="pdf-drop-zone">
        <label for="pdf-input" class="custom-file-upload" id="pdf-upload-btn">
          Upload PDF files
        </label>
        <input type="file" id="pdf-input" name="pdfs" multiple accept=".pdf" />
        <div id="pdf-filename" class="pdf-filename"></div>
      </div>
      
      <div class="buttons">
        <button id="extract-btn">Extract content</button>
        <button id="show-extractions-btn" style="display:none;">Show content</button>
      </div>
      
      <div id="extractions-overview" style="display:none; margin-top:20px;">
        <ul id="extractions-list">
        </ul>
      </div>
      
      <div id="extraction-detail" style="display:none; margin-top:20px;">
        <pre id="extraction-content" class="result-container"></pre>
        <div class="detail-buttons">
          <button id="back-to-overview">Back to overview</button>
          <button id="delete-current-extraction">Delete content</button>
        </div>
    </div>
    </div> 

    <div class="section">
      <h2>Select response level</h2>
      <div class="button-group" id="competence-group">
        <button class="competence-button" data-level="Beginner">Beginner</button>
        <button class="competence-button selected" data-level="Intermediate">Intermediate</button>
        <button class="competence-button" data-level="Advanced">Advanced</button>
      </div>
      
      <label for="question-input"><h2>Ask a question</h2></label>
      <input type="text" id="question-input" placeholder="Your question" />
      <div class="buttons">
        <button id="ask-btn">Ask a question</button>
      </div>
    </div>
    
  <div class="section chat-section">
    <h2>Chat history</h2>
    <div id="chat-container" class="chat-container"></div>
  </div>

    <div id="spinner" class="spinner" style="display: none;"></div>
  </div>
  <script src="/script.js"></script>
</body>
</html>
"""

CSS_CONTENT = """
body {
  font-family: Arial, sans-serif;
  background-color: #f4f4f4;
  margin: 0;
  padding: 20px;
}
.container {
  width: 90%;
  max-width: 800px;
  margin: auto;
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 0 10px rgba(0,0,0,0.1);
  border: 3px solid #262626;
}
.detail-buttons {
  display: flex;
  justify-content: center;
  gap: 12px;
  margin-top: 16px; 
}
h1, h2, h3 {
  text-align: center;
  color: #333;
}
label {
  display: block;
  margin-top: 10px;
}
input[type="text"] {
  width: 100%;
  padding: 10px;
  margin: 10px 0;
  border: 3px solid #262626;
  border-radius: 5px;
  box-sizing: border-box;
  font-size: 16px;
}
.buttons, .button-group {
  text-align: center;
  margin: 10px 0;
}
button {
  padding: 10px 20px;
  margin: 10px 5px;
  border-radius: 5px;
  background-color: #ffffff;
  border: 3px solid #262626;
  color: #262626;
  cursor: pointer;
  font-size: 1em;
  transition: background-color 0.3s ease;
}
button:hover {
  background-color: #262626;
  border: 3px solid #262626;
  color: #ffffff;
}
.competence-button {
  padding: 10px 20px;
  margin: 0 5px;
  border: 3px solid #262626;
  background-color: #ffffff;
  color: #262626;
  border-radius: 5px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}
.competence-button.selected,
.competence-button:hover {
  background-color: #262626;
  border: 3px solid #262626;
  color: #ffffff;
}
.result-container {
  background-color: #ffffff;
  border: 3px solid #262626;
  color: #262626;
  padding: 15px;
  border-radius: 8px;
  margin-top: 20px;
  white-space: normal;
  text-align: justify;
  font-family: Arial, sans-serif;
  max-height: 300px;
  overflow-y: auto;
  box-sizing: border-box;
  padding-right: 8px;
  scrollbar-gutter: stable;
}
.spinner {
  border: 8px solid #262626;
  border-top: 8px solid #00B0F0;
  border-radius: 50%;
  width: 50px;
  height: 50px;
  animation: spin 1s linear infinite;
  margin: 20px auto;
}
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
.section {
  margin-bottom: 40px;
}
.pdf-upload-container {
  border: 3px dashed #262626;
  padding: 20px;
  border-radius: 5px;
  text-align: center;
  margin-top: 10px;
  position: relative;
  transition: background-color 0.3s ease, border-color 0.3s ease;
}
.pdf-upload-container.dragover {
  border-color: #00B0F0;
  background-color: #89E0FF;
}
.pdf-upload-container input[type="file"] {
  display: none;
}
.custom-file-upload {
  display: inline-block;
  padding: 10px 20px;
  border: 3px solid #262626;
  background-color: #262626;
  color: white;
  border-radius: 5px;
  cursor: pointer;
  font-size: 1em;
  transition: background-color 0.3s ease;
}
.custom-file-upload:hover {
  background-color: #ffffff;
  border: 3px solid #262626;
  color: #262626;
}
.pdf-filename {
  margin-top: 10px;
  font-size: 0.9em;
  color: #333;
}
#extractions-overview {
  margin-top: 20px;
  max-height: 300px;
  overflow-y: auto;
  scrollbar-gutter: stable both-edges;
}
#extractions-list {
  list-style: none; 
  margin: 0;
  padding: 0; 
}
#extractions-list li {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  box-sizing: border-box;
  margin: 8px 0;
  padding: 8px;
  border: 3px solid #262626;
  border-radius: 5px;
  list-style: none;
}
#extractions-list li button {
  float: none;
  margin-left: 16px;
}
#extractions-list li .extraction-checkbox {
  width: 18px;
  height: 18px;
  margin-right: 8px;
  accent-color: #262626;
}
#chat-container:empty {
  display: none;
}
.chat-section {
  margin-top: 40px;
  display: none;
}
.chat-section:empty > h2,
.chat-section:empty > #chat-container {
  display: none;
}
.chat-container {
  max-height: 400px;
  overflow-y: auto;
  padding: 10px;
  border: 3px solid #262626;
  border-radius: 8px;
  background: #fff;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.chat-message {
  display: flex;
  align-items: flex-start;
}
.chat-message.user .bubble {
  background: #00B0F0;
  align-self: flex-end;
}
.chat-message.bot .bubble {
  background: #f1f1f1;
  align-self: flex-start;
}
.bubble {
  padding: 10px 14px;
  border-radius: 8px;
  border: 2px solid #262626;
  color: #262626;
  max-width: 75%;
  position: relative;
  font-size: 0.95em;
  line-height: 1.4;
}
.bubble .tts-control {
  position: absolute;
  bottom: 4px;
  right: 6px;
  cursor: pointer;
  font-size: 1.1em;
}
.delete-btn {
    background-color: #ffffff;
    border: 3px solid #f44336;
    color: #f44336;
    transition: background 0.2s, color 0.2s;
}
.delete-btn:hover {
    background-color: #f44336;
    border: 3px solid #f44336;
    color: #ffffff;
}
input:focus {
  outline: none;   
  border: 3px solid #00B0F0;   
}
"""

JS_CONTENT = """
document.addEventListener('DOMContentLoaded', function() {

  function showAlert(msg) {
    alert(msg);
  }

  const extractBtn = document.getElementById('extract-btn');
  const urlInput = document.getElementById('url-input');
  const pdfInput = document.getElementById('pdf-input');
  const pdfDropZone = document.getElementById('pdf-drop-zone');
  const pdfUploadBtn = document.getElementById('pdf-upload-btn');
  const pdfFilename = document.getElementById('pdf-filename');
  const askBtn = document.getElementById('ask-btn');
  const questionInput = document.getElementById('question-input');
  const responseDiv = document.getElementById('response');
  const spinner = document.getElementById('spinner');
  const competenceButtons = document.querySelectorAll('.competence-button');
  const showBtn = document.getElementById('show-extractions-btn');
  const overview = document.getElementById('extractions-overview');
  const list = document.getElementById('extractions-list');
  const detail = document.getElementById('extraction-detail');
  const detailContent = document.getElementById('extraction-content');
  const backBtn = document.getElementById('back-to-overview');
  const deleteCurrentBtn = document.getElementById('delete-current-extraction');
  const chatContainer = document.getElementById('chat-container');
  const chatSection = document.querySelector('.chat-section');
  const playIcon = `<svg width="22" height="22" viewBox="0 0 20 20" style="vertical-align:middle"><polygon points="4,3 18,10 4,17" fill="#262626"/></svg>`;
  const pauseIcon = `<svg width="22" height="22" viewBox="0 0 20 20" style="vertical-align:middle"><rect x="4" y="3" width="4" height="14" fill="#262626"/><rect x="12" y="3" width="4" height="14" fill="#262626"/></svg>`;
  const modelSelect = document.getElementById('model-select');

  let selectedCompetence = "Intermediate";
  let currentExtractionFilename = null;
  let messages = [];

  let conversationId = localStorage.getItem('conversationId');
  if (!conversationId) {
    conversationId = 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('conversationId', conversationId);
  }

  fetch('/list_models')
    .then(res => res.json())
    .then(models => {
      modelSelect.innerHTML = '';
      if (models.length === 0) {
        modelSelect.innerHTML = '<option value="">No models found</option>';
      } else {
        modelSelect.innerHTML = models.map(m => `<option value="${m}">${m}</option>`).join('');
      }
    })
    .catch(err => {
      modelSelect.innerHTML = '<option value="">Error loading models</option>';
    });

  document.getElementById('extract-btn').addEventListener('click', async () => {
    const urlInput = document.getElementById('url-input').value.trim();
    const pdfInput = document.getElementById('pdf-input');
    const formData = new FormData();

    formData.append('urls', urlInput);

    for (let file of pdfInput.files) {
      formData.append('pdfs', file);
    }

    spinner.style.display = 'block';

    try {
      const res  = await fetch('/extract_content', { method: 'POST', body: formData });
      const text = await res.text();
      console.log('Raw /extract_content response:', text);
      const data = JSON.parse(text);
      spinner.style.display = 'none';

      if (data.content) {
        overview.style.display    = 'none';
        detail.style.display      = 'block';
        detailContent.textContent = data.content;
        showBtn.style.display     = 'inline-block';
        refreshExtractionList();
      } else {
        alert("No content found.");
      }
    } catch (err) {
      spinner.style.display = 'none';
      console.error("Parsing or network error:", err);
      alert("Error during extraction. Check the console.");
    }
  });

  competenceButtons.forEach(button => {
    button.addEventListener('click', function() {
      competenceButtons.forEach(btn => btn.classList.remove('selected'));
      this.classList.add('selected');
      selectedCompetence = this.getAttribute('data-level');
    });
  });

  pdfDropZone.addEventListener('dragover', function(e) {
    e.preventDefault();
    e.stopPropagation();
    pdfDropZone.classList.add('dragover');
  });
  
  pdfDropZone.addEventListener('dragleave', function(e) {
    e.preventDefault();
    e.stopPropagation();
    pdfDropZone.classList.remove('dragover');
  });
  
  pdfDropZone.addEventListener('drop', function(e) {
    e.preventDefault();
    e.stopPropagation();
    pdfDropZone.classList.remove('dragover');
    let dt = e.dataTransfer;
    let files = dt.files;
    if (files.length > 0) {
      let dataTransfer = new DataTransfer();
      for (let i = 0; i < files.length; i++) {
        dataTransfer.items.add(files[i]);
      }
      pdfInput.files = dataTransfer.files;
      updatePdfFilenameDisplay();
      updatePdfUploadButton();
    }
  });

  pdfInput.addEventListener('change', function() {
    updatePdfFilenameDisplay();
    updatePdfUploadButton();
  });

  pdfUploadBtn.addEventListener('click', function(e) {
    if (pdfInput.files.length > 0) {
      e.preventDefault();
      pdfInput.value = "";
      updatePdfFilenameDisplay();
      updatePdfUploadButton();
    } else {
      pdfInput.click();
    }
  });

  function updatePdfFilenameDisplay() {
    let fileNames = [];
    for (let i = 0; i < pdfInput.files.length; i++) {
      fileNames.push(pdfInput.files[i].name);
    }
    pdfFilename.innerText = fileNames.join(', ');
  }

  function updatePdfUploadButton() {
    if (pdfInput.files.length > 0) {
      pdfUploadBtn.innerText = "Delete files";
    } else {
      pdfUploadBtn.innerText = "Upload PDF files";
    }
  }

  askBtn.addEventListener('click', async function() {
    const question = questionInput.value.trim();
    if (!question) return;

    addMessage('user', question);
    questionInput.value = '';
    spinner.style.display = 'block';

    const checkedBoxes  = document.querySelectorAll('.extraction-checkbox:checked');
    const selectedFiles = Array.from(checkedBoxes).map(cb => cb.dataset.filename);

    try {
      const res  = await fetch('/ask_question', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation_id: conversationId,
          competence_level: selectedCompetence,
          question: question,
          selected_extractions: selectedFiles,
          selected_model: modelSelect.value
        })
      });
      const data = await res.json();
      spinner.style.display = 'none';

      const cleaned = data.response
        .replace(/\x1b\[[0-9;]*m/g, '')
        .replace(/[\u200B-\u200D\uFEFF]/g, '');

      addMessage('bot', cleaned, data.audio_url);
    } catch (error) {
      spinner.style.display = 'none';
      console.error("Fehler:", error);
      alert("Error when asking the question.");
    }
  });

  showBtn.addEventListener('click', () => {
    const isVisible = overview.style.display === 'block';

    if (isVisible) {
      overview.style.display = 'none';
      detail.style.display   = 'none';
      showBtn.textContent    = 'Show content';
    } else {
      overview.style.display = 'block';
      detail.style.display   = 'none';
      showBtn.textContent    = 'Hide content';
      refreshExtractionList();
    }
  });

  backBtn.addEventListener('click', () => {
    detail.style.display = 'none';
    overview.style.display = 'block';
  });

  deleteCurrentBtn.addEventListener('click', () => {
    if (!currentExtractionFilename) return;
    fetch(`/delete_extraction/${encodeURIComponent(currentExtractionFilename)}`, { method: 'DELETE' })
      .then(r => {
        if (!r.ok) throw new Error("Deletion failed");
        return r.json();
      })
      .then(data => {
        showAlert("File successfully deleted.");  
        currentExtractionFilename = null;
        detail.style.display = 'none';
        overview.style.display = 'block';
        refreshExtractionList();
      })
      .catch(err => {
        console.error(err);
        showAlert("Error during deletion: " + err.message);
      });
  });

  function refreshExtractionList() {
    fetch('/list_extractions')
      .then(r => r.json())
      .then(files => {
        list.innerHTML = '';

        if (files.length) {
          overview.style.display   = 'block';
          detail.style.display     = 'none';
          showBtn.style.display    = 'inline-block';
          showBtn.textContent      = 'Hide content';
        } else {
          overview.style.display   = 'none';
          detail.style.display     = 'none';
          showBtn.style.display    = 'none';
        }

        files.forEach(f => {
          const li = document.createElement('li');
          li.classList.add('result-container');

          const cb = document.createElement('input');
          cb.type = 'checkbox';
          cb.classList.add('extraction-checkbox');
          cb.dataset.filename = f.name;
          cb.addEventListener('click', e => {
            e.stopPropagation();
          });

          const label = document.createElement('span');
          label.textContent = ` ${f.name} (${f.date})`;

          const del = document.createElement('button');
          del.textContent = 'Delete';
          del.classList.add('delete-btn');
          del.addEventListener('click', e => {
            e.stopPropagation();
            fetch(`/delete_extraction/${encodeURIComponent(f.name)}`, { method: 'DELETE' })
              .then(r => {
                if (!r.ok) throw new Error('Deletion failed');
                return r.json();
              })
              .then(() => {
                alert('File successfully deleted.');
                refreshExtractionList();
              })
              .catch(err => {
                console.error(err);
                alert('Error during deletion: ' + err.message);
              });
          });

          li.append(cb, label, del);

          li.addEventListener('click', () => {
            currentExtractionFilename = f.name;
            fetch(`/get_extraction/${encodeURIComponent(f.name)}`)
              .then(r => r.json())
              .then(data => {
                overview.style.display    = 'none';
                detail.style.display      = 'block';
                detailContent.textContent = data.content || data.error;
              });
          });

          list.appendChild(li);
        });

        if (overview.scrollHeight > overview.clientHeight) {
          overview.style.paddingRight = '16px';
        } else {
          overview.style.paddingRight = '0';
        }
      })
      .catch(err => {
        console.error('Error loading the extraction list:', err);
      });
  }

  function addMessage(role, text, audioUrl = null) {
    messages.push({ role, text, audioUrl });
    if (messages.length > 20) messages = messages.slice(-20);
    renderChat();
  }

  function renderChat() {
    if (messages.length > 0) {
      chatSection.style.display = 'block';
    } else {
      chatSection.style.display = 'none';
      return;
    }

    chatContainer.innerHTML = '';
    messages.forEach((msg) => {
      const wrapper = document.createElement('div');
      wrapper.classList.add('chat-message', msg.role);

      const bubble = document.createElement('div');
      bubble.classList.add('bubble');

      const label = document.createElement('strong');
      label.textContent = msg.role === 'user' ? 'Me: ' : 'KI: ';
      label.style.marginRight = '6px';

      const content = DOMPurify.sanitize(markdownit().render(msg.text));
      bubble.innerHTML = content;
      bubble.prepend(label);

      if (msg.role === 'bot' && msg.audioUrl) {
        const ttsBtn = document.createElement('span');
        ttsBtn.classList.add('tts-control');
        ttsBtn.innerHTML = playIcon; 
        bubble.appendChild(ttsBtn);

        const audio = document.createElement('audio');
        audio.src = msg.audioUrl;
        audio.preload = 'none';

        ttsBtn.addEventListener('click', () => {
          if (audio.paused) {
            audio.play();
            ttsBtn.innerHTML = pauseIcon;
          } else {
            audio.pause();
            ttsBtn.innerHTML = playIcon;
          }
        });

        audio.addEventListener('ended', () => {
          ttsBtn.innerHTML = playIcon;
        });

      } else if (msg.role === 'bot' && !msg.audioUrl) {
        const ttsBtn = document.createElement('span');
        ttsBtn.classList.add('tts-control');
        ttsBtn.innerHTML = playIcon;
        let utterance = null;

        ttsBtn.addEventListener('click', () => {
          if (!utterance) {
            utterance = new SpeechSynthesisUtterance(msg.text);
            const voice = speechSynthesis.getVoices().find(v => v.name.includes('Microsoft'));
            if (voice) utterance.voice = voice;
            speechSynthesis.speak(utterance);
            ttsBtn.innerHTML = pauseIcon;
            utterance.onend = () => {
              ttsBtn.innerHTML = playIcon;
              utterance = null;
            };
          } else if (speechSynthesis.paused) {
            speechSynthesis.resume();
            ttsBtn.innerHTML = pauseIcon;
          } else {
            speechSynthesis.pause();
            ttsBtn.innerHTML = playIcon;
          }
        });
        bubble.appendChild(ttsBtn);
      }

      wrapper.appendChild(bubble);
      chatContainer.appendChild(wrapper);
    });
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  refreshExtractionList();
});
"""

@app.route('/')
def home():
    return Response(HTML_CONTENT, mimetype='text/html')

@app.route('/styles.css')
def styles():
    return Response(CSS_CONTENT, mimetype='text/css')

@app.route('/script.js')
def script():
    return Response(JS_CONTENT, mimetype='application/javascript')

@app.route('/conversations/<conv_id>/<filename>')
def serve_conv_audio(conv_id, filename):
    directory = os.path.join(CONV_ROOT, conv_id)
    return send_from_directory(directory, filename)

@app.route('/list_models', methods=['GET'])
def list_models():
    try:
        result = subprocess.run(['ollama', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            return jsonify({"error": result.stderr.strip()}), 500
        lines = result.stdout.strip().split('\n')
        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        models = [m for m in models if not m.startswith("nomic-embed-text")]
        return jsonify(models)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/extract_content', methods=['POST'])
def extract_content_endpoint():
    if request.content_type.startswith('multipart/form-data'):
        url_input      = request.form.get("urls", "").strip()
        pdf_files      = request.files.getlist("pdfs")
        selected_model = request.form.get("selected_model", "llama3.2:latest")
    else:
        data           = request.get_json()
        url_input      = data.get("urls", "").strip()
        pdf_files      = []
        selected_model = data.get("selected_model", "llama3.2:latest")

    if not url_input and (not pdf_files or len(pdf_files) == 0):
        return jsonify({ "content": "" })

    extracted = extract_content(url_input, pdf_files)

    default_msg = "No content extracted from the provided inputs."
    if extracted.strip() and extracted.strip() != default_msg:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = os.path.join(DATA_DIR, f"extraction_{timestamp}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(extracted)

        all_files = {}
        for fname in os.listdir(DATA_DIR):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(DATA_DIR, fname)
            with open(path, encoding="utf-8") as f2:
                all_files[fname] = f2.read()

        build_faiss_index(all_files)

    return jsonify({ "content": extracted or "" })

@app.route('/ask_question', methods=['POST'])
def ask_question():
    global metadatas, all_chunks, all_embeddings

    data            = request.get_json()
    conv_id         = data.get("conversation_id")
    competence      = data.get("competence_level")
    question        = data.get("question", "").strip()
    selected_files  = data.get("selected_extractions", [])
    selected_model  = data.get("selected_model", "llama3.2:latest")

    if not selected_files:
        return jsonify({"response": "Please select at least one source."})

    source_contents = {}
    for src in selected_files:
        content_blocks = []
        src_idx = [i for i, m in enumerate(metadatas) if m["source"] == src]
        for i in src_idx:
            content_blocks.append(all_chunks[i])
        source_contents[src] = "\n\n".join(content_blocks[:5])  

    per_source_answers = []
    for src, content in source_contents.items():
        if not content.strip():
            continue
        ans = answer_per_source(competence, src, content, question, selected_model)
        per_source_answers.append(f"**Answer for {src}:**\n{ans.strip()}")

    if not per_source_answers:
        return jsonify({"response": "No relevant content found."})

    zusammenfassung_prompt = (
        "System: Summarize all of the following answers for each source into an overall view, "
        "and cite the sources as (Source: ...). "
        "If there are overlaps, summarize them; otherwise, distinguish them clearly.\n\n"
        "Responses per source:\n\n"
        + "\n\n---\n\n".join(per_source_answers) +
        "\n\nOverall response:"
    )
    final_answer = query_llama_via_cli(zusammenfassung_prompt)

    tts_fname = generate_tts_conv(final_answer, conv_id)
    audio_url = url_for('serve_conv_audio', conv_id=conv_id, filename=tts_fname)
    append_to_log(conv_id, {
        "timestamp": datetime.utcnow().isoformat()+"Z",
        "question":  question,
        "answer":    final_answer,
        "audio_file": tts_fname,
        "extractions": selected_files
    })

    return jsonify({
        "response":  final_answer,
        "audio_url": audio_url,
        "per_source_answers": per_source_answers 
    })

@app.route('/clear_extracted', methods=['POST'])
def clear_extracted():
    return jsonify({"content": ""})

@app.route('/list_extractions', methods=['GET'])
def list_extractions():
    files = []
    for fname in sorted(os.listdir(DATA_DIR), reverse=True):
        if fname.endswith(".txt"):
            path = os.path.join(DATA_DIR, fname)
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            files.append({
                "name": fname,
                "date": mtime.strftime("%Y-%m-%d %H:%M:%S")
            })
    return jsonify(files)

@app.route('/get_extraction/<filename>', methods=['GET'])
def get_extraction(filename):
    safe = secure_filename(filename)
    path = os.path.join(DATA_DIR, safe)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return jsonify({"content": f.read()})
    return jsonify({"error": "Not found"}), 404

@app.route('/delete_extraction/<filename>', methods=['DELETE'])
def delete_extraction(filename):
    safe = secure_filename(filename)
    path = os.path.join(DATA_DIR, safe)
    if os.path.isfile(path):
        os.remove(path)
        return jsonify({"status": "deleted"})
    return jsonify({"error": "Not found"}), 404

if __name__ == '__main__':
    load_faiss_index() 
    app.run(debug=True, host="0.0.0.0", port=5000)
