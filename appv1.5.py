"""
title: IEC V1.5 [INDIVIDUAL EDUCATIONAL CHATBOT]
author: stefanpietrusky
author_url: https://downchurch.studio/
version: 1.0
"""

from flask import Flask, request, jsonify, Response
import os
import subprocess
import re
import requests
from readability import Document
import fitz
from bs4 import BeautifulSoup
import tiktoken

app = Flask(__name__)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --------------------------
# Tokenization and help functions
# --------------------------
TOKEN_LIMIT = 131072
enc = tiktoken.get_encoding("gpt2")

def tokenize_text(text):
    return enc.encode(text)

def detokenize_text(tokens):
    return enc.decode(tokens)

def split_text_into_blocks(text, max_tokens):
    tokens = tokenize_text(text)
    for i in range(0, len(tokens), max_tokens):
        yield detokenize_text(tokens[i : i + max_tokens])

# --------------------------
# Functions for extracting content
# --------------------------
def get_readable_content(url):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        doc = Document(response.text)
        readable_html = doc.summary()
        soup = BeautifulSoup(readable_html, 'html.parser')
        readable_text = soup.get_text()
        return re.sub(r'\s+', ' ', readable_text).strip()
    except requests.exceptions.RequestException as e:
        return f"Error fetching content: {str(e)}"

def extract_text_from_pdf(file):
    text = ""
    pdf_data = file.read()
    try:
        with fitz.open(stream=pdf_data, filetype="pdf") as pdf:
            for page in pdf:
                text += page.get_text()
    except Exception as e:
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
    return all_content.strip() or "No content extracted from the provided inputs."

# --------------------------
# Functions for answering questions
# --------------------------
def query_llama_via_cli(input_text):
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.1"],
            input=f"{input_text}\n",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            timeout=60
        )
        if result.returncode != 0:
            return f"Error in the model request: {result.stderr.strip()}"
        response = re.sub(r'\x1b\[.*?m', '', result.stdout)
        return response.strip()
    except subprocess.TimeoutExpired:
        return "The model request timed out. Please try again."
    except Exception as e:
        return f"An unexpected error has occurred: {str(e)}"

def generate_responses_from_blocks(competence_level, extracted_content, user_question):
    if not competence_level or not user_question.strip():
        return "Please select a competence level and enter a question."
    if not extracted_content.strip():
        return "No content extracted on which the answer could be based."
    
    if competence_level == "Beginner":
        style_instruction = (
            "Respond in very simple words, avoid technical terms, and keep the answer short and concise. "
            "Explain the topic in a way that even laypersons can understand."
        )
    elif competence_level == "Intermediate":
        style_instruction = (
            "Provide a clear and detailed response, occasionally using technical terms and offering a balanced explanation. "
            "Explain the topic so that the reader gains a good understanding without it becoming too technical."
        )
    elif competence_level == "Advanced":
        style_instruction = (
            "Provide a highly detailed and technical response, use specialized terminology, and offer a comprehensive analysis. "
            "Explain the topic at a high, academic level."
        )
    else:
        style_instruction = "Respond based on the provided information."
    
    base_prompt = (
        f"System Instruction: {style_instruction}\n\n"
        "Note: The following information serves only as a reference. Please phrase the answer in your own words and consider the desired style.\n\n"
        "Provided Information (Excerpt):\n"
    )
    question_prompt = f"\n\nQuestion: {user_question}\n\nAnswer:"
    
    prompt_prefix_tokens = len(tokenize_text(base_prompt)) + len(tokenize_text(question_prompt))
    max_tokens_for_block = TOKEN_LIMIT - prompt_prefix_tokens

    responses = []
    for block in split_text_into_blocks(extracted_content, max_tokens_for_block):
        input_text = base_prompt + block + question_prompt
        response = query_llama_via_cli(input_text)
        responses.append(response)
    return "\n".join(responses)

def generate_response_from_extracted_content(competence_level, extracted_content, url_input, pdf_files, user_question):
    if not competence_level or not user_question.strip():
        return "Please select a competence level and enter a question."
    
    if not extracted_content.strip():
        if not url_input.strip() and not pdf_files:
            return "Please provide either a URL or upload a PDF file to proceed."
        else:
            return "No content extracted to base the answer on."
    
    return generate_responses_from_blocks(competence_level, extracted_content, user_question)

def clear_extracted_content():
    return ""

# --------------------------
# HTML, CSS and JavaScript for the front end
# --------------------------
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <title>IEC V1.5</title>
  <link rel="stylesheet" href="/styles.css">
</head>
<body>
  <div class="container">
    <h1>IEC V1.5</h1>
    
    <div class="section">
      <h2>Extract content</h2>
      <label for="url-input">Enter URLs (optional, comma-separated):</label>
      <input type="text" id="url-input" placeholder="https://example.com">
      
    <div class="pdf-upload-container" id="pdf-drop-zone">
    <label for="pdf-input" class="custom-file-upload" id="pdf-upload-btn">
        Upload PDF files
    </label>
    <input type="file" id="pdf-input" multiple accept=".pdf">
    <div id="pdf-filename" class="pdf-filename"></div>
    </div>
      
      <div class="buttons">
        <button id="extract-btn">Extract content</button>
        <button id="clear-btn">Delete extracted content</button>
      </div>
      
      <div>
        <label>
          <input type="checkbox" id="toggle-content" checked> Show extracted content
        </label>
      </div>
      
      <div id="extracted-content" class="result-container" style="display: none;"></div>
    </div>
    
    <div class="section">
      <h2>Select the level of the answer</h2>
      <div class="button-group" id="competence-group">
        <button class="competence-button" data-level="Beginner">Beginner</button>
        <button class="competence-button selected" data-level="Intermediate">Intermediate</button>
        <button class="competence-button" data-level="Advanced">Advanced</button>
      </div>
      
      <label for="question-input"><h2>Ask a question</h2></label>
      <input type="text" id="question-input" placeholder="Your question">
      <div class="buttons">
        <button id="ask-btn">Ask a question</button>
      </div>
      
      <div id="response" class="result-container" style="display: none;"></div>
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
  border: 2px solid #3498db;
  border-radius: 5px;
  box-sizing: border-box;
}

.buttons, .button-group {
  text-align: center;
  margin: 10px 0;
}

.button-group {
    display: flex;
    justify-content: center;
    gap: 10px;
}

@media (max-width: 600px) {
    .button-group {
        flex-direction: column;
        align-items: center;
        gap: 10px;
    }
}

button {
  padding: 10px 20px;
  margin: 10px 5px;
  border: none;
  border-radius: 5px;
  background-color: #3498db;
  color: white;
  cursor: pointer;
  font-size: 1em;
  transition: background-color 0.3s ease;
}

button:hover {
  background-color: #054b7a;
}

.competence-button {
  padding: 10px 20px;
  margin: 0 5px;
  border: 2px solid #3498db;
  background-color: #3498db;
  color: white;
  border-radius: 5px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.competence-button.selected,
.competence-button:hover {
  background-color: #054b7a;
}

.result-container {
  background-color: #e8f4fb;
  border: 2px solid #3498db;
  padding: 15px;
  border-radius: 8px;
  margin-top: 20px;
  white-space: normal;
  text-align: justify
}

.result-container p,
.result-container li,
.result-container ul {
  margin: 0.3em 0; 
  padding: 0;
  line-height: 1.0;
}

.result-container ul,
.result-container ol {
  padding-left: 1.2em; 
  list-style-position: inside; 
}

.result-container li {
  margin: 0.3em 0; 
}

.spinner {
  border: 8px solid #f3f3f3;
  border-top: 8px solid #3498db;
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
  border: 2px dashed #3498db;
  padding: 20px;
  border-radius: 5px;
  text-align: center;
  margin-top: 10px;
  position: relative;
  transition: background-color 0.3s ease, border-color 0.3s ease;
}

.pdf-upload-container.dragover {
  border-color: #054b7a;
  background-color: #f0f8ff;
}

.pdf-upload-container input[type="file"] {
  display: none;
}

.custom-file-upload {
  display: inline-block;
  padding: 10px 20px;
  border: 2px solid #3498db;
  background-color: #3498db;
  color: white;
  border-radius: 5px;
  cursor: pointer;
  font-size: 1em;
  transition: background-color 0.3s ease;
}

.custom-file-upload:hover {
  background-color: #054b7a;
}

.pdf-filename {
  margin-top: 10px;
  font-size: 0.9em;
  color: #333;
}
"""

JS_CONTENT = """
document.addEventListener('DOMContentLoaded', function() {
  const extractBtn = document.getElementById('extract-btn');
  const urlInput = document.getElementById('url-input');
  const pdfInput = document.getElementById('pdf-input');
  const pdfDropZone = document.getElementById('pdf-drop-zone');
  const pdfUploadBtn = document.getElementById('pdf-upload-btn');
  const pdfFilename = document.getElementById('pdf-filename');
  const extractedContentDiv = document.getElementById('extracted-content');
  const clearBtn = document.getElementById('clear-btn');
  const toggleContent = document.getElementById('toggle-content');
  const askBtn = document.getElementById('ask-btn');
  const questionInput = document.getElementById('question-input');
  const responseDiv = document.getElementById('response');
  const spinner = document.getElementById('spinner');
  const competenceButtons = document.querySelectorAll('.competence-button');
  let selectedCompetence = "Intermediate";

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
      pdfUploadBtn.innerText = "Dateien löschen";
    } else {
      pdfUploadBtn.innerText = "PDF-Dateien hochladen";
    }
  }

  function typeWriterHTML(html, element, speed = 5) {
    let i = 0;
    let isTag = false; 
    let output = "";

    function type() {
      if (i < html.length) {
        let char = html[i];
        output += char;
        
        if (char === "<") {
          isTag = true;
        }
        if (char === ">") {
          isTag = false;
        }
        
        element.innerHTML = output;
        
        i++;
        setTimeout(type, isTag ? 0 : speed);
      }
    }
    type();
  }

  extractBtn.addEventListener('click', function() {
    const urls = urlInput.value.trim();
    const files = pdfInput.files;
    const formData = new FormData();
    formData.append('urls', urls);
    for (let i = 0; i < files.length; i++) {
      formData.append('pdfs', files[i]);
    }
    spinner.style.display = 'block';
    fetch('/extract_content', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      spinner.style.display = 'none';
      extractedContentDiv.style.display = toggleContent.checked ? 'block' : 'none';
      extractedContentDiv.innerText = data.content || "No content extracted.";
    })
    .catch(error => {
      spinner.style.display = 'none';
      console.error("Fehler:", error);
      alert("Error when extracting the content.");
    });
  });

  clearBtn.addEventListener('click', function() {
    fetch('/clear_extracted', {
      method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
      extractedContentDiv.innerText = data.content;
    })
    .catch(error => {
      console.error("Fehler:", error);
      alert("Error when deleting the content.");
    });
  });

  toggleContent.addEventListener('change', function() {
    extractedContentDiv.style.display = this.checked ? 'block' : 'none';
  });

  askBtn.addEventListener('click', function() {
    const question = questionInput.value.trim();
    const extractedContent = extractedContentDiv.innerText;
    if (!question) {
      alert("Please enter a question.");
      return;
    }
    if (!extractedContent) {
      alert("No extracted content available.");
      return;
    }
    spinner.style.display = 'block';
    fetch('/ask_question', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        competence_level: selectedCompetence,
        content: extractedContent,
        question: question
      })
    })
    .then(response => response.json())
    .then(data => {
      spinner.style.display = 'none';
      responseDiv.style.display = 'block';
      responseDiv.innerHTML = "";
      typeWriterHTML(marked.parse(data.response), responseDiv, 5);
    })
    .catch(error => {
      spinner.style.display = 'none';
      console.error("Fehler:", error);
      alert("Fehler beim Stellen der Frage.");
    });
  });
});
"""

# --------------------------
# Flask endpoints
# --------------------------
@app.route('/')
def index():
    return Response(HTML_CONTENT, mimetype='text/html')

@app.route('/styles.css')
def styles():
    return Response(CSS_CONTENT, mimetype='text/css')

@app.route('/script.js')
def script():
    return Response(JS_CONTENT, mimetype='application/javascript')

@app.route('/extract_content', methods=['POST'])
def extract_content_endpoint():
    if request.content_type.startswith('multipart/form-data'):
        url_input = request.form.get("urls", "")
        pdf_files = request.files.getlist("pdfs")
    else:
        data = request.get_json()
        url_input = data.get("urls", "")
        pdf_files = []
    extracted = extract_content(url_input, pdf_files)
    return jsonify({"content": extracted or "Kein Inhalt extrahiert."})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.get_json()
    competence_level = data.get("competence_level")
    extracted_content = data.get("content", "")
    user_question = data.get("question", "")
    if not competence_level or not user_question.strip():
        return jsonify({"response": "Please select a competence level and enter a question."})
    if not extracted_content.strip():
        return jsonify({"response": "No extracted content available."})
    response = generate_response_from_extracted_content(competence_level, extracted_content, "", None, user_question)
    return jsonify({"response": response})

@app.route('/clear_extracted', methods=['POST'])
def clear_extracted():
    return jsonify({"content": ""})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000) 
