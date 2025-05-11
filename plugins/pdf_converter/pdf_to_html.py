"""
pdf_to_html: Convert a PDF (local file or remote URL) to HTML using a vision LLM (Ollama/LM Studio)
with customizable prompts and fallback to local models/OCR. Supports downloading PDFs from the web via wdbx.py,
and can pass database configuration to wdbx.py if supported by it.
"""

import argparse
import concurrent.futures
import html
import io
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union, Any, Callable

import requests
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
import torch
# Ensure specific types from transformers are imported for clarity
from transformers import AutoTokenizer, AutoModelForCausalLM, FlaxAutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase
import jax.numpy as jnp
import pdfplumber  # for table extraction
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import json
from wdbx import WDBX
from config import Settings

# Optional web helper module for downloading PDFs
try:
    import wdbx
except ImportError:
    wdbx = None # type: ignore [assignment]
    # Logger might not be configured yet, so use a default logger for this warning
    logging.getLogger(__name__).warning(
        "wdbx module not found; remote PDF download will be disabled."
    )

# Ollama / LM Studio endpoint (override via env var)
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

# Default prompts (override via env var or CLI)
DEFAULT_VISION_PROMPT = os.getenv(
    "VISION_PROMPT",
    "Convert the content of PDF page {page} to HTML. Preserve structure and formatting as much as possible."
)
DEFAULT_TEXT_PROMPT = os.getenv(
    "TEXT_PROMPT",
    "Convert the following text into structured HTML:"
)

# Logger setup
logger = logging.getLogger(__name__)

# More specific types for Hugging Face models and tokenizers
HuggingFaceModel = Union[PreTrainedModel, Any] # Using Any for Flax model type if not directly PreTrainedModel
HuggingFaceTokenizer = Union[PreTrainedTokenizerBase, Any] # Using Any for flexibility

class LocalTextModelHandler:
    """
    Singleton handler for local text-to-HTML model and tokenizer.

    Manages loading and caching of a single model instance to avoid redundant loads.
    The backend (PyTorch or JAX) is determined at initialization based on the
    TEXT_BACKEND environment variable or the `backend_preference` argument.
    """

    _instance = None
    _initialized: bool = False
    model: Optional[HuggingFaceModel]
    tokenizer: Optional[HuggingFaceTokenizer]
    current_model_name: Optional[str]
    backend: str


    def __new__(cls, *args: Any, **kwargs: Any) -> 'LocalTextModelHandler':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, backend_preference: str = "pt"):
        """
        Initialize the handler with a preferred backend. This is called only once.

        Args:
            backend_preference (str): 'pt' for PyTorch or 'jax' for Flax/JAX.
                                      Defaults to 'pt'.
        """
        if self._initialized:
            return
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        self.backend = backend_preference.lower()
        if self.backend not in ("pt", "jax"):
            logger.warning(
                "Unsupported TEXT_BACKEND '%s'; defaulting to 'pt'. Supported: 'pt', 'jax'.", self.backend
            )
            self.backend = "pt"
        self._initialized = True
        logger.info("LocalTextModelHandler initialized with backend: %s", self.backend)

    def get_model_and_tokenizer(self, model_name: str) -> Tuple[HuggingFaceModel, HuggingFaceTokenizer]:
        """
        Load (or return cached) model and tokenizer for the given Hugging Face model name.

        Args:
            model_name (str): Hugging Face model identifier (e.g., 'gpt2').

        Returns:
            Tuple[HuggingFaceModel, HuggingFaceTokenizer]: The loaded model and tokenizer.

        Raises:
            Exception: If loading the model or tokenizer fails.
            RuntimeError: If the model or tokenizer is not available after attempting to load.
        """
        if self.current_model_name != model_name or self.model is None or self.tokenizer is None:
            logger.info("Loading local text model '%s' (backend=%s).", model_name, self.backend)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                if self.backend == "jax":
                    # JAX-specific imports are conditional to avoid hard dependency if not used
                    self.model = FlaxAutoModelForCausalLM.from_pretrained(model_name, dtype=jnp.float32)
                else: # PyTorch backend
                    self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.current_model_name = model_name
                logger.info("Successfully loaded local text model '%s'.", model_name)
            except Exception as e:
                logger.error("Failed to load local text model '%s': %s", model_name, e)
                self.model = None
                self.tokenizer = None
                self.current_model_name = None
                raise # Re-raise the exception to signal failure

        if self.model is None or self.tokenizer is None:
            # This case should ideally not be reached if exceptions are handled correctly
            raise RuntimeError(f"Model or tokenizer for '{model_name}' is not available after load attempt.")
        return self.model, self.tokenizer

# Global singleton for local text model, backend configured by environment variable at import time.
_local_text_model_handler = LocalTextModelHandler(os.getenv("TEXT_BACKEND", "pt"))

# Initialize WDBX client for anchoring file hashes
_settings = Settings()
_anchor_client = WDBX(vector_dimension=_settings.wdbx_vector_dimension, enable_plugins=_settings.wdbx_enable_plugins)
_anchor_client.initialize()

def call_ollama_vision(image: Image.Image, prompt: str, model: str) -> str:
    """
    Send an image and a fully-formatted prompt to the Ollama vision API.

    Args:
        image (PIL.Image.Image): The page image to send.
        prompt (str): The fully-formatted prompt for the vision model.
        model (str): The name of the vision model to use (e.g., 'llava').

    Returns:
        str: HTML content generated by the vision model, stripped of leading/trailing whitespace.

    Raises:
        RuntimeError: On network errors, non-2xx HTTP status, or if the API returns invalid JSON.
    """
    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()
    files = {"images": ("page.png", image_bytes, "image/png")}
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_API_URL, data=payload, files=files, timeout=120)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as err:
        raise RuntimeError(f"Ollama vision API request failed: {err}") from err
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError as err: # More specific exception
        raise RuntimeError(f"Ollama vision API returned invalid JSON: {err}. Response text: {response.text[:500]}") from err
    return data.get("response", "").strip()

def image_to_html_with_ollama(
    image: Image.Image,
    page_num: int,
    vision_model_name: str = "llava",
    vision_prompt_template: str = DEFAULT_VISION_PROMPT,
    text_model_name: Optional[str] = "gpt2",
    text_prompt_template: str = DEFAULT_TEXT_PROMPT,
    skip_vision: bool = False,
    extract_tables: bool = False,
) -> str:
    """
    Convert a PDF page image to HTML using multiple strategies.

    Strategies in order:
    1. Ollama vision model.
    2. If (1) fails or returns empty: Tesseract OCR.
    3. If OCR succeeds and `text_model_name` is provided (not 'none'): Local text-to-HTML model.
    4. If local model fails, is skipped, or returns empty: Pre-formatted OCR text.

    Args:
        image (PIL.Image.Image): The image of the PDF page.
        page_num (int): The 1-based page number.
        vision_model_name (str): Name of the Ollama vision model.
        vision_prompt_template (str): Prompt template for the vision model (e.g., "Page {page}...").
        text_model_name (Optional[str]): Name of the local text model for fallback.
                                         Set to 'none' or None to disable this fallback.
        text_prompt_template (str): Prompt template for the local text model (e.g., "{text}...").
        skip_vision (bool): Whether to skip the vision model stage.
        extract_tables (bool): Whether to extract tables from the PDF page.

    Returns:
        str: HTML representation of the page content.
    """
    prompt = vision_prompt_template.format(page=page_num)
    html_content = ""

    # Attempt 1: Ollama Vision Model (skipped if skip_vision=True)
    if not skip_vision:
        try:
            logger.info(f"Page {page_num}: Requesting conversion from vision model '{vision_model_name}'.")
            html_content = call_ollama_vision(image, prompt, vision_model_name)
            if not html_content:
                logger.warning(f"Page {page_num}: Vision model '{vision_model_name}' returned empty content.")
        except Exception as vision_err:
            logger.warning(f"Page {page_num}: Vision model '{vision_model_name}' failed: {vision_err}.")
    else:
        logger.info(f"Page {page_num}: Vision model stage skipped per skip_vision flag.")

    # Fallback if vision model did not produce content
    if not html_content:
        logger.info(f"Page {page_num}: Attempting fallback using OCR.")
        try:
            raw_text = pytesseract.image_to_string(image)
            if not raw_text.strip():
                logger.warning(f"Page {page_num}: OCR extracted no text.")
                return f"<p><em>Page {page_num}: No content found (Vision model failed, OCR found no text).</em></p>"
            logger.info(f"Page {page_num}: OCR extracted {len(raw_text)} characters.")

            # Attempt 2: Local Text-to-HTML Model (if configured)
            if text_model_name and text_model_name.lower() != "none":
                try:
                    logger.info(f"Page {page_num}: Attempting conversion with local text model '{text_model_name}'.")
                    local_model, tokenizer = _local_text_model_handler.get_model_and_tokenizer(text_model_name)
                    
                    full_prompt = (
                        text_prompt_template.format(text=raw_text)
                        if "{text}" in text_prompt_template
                        else f"{text_prompt_template}\n\n{raw_text}"
                    )
                    
                    tokenizer_max_len_attr = getattr(local_model.config, 'n_positions', 1024)
                    tokenizer_max_len = int(tokenizer_max_len_attr) if isinstance(tokenizer_max_len_attr, (int, float)) else 1024


                    if _local_text_model_handler.backend == "jax":
                        inputs = tokenizer(full_prompt, return_tensors="jax", truncation=True, max_length=tokenizer_max_len) # type: ignore
                        # Assuming local_model is a JAX model here
                        outputs = local_model.generate(**inputs, max_length=min(tokenizer_max_len * 2, 4096)) # type: ignore
                        generated_ids = jnp.array(outputs)
                    else: # PyTorch
                        device = next(local_model.parameters()).device # type: ignore
                        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=tokenizer_max_len).to(device) # type: ignore
                        with torch.no_grad():
                            outputs = local_model.generate(**inputs, max_length=min(tokenizer_max_len * 2, 4096)) # type: ignore
                        generated_ids = outputs.cpu()

                    decoded_html_list = tokenizer.batch_decode(generated_ids, skip_special_tokens=True) # type: ignore
                    html_content = decoded_html_list[0] if decoded_html_list else ""
                    
                    if html_content:
                        logger.info(f"Page {page_num}: Successfully converted OCR text using local model '{text_model_name}'.")
                    else:
                        logger.warning(f"Page {page_num}: Local model '{text_model_name}' returned empty content.")
                except Exception as local_err:
                    logger.error(f"Page {page_num}: Local text model '{text_model_name}' failed: {local_err}.")
            else:
                logger.info(f"Page {page_num}: Local text model processing skipped (text_model_name: '{text_model_name}').")

            # Fallback 3: Pre-formatted OCR text if local model failed or was skipped
            if not html_content:
                logger.info(f"Page {page_num}: Using pre-formatted OCR text as final fallback.")
                escaped_text = html.escape(raw_text)
                html_content = f"<pre>{escaped_text}</pre>"
        except Exception as ocr_err:
            logger.error(f"Page {page_num}: OCR processing failed: {ocr_err}")
            html_content = f"<p><em>Page {page_num}: Error during OCR processing: {html.escape(str(ocr_err))}</em></p>"

    if not html_content: # If all attempts failed
        html_content = f"<p><em>Page {page_num}: Content conversion failed through all stages.</em></p>"
    return html_content

def pdf_to_html(
    pdf_path: Path,
    html_path: Path,
    vision_model_name: str = "llava",
    vision_prompt_template: str = DEFAULT_VISION_PROMPT,
    text_model_name: Optional[str] = "gpt2",
    text_prompt_template: str = DEFAULT_TEXT_PROMPT,
    max_workers: int = None,
    skip_vision: bool = False,
    extract_tables: bool = False,
    encrypt_html: bool = False,
    anchor_hash_flag: bool = False,
) -> None:
    """
    Convert all pages of a PDF file to a single HTML document.

    Processes pages in parallel. The resulting HTML includes basic styling.

    Args:
        pdf_path (Path): Path to the local PDF file.
        html_path (Path): Path where the output HTML file will be saved.
        vision_model_name (str): Name of the Ollama vision model to use.
        vision_prompt_template (str): Prompt template for the vision model.
        text_model_name (Optional[str]): Name of the local text model for OCR fallback.
        text_prompt_template (str): Prompt template for the local text model.
        max_workers (int): Number of worker threads to use for parallel processing.
        skip_vision (bool): Whether to skip the vision model stage.
        extract_tables (bool): Whether to extract tables from the PDF pages.
        encrypt_html (bool): Whether to encrypt the output HTML file using AES-GCM.
        anchor_hash_flag (bool): Whether to anchor the encrypted HTML hash to the blockchain via WDBX.
    """
    logger.info(f"Starting PDF conversion: {pdf_path} -> {html_path} using vision model '{vision_model_name}'.")
    try:
        pages = convert_from_path(str(pdf_path))
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}")
        html_path.write_text(
            f"<html><body><h1>Error</h1><p>Could not process PDF file: {html.escape(str(e))}</p></body></html>",
            encoding="utf-8"
        )
        return

    if not pages:
        logger.error(f"No pages found in PDF: {pdf_path}")
        html_path.write_text(
            "<html><body><h1>Error</h1><p>No pages found in PDF.</p></body></html>",
            encoding="utf-8"
        )
        return

    logger.info(f"PDF has {len(pages)} pages. Processing in parallel with max_workers={max_workers or os.cpu_count() or 4}... (extract_tables={extract_tables})")
    page_html_contents = [""] * len(pages)
    max_workers = max_workers or os.cpu_count() or 4

    # Pre-extract tables if requested
    raw_tables = [[] for _ in pages]
    if extract_tables:
        try:
            with pdfplumber.open(str(pdf_path)) as pdfp:
                raw_tables = [page.extract_tables() or [] for page in pdfp.pages]
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}; continuing without tables.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                image_to_html_with_ollama,
                image,
                idx + 1,
                vision_model_name,
                vision_prompt_template,
                text_model_name,
                text_prompt_template,
                skip_vision,
                extract_tables,
            ): idx
            for idx, image in enumerate(pages)
        }
        for i, future in enumerate(concurrent.futures.as_completed(future_to_index)):
            idx = future_to_index[future]
            page_num = idx + 1
            try:
                page_html_contents[idx] = future.result()
                logger.info(f"Completed processing page {page_num}/{len(pages)} ({((i+1)/len(pages))*100:.1f}% done)")
            except Exception as proc_err:
                logger.error(f"Error processing page {page_num}: {proc_err}")
                page_html_contents[idx] = (
                    f"<h2>Page {page_num}</h2>"
                    f"<p><em>Error converting page: {html.escape(str(proc_err))}</em></p>"
                )

    # Assemble the final HTML document
    html_doc_parts = [
        "<!DOCTYPE html>",
        "<html lang=\"en\">",
        "<head>",
        '    <meta charset="utf-8">',
        f'    <title>Converted PDF: {html.escape(pdf_path.name)}</title>',
        "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f4f4f4; color: #333; line-height: 1.6; }",
        "        .container { max-width: 900px; margin: 20px auto; padding: 20px; background-color: #fff; box-shadow: 0 0 10px rgba(0,0,0,0.1); }",
        "        h1.doc-title { text-align: center; color: #444; margin-bottom: 30px; }",
        "        .page-container { border: 1px solid #ddd; margin-bottom: 20px; padding: 20px; background-color: #fff; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }",
        "        .page-header { font-size: 0.9em; color: #777; margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 5px; text-align: right; }",
        "        pre { white-space: pre-wrap; word-wrap: break-word; background-color: #f9f9f9; padding: 15px; border: 1px solid #eee; border-radius: 4px; font-size: 0.95em; }",
        "        img { max-width: 100%; height: auto; }",
        "        table { width: 100%; border-collapse: collapse; margin: 10px 0; }",
        "        table, th, td { border: 1px solid #ccc; }",
        "        th, td { padding: 8px; text-align: left; }",
        "    </style>",
        "</head>",
        "<body>",
        '    <div class="container">',
        f'        <h1 class="doc-title">Conversion of: {html.escape(pdf_path.name)}</h1>'
    ]
    for i, page_content in enumerate(page_html_contents):
        page_num = i + 1
        html_doc_parts.append(f'        <div class="page-container" id="page-{page_num}">')
        html_doc_parts.append(f'            <div class="page-header">Page {page_num}</div>')
        html_doc_parts.append(f'            <div>{page_content}</div>')
        # Render extracted tables for this page
        if extract_tables and raw_tables[i]:
            html_doc_parts.append('            <div class="tables-container">')
            for table in raw_tables[i]:
                html_doc_parts.append('                <table>')
                for row in table:
                    cells = ''.join(f'<td>{html.escape(str(cell or ""))}</td>' for cell in row)
                    html_doc_parts.append(f'                    <tr>{cells}</tr>')
                html_doc_parts.append('                </table>')
            html_doc_parts.append('            </div>')
        html_doc_parts.append(f'        </div>')
    html_doc_parts.append("    </div>")
    html_doc_parts.append("</body>")
    html_doc_parts.append("</html>")
    
    html_output = "\n".join(html_doc_parts)
    html_path.write_text(html_output, encoding="utf-8")
    logger.info(f"Successfully converted {pdf_path} to {html_path}.")

    if encrypt_html:
        # Encrypt the HTML content
        aes_gcm = AESGCM(os.urandom(16))  # Generate a new AES-GCM key for each encryption
        encrypted_html = aes_gcm.encrypt(os.urandom(16), html_output.encode('utf-8'))
        encrypted_html_path = html_path.with_suffix(html_path.suffix + ".enc")
        encrypted_html_path.write_text(encrypted_html.hex(), encoding="utf-8")
        logger.info(f"Encrypted HTML saved to: {encrypted_html_path}")

    if anchor_hash_flag:
        # Anchor the encrypted HTML hash via WDBX
        _anchor_client.anchor_hash(encrypted_html_path)
        logger.info(f"HTML hash anchored in WDBX: {encrypted_html_path}")

def main() -> None:
    """
    Command-line interface for PDF to HTML conversion.

    Supports local PDF file paths or remote HTTP/HTTPS URLs for PDF input.
    If a URL is provided, the `wdbx` module (if available) will be used to download it.
    This script can also pass a database URL to `wdbx` if `wdbx` supports database integration
    via a `configure_database` function.
    """
    if __name__ == "__main__" and not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", # Use %(name)s for logger name
            stream=sys.stdout,
        )

    parser = argparse.ArgumentParser(
        description=(
            "Convert a PDF (local file or remote URL) to HTML using a vision LLM "
            "(Ollama/LM Studio compatible) with customizable prompts and fallback mechanisms. "
            "Can utilize the 'wdbx' module for downloading remote PDFs and configuring its database usage."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_pdf", type=str, help="Path to the input PDF file or a URL (http/https).")
    parser.add_argument("output_html", type=Path, help="Path for the output HTML file.")
    parser.add_argument(
        "--model", 
        default=os.getenv("OLLAMA_VISION_MODEL", "llava"),
        help="Vision model name for Ollama/LM Studio (e.g., 'llava'). Env: OLLAMA_VISION_MODEL."
    )
    parser.add_argument(
        "--vision-prompt",
        default=DEFAULT_VISION_PROMPT,
        help="Prompt template for the vision model. Use '{page}' for page number. Env: VISION_PROMPT."
    )
    parser.add_argument(
        "--text-model",
        default=os.getenv("TEXT_MODEL", "gpt2"),
        help="Local text model for OCR fallback (e.g., 'gpt2', 'distilgpt2'). Set to 'none' to disable. Env: TEXT_MODEL."
    )
    parser.add_argument(
        "--text-prompt",
        default=DEFAULT_TEXT_PROMPT,
        help="Prompt template for the local text model. Use '{text}' for OCR content. Env: TEXT_PROMPT."
    )
    parser.add_argument(
        "--ollama-api-url",
        default=OLLAMA_API_URL,
        help="Ollama/LM Studio API URL. Env: OLLAMA_API_URL."
    )
    parser.add_argument(
        "--text-backend",
        default=os.getenv("TEXT_BACKEND", "pt"),
        choices=["pt", "jax"],
        help="Backend for local text model ('pt' for PyTorch, 'jax' for JAX). Configured at script start. Env: TEXT_BACKEND."
    )
    parser.add_argument(
        "--wdbx-db-url",
        default=os.getenv("WDBX_DATABASE_URL"),
        type=str,
        help="Optional database URL for wdbx module (if wdbx supports it via a 'configure_database' function). Env: WDBX_DATABASE_URL."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of threads for parallel processing (defaults to CPU count)."
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Skip the vision LLM stage and use OCR fallback directly."
    )
    parser.add_argument(
        "--extract-tables",
        action="store_true",
        help="Extract tables from the PDF pages."
    )
    parser.add_argument(
        "--encrypt-html",
        action="store_true",
        help="Encrypt the output HTML file using AES-GCM."
    )
    parser.add_argument(
        "--anchor-hash",
        action="store_true",
        help="Anchor the encrypted HTML hash to the blockchain via WDBX."
    )
    args = parser.parse_args()

    global OLLAMA_API_URL, DEFAULT_VISION_PROMPT, DEFAULT_TEXT_PROMPT
    OLLAMA_API_URL = args.ollama_api_url
    DEFAULT_VISION_PROMPT = args.vision_prompt
    DEFAULT_TEXT_PROMPT = args.text_prompt

    if _local_text_model_handler.backend != args.text_backend:
        logger.warning(
            f"TEXT_BACKEND specified via CLI ('{args.text_backend}') differs from the active backend "
            f"('{_local_text_model_handler.backend}'), set by environment or default. "
            "To change the active backend, set TEXT_BACKEND environment variable before running."
        )

    input_source_str = args.input_pdf
    actual_pdf_path: Path
    
    if input_source_str.lower().startswith(("http://", "https://")):
        # Remote PDF download (try wdbx, else fallback to direct HTTP)
        args.output_html.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(wdbx, 'download_file'):
            # Use wdbx for download
            if hasattr(wdbx, 'configure_database') and args.wdbx_db_url:
                wdbx.configure_database(args.wdbx_db_url)
            try:
                downloaded_file_path_str = wdbx.download_file(input_source_str, args.output_html.parent)
                actual_pdf_path = Path(downloaded_file_path_str)
                logger.info(f"PDF downloaded via wdbx to: {actual_pdf_path}")
            except Exception as e:
                logger.error(f"Failed to download PDF via wdbx from URL '{input_source_str}': {e}")
                sys.exit(1)
        else:
            # Fallback: download using requests
            try:
                logger.info(f"Downloading PDF directly from URL: {input_source_str}")
                response = requests.get(input_source_str, stream=True, timeout=60)
                response.raise_for_status()
                local_filename = input_source_str.split('/')[-1] or 'downloaded_file.pdf'
                local_path = args.output_html.parent / local_filename
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                actual_pdf_path = local_path
                logger.info(f"PDF downloaded via HTTP to: {actual_pdf_path}")
            except Exception as e:
                logger.error(f"Failed to download PDF via HTTP from URL '{input_source_str}': {e}")
                sys.exit(1)
    else:
        actual_pdf_path = Path(input_source_str)
        if not actual_pdf_path.is_file(): # More specific check
            logger.error(f"Input PDF file not found or is not a file: {actual_pdf_path}")
            sys.exit(1)

    args.output_html.parent.mkdir(parents=True, exist_ok=True)

    pdf_to_html(
        pdf_path=actual_pdf_path,
        html_path=args.output_html,
        vision_model_name=args.model,
        vision_prompt_template=DEFAULT_VISION_PROMPT,
        text_model_name=args.text_model,
        text_prompt_template=DEFAULT_TEXT_PROMPT,
        max_workers=args.workers,
        skip_vision=args.skip_vision,
        extract_tables=args.extract_tables,
        encrypt_html=args.encrypt_html,
        anchor_hash_flag=args.anchor_hash,
    )

    logger.info(
        "Conversion process finished for '%s'. Output at '%s'.",
        actual_pdf_path, args.output_html
    )

if __name__ == "__main__":
    main()
