from pathlib import Path
from tempfile import TemporaryDirectory

from flask import Blueprint, jsonify, request, send_file

from plugins.pdf_converter.pdf_to_html import pdf_to_html

pdf_bp = Blueprint("pdf_converter", __name__, url_prefix="/pdf")


@pdf_bp.route("/convert", methods=["POST"])
def convert_pdf():
    """Upload a PDF and return the converted HTML file."""
    if "pdf_file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["pdf_file"]
    filename = file.filename
    if filename == "":
        return jsonify({"error": "Filename is empty."}), 400
    with TemporaryDirectory() as tmpdir:
        upload_path = Path(tmpdir) / filename
        html_filename = f"{Path(filename).stem}.html"
        html_path = Path(tmpdir) / html_filename
        file.save(upload_path)
        try:
            pdf_to_html(upload_path, html_path)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        return send_file(html_path, as_attachment=True, download_name=html_filename)
