from flask import Blueprint, request, render_template_string, jsonify, url_for
import os, json
from werkzeug.utils import secure_filename
import jedi

strim_bp = Blueprint("strim", __name__, url_prefix="/strim")

# Load plugin metadata for options
INFO_PATH = os.path.join(os.path.dirname(__file__), "info.json")
with open(INFO_PATH) as f:
    _info = json.load(f)
# Determine and prepare data directory
DATA_DIR = os.path.join(os.getcwd(), _info.get("options", {}).get("data_directory", "strim_data"))
os.makedirs(DATA_DIR, exist_ok=True)

# Helpers

def list_files():
    return [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]

def get_file_path(filename):
    safe = secure_filename(filename)
    return os.path.join(DATA_DIR, safe if safe.endswith('.txt') else f"{safe}.txt")

# Routes

@strim_bp.route("/", methods=["GET"])
def file_list():
    files = list_files()
    return render_template_string(
        """
        <h1>Strim Editor</h1>
        <ul>
          {% for fname in files %}
            <li>
              <a href="{{ url_for('strim.edit', filename=fname) }}">{{ fname }}</a>
              <button onclick="deleteFile('{{ fname }}')">Delete</button>
            </li>
          {% endfor %}
        </ul>
        <h2>Create New File</h2>
        <form id="newForm">
          <input type="text" name="filename" placeholder="Filename" required/>
          <button type="submit">Create</button>
        </form>
        <script>
        document.getElementById('newForm').onsubmit = function(e) {
          e.preventDefault();
          const fn = this.filename.value;
          window.location = '{{ url_for("strim.edit", filename="") }}' + fn;
        };
        function deleteFile(fname) {
          if(!confirm('Delete ' + fname + '?')) return;
          fetch('{{ url_for("strim.delete", filename="") }}' + fname, { method: 'POST' })
            .then(r => r.json())
            .then(j => { alert(j.status || j.error); location.reload(); });
        }
        </script>
        """, files=files
    )

@strim_bp.route("/edit/<path:filename>", methods=["GET"])
def edit(filename):
    fp = get_file_path(filename)
    content = ""
    if os.path.exists(fp):
        with open(fp, "r", encoding="utf-8") as f:
            content = f.read()
    return render_template_string(
        """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Editing {{ filename }} - Strim Editor</title></head>
<body>
<div id="editor" style="width:100%;height:80vh;border:1px solid #ccc;"></div>
<button onclick="save()">Save</button>
<button onclick="window.location='{{ url_for('strim.file_list') }}'">Back</button>
<script src="https://unpkg.com/monaco-editor@0.33.0/min/vs/loader.js"></script>
<script>
require.config({ paths: { 'vs': 'https://unpkg.com/monaco-editor@0.33.0/min/vs' } });
require(['vs/editor/editor.main'], function() {
    window.editor = monaco.editor.create(
        document.getElementById('editor'),
        { value: `{{ content|replace("`","\\`") }}`, language: 'python', automaticLayout: true }
    );
    monaco.languages.registerCompletionItemProvider('python', {
        triggerCharacters: ['.', '(', '['],
        provideCompletionItems: (model, position) => {
            return fetch(
                '{{ url_for("strim.complete", filename=filename) }}',
                { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ line: position.lineNumber, column: position.column }) }
            ).then(r => r.json()).then(data => ({
                suggestions: data.map(item => ({
                    label: item.name,
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: item.complete || item.name,
                    range: undefined
                }))
            }));
        }
    });
    function updateDiagnostics() {
        fetch('{{ url_for("strim.diagnostics", filename=filename) }}')
            .then(r => r.json())
            .then(diags => {
                const markers = diags.map(d => ({
                    severity: monaco.MarkerSeverity.Error,
                    message: d.message,
                    startLineNumber: d.line || 1,
                    startColumn: d.column || 1,
                    endLineNumber: d.line || 1,
                    endColumn: d.column ? d.column + 1 : 1
                }));
                monaco.editor.setModelMarkers(window.editor.getModel(), 'owner', markers);
            });
    }
    editor.getModel().onDidChangeContent(() => { clearTimeout(window._diag); window._diag = setTimeout(updateDiagnostics, 500); });
    updateDiagnostics();
});
function save() {
    fetch(
        '{{ url_for("strim.save", filename=filename) }}',
        { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ content: window.editor.getValue() }) }
    ).then(r => r.json()).then(j => alert(j.status || j.error));
}
</script>
</body>
</html>
""", filename=filename, content=content)

@strim_bp.route("/save/<path:filename>", methods=["POST"])
def save(filename):
    data = request.get_json(force=True) or {}
    content = data.get('content', '')
    fp = get_file_path(filename)
    try:
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(content)
        return jsonify({'status': 'saved'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@strim_bp.route("/delete/<path:filename>", methods=["POST"])
def delete(filename):
    fp = get_file_path(filename)
    try:
        if os.path.exists(fp):
            os.remove(fp)
            return jsonify({'status': 'deleted'}), 200
        else:
            return jsonify({'error': 'not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@strim_bp.route("/files", methods=["GET"])
def files_api():
    return jsonify(list_files()), 200

@strim_bp.route("/raw/<path:filename>", methods=["GET"])
def raw(filename):
    fp = get_file_path(filename)
    if os.path.exists(fp):
        return open(fp, 'r', encoding='utf-8').read(), 200, {'Content-Type': 'text/plain'}
    return 'Not found', 404

@strim_bp.route("/complete/<path:filename>", methods=["POST"])
def complete(filename):
    """Provide completions via jedi for the given file and position."""
    fp = get_file_path(filename)
    source = ''
    if os.path.exists(fp):
        source = open(fp, 'r', encoding='utf-8').read()
    data = request.get_json(force=True) or {}
    line = data.get('line', 1)
    column = data.get('column', 1)
    try:
        script = jedi.Script(source, path=fp)
        comps = script.complete(line, column)
        result = [ {'name': c.name, 'complete': c.complete, 'type': c.type} for c in comps ]
    except Exception:
        result = []
    return jsonify(result)

@strim_bp.route("/diagnostics/<path:filename>", methods=["GET"])
def diagnostics(filename):
    """Return syntax errors by attempting to compile the code."""
    fp = get_file_path(filename)
    source = ''
    if os.path.exists(fp):
        source = open(fp, 'r', encoding='utf-8').read()
    diags = []
    try:
        compile(source, fp, 'exec')
    except SyntaxError as e:
        diags = [{ 'message': e.msg, 'line': e.lineno, 'column': e.offset }]
    except Exception:
        diags = []
    return jsonify(diags) 