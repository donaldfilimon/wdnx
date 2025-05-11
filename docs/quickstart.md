# WDBX Package Quickstart

Welcome to the `wdbx` package! This guide walks through the core usage patterns.

## Installation

```bash
pip install wdbx  # or install your local package
```

## Synchronous Client

```python
from wdbx import WDBX

# Create and initialize
db = WDBX(vector_dimension=384, enable_plugins=False)
db.initialize()

# Store a vector
vector_id = db.store([0.1, 0.2, 0.3], {"text": "example"})
print(f"Stored vector ID: {vector_id}")

# Search similar vectors
results = db.search([0.1, 0.2, 0.3], limit=5)
print("Search results:", results)

# Shutdown when done
db.shutdown()
```

## Asynchronous Client

```python
import asyncio
from wdbx import AsyncWDBX

async def main():
    db = AsyncWDBX(vector_dimension=384)
    await db.initialize()

    # Store and search
    vector_id = await db.store([0.1, 0.2, 0.3], {"text": "example async"})
    print(f"Stored vector ID: {vector_id}")

    results = await db.search([0.1, 0.2, 0.3], limit=3)
    print("Async search results:", results)

    await db.shutdown()

asyncio.run(main())
```

## PDF Download Utility

```python
from wdbx import download_file, configure_database

# Optional: configure a database (stub)
configure_database("postgresql://user:pass@localhost/db")

# Download a remote PDF to a local folder
pdf_path = download_file("https://example.com/document.pdf", "./downloads")
print(f"Downloaded PDF to: {pdf_path}")
```

## Metrics Server

```python
from wdbx import start_metrics_server

# Start Prometheus metrics at http://0.0.0.0:8000/metrics
start_metrics_server(port=8000, addr='0.0.0.0')
```

## Artifact Storage

```python
# Store a model file as an artifact in WDBX
artifact_id = db.store_model(
    "models/my_model.pkl",
    {"description": "Fine-tuned language model"}
)
print(f"Stored model artifact ID: {artifact_id}")

# Retrieve a stored model artifact to local path
db.load_model(artifact_id, "models/downloaded_model.pkl")
```

## Lylex Database Integration

The `lylex` package can store and query interactions using a WDBX vector database via `LylexDB`:

```python
from lylex import LylexDB

# Initialize the database wrapper
db = LylexDB(vector_dimension=384)

# Store a prompt-response interaction
db_id = db.store_interaction("Hello, world!", "Hi there!")
print(f"Stored interaction ID: {db_id}")

# Query similar past interactions
results = db.search_interactions("Hello?")
print(results)

# Shutdown when done
db.shutdown()
```

## Lylex Model Fine-Tuning

```python
from lylex.ai import LylexModelHandler

# Initialize handler and load a pre-trained model
handler = LylexModelHandler(backend="pt")
handler.load_model("gpt2")

# Fine-tune on your dataset (e.g., a Hugging Face Dataset object)
handler.train(
    train_dataset=my_dataset,
    output_dir="./fine_tuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
)
# The fine-tuned model and tokenizer are saved to './fine_tuned'
```

## Lylex Artifact Storage

```python
from lylex import LylexDB

# Initialize database wrapper
db = LylexDB(vector_dimension=384)

# Store a trained model file
artifact_id = db.store_model(
    "./fine_tuned/pytorch_model.bin",
    {"description": "gpt2 fine-tuned"}
)
print(f"Stored model artifact: {artifact_id}")

# Retrieve the artifact to disk
db.load_model(artifact_id, "./downloaded_model.bin")
```

## Lylex Neural Backtrace

```python
from lylex import LylexDB

# Initialize database wrapper with embeddings
db = LylexDB(
    vector_dimension=384,
    embed_fn=my_embedding_function,
)

# Perform neural backtrace for a prompt
backtrace_info = db.neural_backtrace("Analyze this input text for neural backtrace.")
print(backtrace_info)

# Or directly backtrace on an embedding vector
vector = my_embedding_function("Another input")
info = db.backtrace_pattern(vector)
print(info)
```

## LylexAgent Conversational Client

```python
from lylex.ai import LylexAgent
from lylex.db import LylexDB

# Initialize memory-backed agent
memory = LylexDB(vector_dimension=384)
agent = LylexAgent(
    model_name="gpt2",
    backend="pt",
    memory_db=memory,
    memory_limit=5
)

print("Enter 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = agent.chat(user_input, max_length=100)
    print("Agent:", response)
```

## Automatic Outdated Package Monitoring

WDBX can automatically record and query outdated Python packages:

1. **Recording**  
   - On startup and every `<update_interval_minutes>` (default 60), the app invokes `wdbx.record_outdated_packages()`, storing metadata entries with:
     ```json
     {
       "package": "…",
       "current_version": "…",
       "latest_version": "…"
     }
     ```
   - Tune the cadence by setting `UPDATE_INTERVAL_MINUTES` in your `.env` file.

2. **CLI Access**  
   Use the new Terminal CLI group:
   ```bash
   flask --app app term check-updates           # Print outdated packages
   flask --app app term check-updates --record  # Also record into WDBX
   ```

3. **Programmatic Query**  
   - **Direct WDBX client**:
     ```python
     from wdbx import WDBX
     db = WDBX(vector_dimension=1, enable_plugins=True)
     db.initialize()
     # Record and retrieve
     count = db.record_outdated_packages()
     # Query stored entries
     vector = [0.0] * db.vector_dimension
     results = db.search(vector, limit=100)
     packages = [m for _id, _score, m in results if 'package' in m]
     ```
   - **LylexDB wrapper**:
     ```python
     from lylex import LylexDB
     db = LylexDB(vector_dimension=1)
     packages = db.search_outdated_packages()
     ```
   - **LylexAgent**:
     ```python
     from lylex.ai import LylexAgent
     agent = LylexAgent(model_name='gpt2', memory_db=db)
     print(agent.packages_to_upgrade())
     ```

These features give you a "knowledge base" of package health that you can integrate into dashboards, alerts, or AI workflows.

For more details, explore the source code and refer to the docstrings for each function. 