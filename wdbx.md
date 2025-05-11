# WDBX: Wide Distributed Block Database

## Overview

WDBX is a super-speedy, ultra-responsive distributed database designed for multi-personality AI systems. It blends vector similarity search, blockchain-style integrity, multiversion concurrency control (MVCC), and multi-head attention to deliver low-latency, high-throughput data storage and retrieval.

## Key Features

- **Embedding Vector Storage**: Efficiently store, retrieve, and search high-dimensional embedding vectors using advanced indexing structures and similarity search techniques.
- **Blockchain-Inspired Data Integrity**: Chain data blocks with cryptographic hashes to ensure tamper-evident logs and verifiable data histories.
- **MVCC Concurrency**: Support high-concurrency read/write operations with multiversion concurrency control, minimizing lock contention and ensuring snapshot isolation.
- **Neural Backtracking**: Trace activation pathways across stored vectors to analyze semantic drift and model behavior over time.
- **Multi-Head Attention**: Apply transformer-like attention mechanisms to sequence modeling and query processing for enhanced relevance ranking.
- **Multi-Persona Framework**: Maintain isolated contexts for multiple AI personas, enabling concurrent conversations or tasks without cross-talk.
- **Content Filtering & Bias Mitigation**: Integrate built-in filters and bias checking pipelines to ensure safe, balanced responses.
- **Asynchronous HTTP Server**: Provide RESTful API endpoints via an aiohttp-based server for seamless integration into AI workflows.

## Architecture

1. **Data Blocks & Chaining**: Data is organized into blocks containing batched embedding vectors and metadata. Each block references the previous via SHA-256 hashes, forming an immutable chain.
2. **Indexing & Search**: Uses optimized vector indexes (e.g., HNSW, IVF) for approximate nearest-neighbor search, providing sub-50ms query responses at scale.
3. **Concurrency Control**: Read and write transactions use MVCC to create non-blocking snapshots, minimizing write amplification and ensuring consistent read views.
4. **Attention Modules**: Multi-head attention modules run as part of query processing pipelines, weighting relevance across vector dimensions.

## Performance Comparison

| Feature                    | WDBX                                    | Traditional DB (SQL/NoSQL)         |
|----------------------------|-----------------------------------------|------------------------------------|
| **Sharding**               | Automatic, fine-tuned                   | Manual or partial                  |
| **Latency**                | ~20–30% lower for AI workloads          | Standard range                     |
| **MVCC Overheads**         | Optimized for high-frequency logs       | General-purpose overheads          |
| **Security**               | AES-256 encryption + RBAC               | Vendor-dependent                   |
| **Scalability**            | Seamless horizontal scaling            | Limited by architecture            |
| **Data Retrieval**         | Vector-based similarity search          | Relational or key-value access     |
| **Redundancy**             | Built-in block chaining and replication | Configurable, often external tools |

## Installation

1. Clone the repository and navigate to the project folder.
2. Install dependencies and run the database server:

```bash
pip install -r requirements.txt
python database.py
```

3. Ensure Python 3.9–3.13 is installed; no additional system dependencies are required.

## Configuration

You can configure the application via environment variables or a `.env` file at the project root. The following are supported:
- `FLASK_SECRET_KEY`: Secret key for Flask sessions. Defaults to `replace-me-with-secure-key`.
- `UPLOAD_FOLDER`: Path to store uploaded PDF files. Defaults to `uploads`.
- `OUTPUT_FOLDER`: Path to store generated HTML files. Defaults to `generated_html`.
- `WDBX_VECTOR_DIMENSION`: Dimension of the embedding vectors. Defaults to `1`.
- `WDBX_ENABLE_PLUGINS`: Set to `true` or `false` to enable WDBX plugins. Defaults to `false`.
- `METRICS_PORT`: Port for the Prometheus metrics server. Defaults to `8000`.
- `METRICS_ADDR`: Address for the metrics server to bind. Defaults to `0.0.0.0`.

## Usage Example

```bash
# Start the asynchronous HTTP server
python database.py --host 0.0.0.0 --port 8080
```

```python
import requests

# Store an embedding vector
response = requests.post(
    'http://localhost:8080/vectors',
    json={'id': 'item123', 'vector': [0.1, 0.2, 0.3, ...]}
)
print(response.json())  # Confirmation of storage

# Query nearest neighbors
response = requests.post(
    'http://localhost:8080/query',
    json={'vector': [0.1, 0.2, 0.3, ...], 'top_k': 5}
)
print(response.json())  # List of nearest vectors
```

## Roadmap

- **JAX Acceleration**: Integrate JIT-compiled kernels to optimize latency-critical operations.
- **Plugin System**: Enable custom filter pipelines and attention models.
- **Monitoring Dashboard**: Real-time visualization of chain integrity and query metrics.

## References

- Python implementation: [`database.py`](https://github.com/donaldfilimon/wdbx_python/blob/main/database.py)
- Comparative analysis and performance benchmarks in official README (section 6.4).

## API Reference

**POST** `/vectors`
- Request:
  - `id` (string): Unique identifier for the vector.
  - `vector` (array of float): The embedding vector values.
- Response:
  - `status` (string): Confirmation message.
  - `stored_id` (string): ID of stored vector.

**POST** `/query`
- Request:
  - `vector` (array of float): Query embedding vector.
  - `top_k` (integer): Number of nearest neighbors.
- Response:
  - `results` (array): Array of objects with `id` and `distance`.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 