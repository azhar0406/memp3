# memp3

**AI memory encoded in sound. Lossless, fast, and hardware-adaptive.**

`memp3` is a cutting-edge AI memory system that compresses and stores textual knowledge into FLAC audio files, achieving 10–15× compression and sub-100ms retrieval—even at scale.

## 🚀 Features

- 🔊 **Text-to-FLAC Encoding**: Encodes text across the 20Hz–24kHz audio spectrum
- ⚡ **Fast Retrieval**: Retrieves from millions of chunks in under 100ms
- 🛡️ **Error Correction**: Multi-layer ECC including Reed-Solomon, LDPC, and CRC
- 📦 **Delta Updates**: Version-controlled, append-only knowledge with write-ahead logging
- 🌐 **Cloud Ready**: Streams memory chunks via CDN, S3, or self-hosted MinIO
- 🔎 **Semantic Search**: FAISS-powered fast vector similarity search
- ⚙️ **Hardware Adaptive**: Automatically configures encoding for servers, mobiles, and embedded devices

## 🔧 Tech Stack

- **Languages**: Python 3.11+
- **Audio**: FLAC, librosa, Soundfile, PyDub
- **Storage**: RocksDB, SQLite, AWS S3
- **Search**: FAISS, Bloom Filters
- **ML**: Sentence Transformers, ONNX Runtime
- **API**: FastAPI, Uvicorn, WebSockets
- **Infra**: Docker, Kubernetes, Prometheus, Grafana

## 📦 Installation

```bash
git clone https://github.com/azhar0406/memp3.git
cd memp3
poetry install
```

Or with Docker:

```bash
docker build -t memp3 .
docker run -p 8000:8000 memp3
```

## 🔍 Usage

Send a POST request to `/encode` with your text. Retrieve it with a `/query`.

Example:
```bash
curl -X POST http://localhost:8000/encode -d '{"text": "Hello World"}'
```

## 📊 Benchmarks

- 10–15× compression ratio
- <100ms p99 retrieval latency
- Handles millions of chunks reliably

## 📜 License

MIT License. See [`LICENSE`](./LICENSE) for details.

---

### 💡 Why the name "memp3"?

A play on “memory” + “MP3” — `memp3` encodes structured memory into audio, going beyond compression into a new dimension of storage.

---

## 🧠 System Overview

See [`memp3-complete-plan.md`](./memp3-complete-plan.md) for the full system design and implementation roadmap.
