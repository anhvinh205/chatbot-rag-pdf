# 🤖 Chatbot RAG PDF

Chatbot hỏi đáp thông minh từ tài liệu PDF, sử dụng kỹ thuật **RAG (Retrieval-Augmented Generation)** kết hợp mô hình ngôn ngữ **TinyLlama** chạy hoàn toàn trên máy cục bộ — không cần API key, không cần kết nối internet.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green)
![Chainlit](https://img.shields.io/badge/UI-Chainlit-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ Tính năng

- 📄 Đọc và xử lý file PDF bất kỳ
- 🔍 Tìm kiếm ngữ nghĩa (semantic search) với ChromaDB
- 🧠 Trả lời câu hỏi dựa trên nội dung tài liệu thực tế
- 💬 Giao diện chat thân thiện qua Chainlit
- ⚡ Cache model — chỉ load 1 lần, các phiên sau khởi động nhanh
- 🔒 Chạy hoàn toàn offline trên máy cá nhân

---

## 🏗️ Kiến trúc hệ thống

```
User câu hỏi
     │
     ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Chainlit   │────▶│   RAG Pipeline   │────▶│  TinyLlama 1.1B     │
│  (app.py)   │     │    (rag.py)      │     │  (llm.py)           │
└─────────────┘     └──────────────────┘     └─────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌──────────────────┐    ┌─────────────────────┐
     │  ChromaDB        │    │  all-MiniLM-L6-v2   │
     │  (Vector Store)  │    │  (Embedding Model)  │
     └──────────────────┘    └─────────────────────┘
```

| Thành phần | Công nghệ |
|---|---|
| Giao diện chat | Chainlit |
| Mô hình ngôn ngữ | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Embedding | sentence-transformers/all-MiniLM-L6-v2 |
| Vector database | ChromaDB |
| RAG framework | LangChain |
| Đọc PDF | PyPDF |

---

## 📁 Cấu trúc thư mục

```
chatbot-rag-pdf/
├── app.py              # Entry point — giao diện Chainlit
├── llm.py              # Khởi tạo mô hình ngôn ngữ TinyLlama
├── rag.py              # Pipeline RAG: load PDF, embedding, truy vấn
├── requirements.txt    # Các thư viện cần thiết
├── data/
│   └── documents.pdf   # ← Đặt file PDF của bạn vào đây
└── .chainlit/
    └── config.toml     # Cấu hình Chainlit
```

---

## ⚙️ Cài đặt & Chạy

### Yêu cầu

- Python 3.9 trở lên
- RAM tối thiểu 4GB (khuyến nghị 8GB)

### Bước 1 — Clone repository

```bash
git clone https://github.com/anhvinh205/chatbot-rag-pdf.git
cd chatbot-rag-pdf
```

### Bước 2 — Tạo môi trường ảo

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Bước 3 — Cài đặt thư viện

```bash
pip install -r requirements.txt
```

> ⚠️ Lần đầu chạy sẽ tự động tải model TinyLlama (~600MB) và embedding model (~90MB) từ HuggingFace. Cần kết nối internet cho bước này.

### Bước 4 — Thêm file PDF

Đặt file PDF của bạn vào thư mục `data/` và đổi tên thành `documents.pdf`:

```bash
cp /đường/dẫn/tới/file.pdf data/documents.pdf
```

### Bước 5 — Khởi chạy

```bash
chainlit run app.py
```

Mở trình duyệt tại **http://localhost:8000 và bắt đầu đặt câu hỏi! 🎉

## 💡 Cách hoạt động

1. **Load PDF** — PyPDF đọc và trích xuất toàn bộ văn bản từ file PDF.
2. **Chunking** — Văn bản được chia thành các đoạn nhỏ 500 ký tự (overlap 50 ký tự).
3. **Embedding** — Mỗi đoạn văn bản được chuyển thành vector số bằng `all-MiniLM-L6-v2`.
4. **Lưu vào ChromaDB** — Các vector được lưu vào cơ sở dữ liệu vector ChromaDB.
5. **Truy vấn** — Khi user đặt câu hỏi, hệ thống tìm 2 đoạn văn bản liên quan nhất.
6. **Sinh câu trả lời** — TinyLlama đọc context tìm được và trả lời câu hỏi.

## 🛠️ Tuỳ chỉnh
**Đổi file PDF:** Thay file tại `data/documents.pdf` hoặc sửa biến `PDF_PATH` trong `app.py`.

**Tăng độ dài câu trả lời:** Sửa `max_new_tokens` trong `llm.py` (mặc định: 256).

**Tăng số đoạn context:** Sửa `search_kwargs={"k": 2}` trong `rag.py` thành `k: 4` để lấy nhiều đoạn hơn.

**Đổi model mạnh hơn:** Thay `model_name` trong `llm.py`, ví dụ `mistralai/Mistral-7B-Instruct-v0.2` (cần GPU).

## 🐛 Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `FileNotFoundError: data/documents.pdf` | Chưa có file PDF | Đặt file PDF vào thư mục `data/` |
| `OutOfMemoryError` | RAM không đủ | Đóng bớt ứng dụng hoặc dùng máy RAM cao hơn |
| Model tải rất chậm lần đầu | Đang download ~700MB | Chờ hoàn tất, các lần sau sẽ nhanh hơn |
| Câu trả lời không liên quan | Chunk size chưa phù hợp | Thử tăng `chunk_size` lên 800-1000 trong `rag.py` |

## 📦 Thư viện sử dụng
- [LangChain](https://github.com/langchain-ai/langchain) — RAG framework
- [Chainlit](https://github.com/Chainlit/chainlit) — Chat UI
- [ChromaDB](https://github.com/chroma-core/chroma) — Vector database
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — TinyLlama model
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) — Embedding model
