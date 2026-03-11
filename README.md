# RAG-Based Medical Assistant

This project is a Retrieval-Augmented Generation (RAG) medical assistant with:

- **Backend** (`backend/`): FastAPI service that:
  - Accepts PDF uploads, embeds them with `sentence-transformers/all-MiniLM-L6-v2`, and stores vectors in **ChromaDB**.
  - Prevents duplicate indexing using a **file hash**.
  - Exposes an API to ask questions grounded in the uploaded documents using Groq LLMs.
- **Frontend** (`frontend/`): Next.js 16 single-page UI that:
  - Lets you upload a PDF.
  - Shows processing status (Processed / Already indexed / Processing).
  - Provides an input to ask questions and displays answers with source documents.

---

## Prerequisites

- Python 3.10+ (backend uses **uv** for dependency management)
- Node.js 18+ and npm (for the Next.js frontend)
- A Groq API key

---

## Backend (FastAPI + uv)

From the `backend/` folder:

1. **Create a `.env` file** (this file is ignored by git):

```bash
cd backend
cp .env  # create .env
```

Add:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

2. **Install dependencies with uv**:

```bash
cd backend
uv sync
```

3. **Run the API server**:

```bash
uv run uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

FastAPI docs (for manual testing) will be available at:

- `http://localhost:8000/docs`

Key endpoints:

- `POST /upload_pdf` – upload a PDF, index it in ChromaDB, returns status:
  - `"processed"` or `"already_indexed"`
- `POST /ask` – ask a question about the indexed documents, returns:
  - `answer` and `sources` (list of document names)

---

## Frontend (Next.js)

From the `frontend/` folder:

1. **Install dependencies**:

```bash
cd frontend
npm install
```

2. **Configure backend URL** (optional in development):

Create `frontend/.env.local`:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

3. **Run the dev server**:

```bash
npm run dev
```

Then open:

- `http://localhost:3000`

You’ll see a single-page UI with:

- **Upload PDF** section:
  - Drag & Drop or click to select a PDF.
  - `Upload` button and status line:
    - ✔ Processed
    - ✔ Already indexed
    - ⏳ Processing
- **Ask Questions** section:
  - Input field for your question.
  - `Ask` button **enabled only after a PDF is successfully processed**.
- **Answer** section:
  - Model answer.
  - List of source PDF filenames.

---

## Project Notes

- Vector storage uses **ChromaDB**.
- Duplicate PDFs are detected via a **SHA-256 file hash** stored in document metadata.
- FastAPI endpoints (backend)
- Next.js single-page app (frontend)
