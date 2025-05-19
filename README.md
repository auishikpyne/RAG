# RAG
┌──────────────────────┐          ┌──────────────────────┐
│      User Interface  │          │   Document Processing│
│   (Gradio Web App)   │          │      Pipeline        │
├──────────────────────┤          ├──────────────────────┤
│                      │          │                      │
│  ┌────────────────┐  │  PDF     │ 1. PDF Text Extraction│
│  │   PDF Upload   ├──┼─────────►│   (PyPDF2)          │
│  └────────────────┘  │          │                      │
│                      │          │ 2. Text Chunking     │
│  ┌────────────────┐  │          │   (NLTK Sentence     │
│  │  Question Input├──┼──┐       │    Tokenization)     │
│  └────────────────┘  │  │       │                      │
│                      │  │       │ 3. Metadata Tagging  │
│  ┌────────────────┐  │  │       │   (Chunk ID Tracking)│
│  │  Answer &      │◄─┼──┘       └──────────────────────┘
│  │  Sources Display│  │                    ▲
│  └────────────────┘  │                    │
└──────────┬───────────┘                    │
           │                                │
           │                                │
           ▼                                ▼
┌──────────────────────┐          ┌──────────────────────┐
│   AI Inference       │          │  Vector Knowledge    │
│   Engine             │          │  Base                │
├──────────────────────┤          ├──────────────────────┤
│                      │          │                      │
│ 1. Contextual Prompt │          │  FAISS Index         │
│    Construction      │          │  (Hugging Face       │
│                      │          │   MiniLM-L6-v2       │
│ 2. LLM Response      │◄─────────┤  Embeddings)         │
│    Generation        │          │                      │
│   (Mistral-7B)       │          │                      │
└──────────────────────┘          └──────────────────────┘

RAG
![Demo Screenshot](assets/chat_pdf.png)
