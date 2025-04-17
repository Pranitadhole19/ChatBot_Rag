# LLM Chatbot RAG Assistant

This project implements a chatbot using **Retrieval-Augmented Generation (RAG)** with a Large Language Model (LLM) and **LangChain**. The chatbot supports general interactions with the LLM and allows users to upload **PDF files** to ask questions related to the content of those PDFs.

---

## Features

- **General Chat**: Interact with the LLM for general knowledge or casual queries.
- **PDF Upload and Query**: Upload PDFs and ask questions based on the content within those documents.
- **Embedding & Vector Search**: Uses `sentence-transformers/all-MiniLM-L12-v2` and stores embeddings in a FAISS vector store.
- **Cosine Similarity**: Matches user queries with relevant chunks of PDFs.
- **Streamlit Interface**: Interactive UI for chatting and uploading PDFs.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YourGitHubUsername/rag_chatbot_pdf.git
cd rag_chatbot_pdf
```

### 2. Install Dependencies

#### (Recommended) Create a Virtual Environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

#### Install Required Packages

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### 📍 Local Jupyter Environment

Run the notebook directly using Jupyter Notebook or Jupyter Lab:

```bash
jupyter notebook chatbot_rag.ipynb
```

Follow the Streamlit link (if added in notebook) or test individual cells for PDF upload + chatbot responses.

---

## 📁 Project Structure

```
rag_chatbot_pdf/
│
├── chatbot_rag.ipynb         # Main Jupyter Notebook for RAG chatbot
├── README.md                 # Project documentation
├── requirements.txt          # List of all required libraries
├── utils.py                  # Utility file for embedding and retrieval
└── models/                   # Directory for model caching
```

---

## 📤 Usage

### 1. Upload PDFs

Use the uploader widget in the notebook (or UI) to add PDF files.

### 2. Ask Questions

Enter your question in the input field. The chatbot uses FAISS + LangChain to fetch relevant content from the PDFs and responds contextually.

---

## 🧠 Tech Stack

- Python
- LangChain
- FAISS
- Sentence-Transformers
- Streamlit
- PyPDFLoader

---

## 📝 Notes

- The model `all-MiniLM-L12-v2` is lightweight and fast for basic RAG tasks.
- You can scale it further by switching to other embedding models or using external vector DBs like Pinecone or Weaviate.

---

## 🤝 Contributing

Feel free to fork the repository and improve or customize the chatbot.

---

## 📃 License

This project is open-source and available under the [MIT License](LICENSE).
