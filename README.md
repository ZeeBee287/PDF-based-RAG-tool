# 📄 PDF RAG Q\&A App – Streamlit + FAISS + GROQ

**PDF RAG Q\&A App** is an AI-powered web app that:

* 📤 Accepts any PDF document,
* 🔍 Splits and embeds the content using `sentence-transformers` + `FAISS`,
* 🤖 Answers natural language questions using **Groq-hosted LLaMA 3**,
* 💬 Maintains a Q\&A history — all in an interactive **Streamlit** interface.

---

## 🚀 Features

* 📖 **PDF Text Extraction** using `PyPDF2`
* ✂️ **Automatic Chunking** for long documents
* 🧠 **Embedding** with `MiniLM` (Hugging Face)
* 🔎 **Fast Semantic Search** via `FAISS`
* 🤖 **Answer Generation** using LLaMA 3 on **Groq**
* 💬 **Chat History** tracked during session
* 🌐 **Colab-Compatible UI** with **LocalTunnel** link

---

## 📁 Project Structure

```
pdf-rag-streamlit-groq/
│
├── app.py               # Streamlit UI
├── rag_module.py        # RAG logic: extraction, embeddings, query
├── requirements.txt     # All Python dependencies
└── README.md            # You're here!
```

---

## 🔧 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🔑 API Keys Required

### Get GROQ API Key:

1. Visit [https://console.groq.com/keys](https://console.groq.com/keys)
2. Create a free account
3. Click **Create API Key** → Give it a name → Submit
4. Copy your key

Set your key as an environment variable (e.g. in Colab or `.env`):

```env
GROQ_API_KEY_1=your_groq_api_key_here
```

---

## 💻 How to Run in Google Colab

1. Upload `app.py`, `rag_module.py`, and `requirements.txt` to your Colab environment.
2. Install dependencies:

```python
!pip install -r requirements.txt
```

3. Set your API key in the notebook:

```python
import os
os.environ['GROQ_API_KEY_1'] = "your_groq_api_key"
```

4. Launch the app publicly:

```bash
!wget -q -O - ipv4.icanhazip.com  # Show Colab IP (optional)
!streamlit run app.py & npx localtunnel --port 8501
```

5. Open the `.loca.lt` link to start chatting with your PDF! 🎉

---

## 🤖 Models Used

| Purpose           | Model / Tool                             |
| ----------------- | ---------------------------------------- |
| Embedding Text    | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search     | `FAISS` (Flat L2 index)                  |
| Answer Generation | `LLaMA3-8B-8192` hosted on **Groq**      |

---

## ✨ Future Ideas

* 📑 Support **multiple PDFs** in a single session
* 🧾 Show **which chunk or page** the answer came from
* 📥 Export full Q\&A history
* 🎙️ Add voice input / narration

---

## 🧑‍💻 Author

**Zahra Batool :D**
