# AgenticRAG - AI-Powered Agent with Smart Conversation and Retrieval-Augmented Generation

## 🚀 Overview

**AgenticRAG** is an advanced **AI-powered retrieval-augmented generation (RAG) Agent** designed to provide users with an interactive and intelligent conversational experience. Built using **LangChain**, it leverages an intelligent agent capable of retrieving relevant chunks from a custom `AI index Report 2025` based on the user's query. The agent is equipped with **memory** to handle ongoing conversations and can determine whether to perform a RAG process based on the query’s nature.

The application allows users to interact with the AI agent, either by asking questions or engaging in casual conversation. The agent responds promptly and smartly, while using RAG for information retrieval only when needed, ensuring efficiency.

## 📜 Table of Contents

- [Architecture](#-architecture)
- [Features](#-features)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Technologies Used](#-technologies-used)
- [Future Enhancements](#-future-enhancements)

---

## 🏗️ Architecture

<!-- * ADD Image -->

The system follows a **Retrieval-Augmented Generation (RAG)** architecture that combines both conversational AI and information retrieval, powered by LangChain. The process involves:

1. **Agent Creation**: The LangChain agent is set up with the ability to perform multiple tasks: casual conversation or RAG, depending on the query type.
2. **Memory & Context**: The agent is designed to remember prior interactions, allowing it to engage in context-aware conversations.
3. **Query Analysis**: When a user submits a query, the agent first analyzes whether it’s a general conversational query or one that requires retrieving detailed data (e.g., "Provide the table of contents of this report"). In addition, there is a query re-formulation part for better retrieval.
4. **RAG Execution**: If the query demands more specific information, the agent performs RAG to retrieve relevant document chunks from the **AI Index Report 2025**.
5. **Reasoning Steps**: The agent can provide detailed reasoning steps for RAG queries, depending on the user's preference. The agent decides whether to show intermediate results or skip to the final answer.

### **Basic RAG Architecture**

The RAG system is composed of:
- **Memory**: Stores prior interactions and updates context.
- **Retrieval Tool**: Retrieves relevant document chunks from the `AI Index Report 2025`.
- **Generation Tool**: Uses LLMs for generating responses, either as final answers or with reasoning steps.
  
---

## ✨ Features

✅ **Agentic RAG System**: The agent intelligently decides whether to perform a RAG process based on the query.\
✅ **Smart Memory**: The agent remembers previous interactions, allowing for context-aware conversations.\
✅ **Conditional RAG Execution**: If the query requires it, the agent performs RAG by retrieving relevant chunks from the AI Index Report 2025.\
✅ **Reasoning Steps**: Users can opt to see the intermediate reasoning steps used by the agent when processing the query.\
✅ **Natural Conversations**: The agent can handle casual conversational queries (e.g., "Hello, how are you?") without performing RAG.\
✅ **User-Controlled Reasoning**: The user can control whether to view the reasoning steps or just the final answer, providing flexibility in how the agent responds.\
✅ **Streamlit Interface**: A user-friendly interface that shows the agent’s responses and reasoning steps interactively.

---

## 🔧 Installation & Setup

### **1️⃣ Clone the Repository**

```sh
git clone https://github.com/user/AgenticRAG.git
cd AgenticRAG
```

### **2️⃣ Create and Activate Virtual Environment**
```sh
python -m venv agentic-rag-env
source agentic-rag-env/bin/activate  # On macOS/Linux
agentic-rag-env\Scripts\activate  # On Windows
```

### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **4️⃣ Run the Application**
```sh
streamlit run src/app.py
```

### **5️⃣ View the Interface**
After following the above instructions, you may expect to see this interface:
<!-- * ADD Image here -->

## 📖 Usage
1. Open the app in your browser (default: http://localhost:8501).

2. Enter your `COHERE_API_KEY` in its proprt place, both `trial` and `production` keys works correctly.

3. Select an **Embedding Model**  - Note: The `cohere/embed-v4.0` model, when used with a `trial_key`, is limited to processing `100,000` tokens per minute. This rate limit may cause slower processing for large documents like the `AI Index Report 2025` due to enforced waiting between batches. However, despite the slower throughput, it is much more efficient and accurate compared to `sentence-transformers/all-mpnet-base-v2` especially for high-quality semantic embeddings.

4. Upload the `2025 AI Index Report` in the file uploader area. Once, you uploaded it, it starts processing the PDFs, splitting it, creating chunks, and indexing it into a vector store. 

5. Engage in a conversation with the AI agent or ask it to retrieve information from the AI Index Report 2025.

6. The agent will automatically decide whether to perform RAG based on the query type.

7. If RAG is performed, the agent will retrieve relevant chunks and either present intermediate reasoning steps or the final answer based on your selection.

Example:

- Casual Conversation: If you ask, “Hello, how are you?”, the agent will greet you without performing any RAG.

- Specific Query: If you ask, “Provide to me the complete welcome message from the co-directors of the report”, the agent will perform RAG, retrieve relevant chunks, and generate an appropriate response.

## 🔧 Technologies Used
- **LangChain** - For building the intelligent agent with memory and retrieval-augmented generation capabilities.

- **Cohere** - LLMs used for generation and embedding tasks (providing *responses*).

- **Chroma** - Vector databases for storing and retrieving document chunks.

- **Streamlit** - Interactive UI for easy user interaction.

## 🔮 Future Enhancements
- ✅ Multi-model support for more flexible generation (e.g., OpenAI GPT models).

- ✅ Enhanced memory management for long-term, context-aware conversations.

- ✅ Fine-tuned retrieval with advanced filtering and re-ranking techniques.

- ✅ Multi-turn conversations with long-term memory and reasoning enhancements.

## 💬 Have Questions?
Reach out on GitHub or open an issue!

---
🎯 AgenticRAG - Your Intelligent AI Agent for Smart Conversations and Data Retrieval! 🚀