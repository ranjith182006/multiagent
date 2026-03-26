# multiagent
using mutliple agent to chatbot
# 🤖 Multi-Agent Orchestration with Memory (Milestone 3)

## 📌 Project Overview

This project implements a **multi-agent system using LangChain** with memory and agent collaboration.

---

## 🎯 Objective

To build a system where multiple AI agents work together and use memory for improved responses.

---

## 🧠 Agents Used

### 🔍 Research Agent

* Generates detailed answers
* Uses TinyLlama model

### 📝 Summarizer Agent

* Converts detailed answers into short summaries
* Uses FLAN-T5 model

---

## 🔄 Workflow

User Input
→ Research Agent (detailed output)
→ Shared Memory Update
→ Summarizer Agent (short output)
→ Final Response

---

## 🧠 Memory System

### 1. Conversation Memory

* Stores chat history for each agent

### 2. Shared Memory (Vector Store)

* Stores important information using FAISS
* Enables future retrieval

---

## ⚙️ Technologies Used

* Python
* LangChain
* HuggingFace Transformers
* FAISS (Vector Database)

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## 💡 Example

Input:

```
What is Artificial Intelligence?
```

Output:

* Research Answer: Detailed explanation
* Summary: Short explanation

---

## 🏆 Key Features

* Multi-agent collaboration
* Memory-based reasoning
* Role-based agent design
* Sequential orchestration

---

## 📌 Conclusion

This project demonstrates how multiple AI agents can collaborate using memory to produce structured and meaningful outputs.

---
