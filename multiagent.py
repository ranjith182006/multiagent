# ================================
# MULTI-AGENT WITH 2 MODELS
# ================================

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain.memory import ConversationBufferMemory
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# ================================
# 1️⃣ LOAD MODELS
# ================================

# TinyLlama → Research Agent
research_pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=200
)

research_llm = HuggingFacePipeline(pipeline=research_pipe)

# FLAN-T5 → Summarizer Agent
summary_pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=150
)

summary_llm = HuggingFacePipeline(pipeline=summary_pipe)

# ================================
# 2️⃣ MEMORY SETUP
# ================================

research_memory = ConversationBufferMemory(memory_key="chat_history")
summary_memory = ConversationBufferMemory(memory_key="chat_history")

embedding = HuggingFaceEmbeddings()
vectorstore = FAISS.from_texts(["start"], embedding)

shared_memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever()
)

# ================================
# 3️⃣ PROMPTS
# ================================

research_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Answer the following question clearly and correctly.

Question: {question}

Answer:
"""
)

summary_prompt = PromptTemplate(
    input_variables=["data"],
    template="""
Give a short 2-line answer:

{data}
"""
)

# ================================
# 4️⃣ AGENTS
# ================================

research_agent = LLMChain(
    llm=research_llm,
    prompt=research_prompt,
    memory=research_memory
)

summary_agent = LLMChain(
    llm=summary_llm,
    prompt=summary_prompt,
    memory=summary_memory
)

# ================================
# 5️⃣ ORCHESTRATION
# ================================
def run_multi_agent(query):

    print("\n🔍 Research Agent...")
    research_output = research_agent.run(query)

    # CLEAN (important)
    research_output = research_output.strip()

    print("\n📖 Research Answer:\n", research_output)

    print("\n📚 Saving to shared memory...")
    shared_memory.save_context(
        {"input": query},
        {"output": research_output}
    )

    print("\n📝 Summarizer Agent...")
    summary_output = summary_agent.run(research_output)

    summary_output = summary_output.strip()

    print("\n📌 Final Summary:")

    return summary_output

# ================================
# 6️⃣ MAIN LOOP
# ================================

print("Dual Model Multi-Agent Ready (type 'exit' to quit)")

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    result = run_multi_agent(user_input)

    print("\nAI:", result)