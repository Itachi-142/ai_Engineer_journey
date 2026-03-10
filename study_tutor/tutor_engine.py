# tutor_engine.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
#from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
load_dotenv()

EXPLANATION_PROMPT = """You are a expert teacher. When given a concept:
1. Give a simple 2-3 sentence explanation
2. Provide one real-world analogy
3. Give one concrete example
Keep it concise and clear."""

SOCRATIC_PROMPT = """You are a Socratic tutor. Rules:
1. NEVER give the final answer directly
2. Guide the student with leading questions and hints
3. If the student asks for the answer, give another hint instead
4. Acknowledge correct thinking, redirect incorrect thinking
5. One question at a time — do not overwhelm"""

def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY")
    )

def explain_concept(topic: str) -> str:
    llm = get_llm()
    messages = [
        SystemMessage(content=EXPLANATION_PROMPT),
        HumanMessage(content=f"Explain: {topic}")
    ]
    response = llm.invoke(messages)
    return response.content

def socratic_chat(user_message: str, chat_history: list) -> tuple[str, list]:
    llm = get_llm()
    messages = [SystemMessage(content=SOCRATIC_PROMPT)]
    
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    
    messages.append(HumanMessage(content=user_message))
    response = llm.invoke(messages)
    
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": response.content})
    
    return response.content, chat_history