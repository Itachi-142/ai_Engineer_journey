# quiz_generator.py
import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

QUIZ_PROMPT = """You are a quiz generator. When given a topic, generate exactly 5 multiple choice questions.

Return ONLY a JSON array with this exact structure, no extra text:
[
    {
        "question": "question text",
        "options": ["A", "B", "C", "D"],
        "correct_answer": "exact text of correct option",
        "explanation": "why this is correct"
    }
]"""

def generate_quiz(topic: str) -> list:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        api_key=os.getenv("GROQ_API_KEY")
    )
    messages = [
        SystemMessage(content=QUIZ_PROMPT),
        HumanMessage(content=f"Generate a quiz on: {topic}")
    ]
    response = llm.invoke(messages)
    
    # strip markdown fences if present
    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    
    return json.loads(text.strip())