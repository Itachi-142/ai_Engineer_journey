# evaluator.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

JUDGE_PROMPT = """You are an answer evaluator. 
Given a question, the correct answer, and a student's answer:
- Score the student's answer from 1 to 5
- 5 = perfectly correct, 1 = completely wrong
- Give a one-line feedback explaining the score

Respond ONLY in this JSON format, no extra text:
{
    "score": <1-5>,
    "feedback": "one line explanation"
}"""

def evaluate_answer(question: str, correct_answer: str, user_answer: str) -> dict:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    prompt = f"""Question: {question}
Correct Answer: {correct_answer}
Student's Answer: {user_answer}"""

    messages = [
        SystemMessage(content=JUDGE_PROMPT),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    
    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    
    import json
    return json.loads(text.strip())