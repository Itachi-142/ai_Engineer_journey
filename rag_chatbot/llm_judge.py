from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag_engine import RAGChatbot
from dotenv import load_dotenv
import os
import json

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY").strip()

# Initialize judge LLM
judge_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,  # zero temp for consistent judgements
    api_key=GROQ_API_KEY
)

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a strict evaluation judge for RAG systems.
    Score the faithfulness of the answer on a scale of 1-5.
    
    Faithfulness means: every claim in the answer is supported by the context.
    
    Score 1: Answer contains claims not in context, or contradicts context
    Score 2: Answer mostly uses context but adds some outside knowledge
    Score 3: Answer uses context but includes minor unsupported inferences
    Score 4: Answer is mostly faithful with very minor extrapolations
    Score 5: Every claim is directly supported by the context
    
    Respond ONLY with a JSON object like this:
    {{"score": 4, "reason": "one sentence explanation"}}
    Nothing else. No preamble."""),
    ("human", """Context: {context}
    
Question: {question}
Answer: {answer}

Evaluate faithfulness:""")
])

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a strict evaluation judge for RAG systems.
    Score the relevance of the answer on a scale of 1-5.
    
    Relevance means: the answer directly addresses what was asked.
    
    Score 1: Answer is completely off-topic
    Score 2: Answer barely addresses the question
    Score 3: Answer partially addresses the question
    Score 4: Answer mostly addresses the question
    Score 5: Answer directly and completely addresses the question
    
    Respond ONLY with a JSON object like this:
    {{"score": 4, "reason": "one sentence explanation"}}
    Nothing else. No preamble."""),
    ("human", """Question: {question}
Answer: {answer}

Evaluate relevance:""")
])

def judge_answer(question, answer, context):
    """Score an answer on faithfulness and relevance"""
    
    # Score faithfulness
    faith_chain = FAITHFULNESS_PROMPT | judge_llm | StrOutputParser()
    faith_raw = faith_chain.invoke({
        "context": context,
        "question": question,
        "answer": answer
    })
    
    # Score relevance
    rel_chain = RELEVANCE_PROMPT | judge_llm | StrOutputParser()
    rel_raw = rel_chain.invoke({
        "question": question,
        "answer": answer
    })
    
    # Parse JSON responses
    try:
        faith_result = json.loads(faith_raw.strip())
    except json.JSONDecodeError:
        faith_result = {"score": 0, "reason": f"Parse error: {faith_raw}"}
    
    try:
        rel_result = json.loads(rel_raw.strip())
    except json.JSONDecodeError:
        rel_result = {"score": 0, "reason": f"Parse error: {rel_raw}"}
    
    return {
        "faithfulness": faith_result,
        "relevance": rel_result
    }

def run_llm_evaluation(doc_path):
    """Run LLM-as-Judge evaluation"""
    
    TEST_CASES = [
        {"question": "What is an array?", "should_answer": True},
        {"question": "What is a linked list?", "should_answer": True},
        {"question": "What is binary search?", "should_answer": True},
        {"question": "What is the capital of France?", "should_answer": False},
    ]
    
    print(f"\n{'='*60}")
    print(f"LLM-AS-JUDGE EVALUATION REPORT")
    print(f"Document: {doc_path}")
    print(f"{'='*60}\n")
    
    chatbot = RAGChatbot(api_key=GROQ_API_KEY)
    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()
    chatbot.load_document(text, source=os.path.basename(doc_path))
    
    total_faith = 0
    total_rel = 0
    count = 0
    
    for i, test in enumerate(TEST_CASES, 1):
        print(f"Test {i}: {test['question']}")
        
        response, sources = chatbot.chat(test["question"])
        context = "\n\n".join([doc.page_content for doc in sources])
        
        refused = "don't have that information" in response.lower()
        
        if not test["should_answer"]:
            status = "✅ PASS" if refused else "❌ FAIL"
            print(f"  Expected: REFUSE | Got: {'REFUSED' if refused else 'ANSWERED'}")
            print(f"  Status: {status}")
        else:
            scores = judge_answer(test["question"], response, context)
            faith_score = scores["faithfulness"]["score"]
            rel_score = scores["relevance"]["score"]
            total_faith += faith_score
            total_rel += rel_score
            count += 1
            
            print(f"  Faithfulness: {faith_score}/5 — {scores['faithfulness']['reason']}")
            print(f"  Relevance:    {rel_score}/5 — {scores['relevance']['reason']}")
            status = "✅ PASS" if faith_score >= 3 and rel_score >= 3 else "❌ FAIL"
            print(f"  Status: {status}")
        
        print(f"  Answer: {response[:100]}...")
        print()
    
    if count > 0:
        print(f"{'='*60}")
        print(f"AVERAGE FAITHFULNESS: {total_faith/count:.1f}/5")
        print(f"AVERAGE RELEVANCE:    {total_rel/count:.1f}/5")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    run_llm_evaluation("test_doc.txt")