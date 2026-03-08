from rag_engine import RAGChatbot
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY").strip()

# Test dataset — question, expected keywords in answer, should_answer
# should_answer=False means the question is NOT in the document
TEST_CASES = [
    {
        "question": "What is an array?",
        "expected_keywords": ["collection", "elements", "index"],
        "should_answer": True
    },
    {
        "question": "What is a linked list?",
        "expected_keywords": ["links", "nodes", "connection"],
        "should_answer": True
    },
    {
        "question": "What is the capital of France?",
        "expected_keywords": [],
        "should_answer": False
    },
    {
        "question": "What is cloud computing?",
        "expected_keywords": [],
        "should_answer": False  # not in DSA doc
    }
]

def evaluate_answer(answer, expected_keywords, should_answer):
    """Score a single answer"""
    refused = "don't have that information" in answer.lower()

    if not should_answer:
        # Bot should refuse
        correct = refused
        return {
            "expected": "REFUSE",
            "got": "REFUSED" if refused else "ANSWERED",
            "correct": correct
        }
    else:
        # Bot should answer with expected keywords
        if refused:
            return {
                "expected": "ANSWER",
                "got": "REFUSED",
                "correct": False
            }
        keywords_found = [kw for kw in expected_keywords
                         if kw.lower() in answer.lower()]
        score = len(keywords_found) / len(expected_keywords) if expected_keywords else 1.0
        return {
            "expected": "ANSWER",
            "got": "ANSWERED",
            "correct": score >= 0.5,
            "keyword_score": f"{len(keywords_found)}/{len(expected_keywords)}"
        }

def run_evaluation(doc_path):
    """Run full evaluation on a document"""
    print(f"\n{'='*60}")
    print(f"RAG EVALUATION REPORT")
    print(f"Document: {doc_path}")
    print(f"{'='*60}\n")

    # Load chatbot and document
    chatbot = RAGChatbot(api_key=GROQ_API_KEY)
    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()
    chatbot.load_document(text, source=os.path.basename(doc_path))

    results = []
    for i, test in enumerate(TEST_CASES, 1):
        print(f"Test {i}: {test['question']}")
        response, sources = chatbot.chat(test["question"])
        result = evaluate_answer(
            response,
            test["expected_keywords"],
            test["should_answer"]
        )
        result["question"] = test["question"]
        result["answer"] = response
        result["num_sources"] = len(sources)
        results.append(result)

        status = "✅ PASS" if result["correct"] else "❌ FAIL"
        print(f"  Status: {status}")
        print(f"  Expected: {result['expected']} | Got: {result['got']}")
        if "keyword_score" in result:
            print(f"  Keyword score: {result['keyword_score']}")
        print(f"  Sources retrieved: {result['num_sources']}")
        print(f"  Answer: {response[:100]}...")
        print()

    # Summary
    passed = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"{'='*60}")
    print(f"SUMMARY: {passed}/{total} tests passed ({100*passed//total}%)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Point this at any .txt file in your project
    run_evaluation("test_doc.txt")
