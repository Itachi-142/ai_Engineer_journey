from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import create_agent
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY").strip()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    api_key=GROQ_API_KEY
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = None

def load_doc(text, source="doc"):
    global vectorstore
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=75)
    doc = Document(page_content=text, metadata={"source": source})
    chunks = splitter.split_documents([doc])
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="agent_docs"
    )
    print(f"Loaded {len(chunks)} chunks")

@tool
def search_documents(query: str) -> str:
    """Search the loaded documents for information relevant to the query.
    Use this tool when the question is about the document content."""
    if vectorstore is None:
        return "No documents loaded yet."
    results = vectorstore.similarity_search(query, k=3)
    if not results:
        return "No relevant information found in documents."
    return "\n\n".join([doc.page_content for doc in results])

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.
    Use this tool for any arithmetic or math calculations.
    Input should be a valid Python math expression like '2 + 2' or '15 * 4'."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

tools = [search_documents, calculator]

system_prompt = """You are a helpful assistant with access to tools.
- For questions about documents, arrays, linked lists, binary search → use search_documents tool
- For math calculations → use calculator tool
- For general knowledge questions → answer directly"""

agent = create_agent(llm, tools, system_prompt=system_prompt)

if __name__ == "__main__":
    with open("test_doc.txt", "r") as f:
        text = f.read()
    load_doc(text)

    print("\n" + "="*50)
    print("AGENT TEST")
    print("="*50)

    test_questions = [
        "What is an array?",
        "What is 25 * 4?",
        "What is binary search?",
        "What is 100 + 200 + 300?",
        "What is the capital of France?"
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")
        try:
            response = agent.invoke({
                "messages": [("human", question)]
            })
            print(f"Answer: {response['messages'][-1].content}")
        except Exception as e:
            print(f"Error: {e}")