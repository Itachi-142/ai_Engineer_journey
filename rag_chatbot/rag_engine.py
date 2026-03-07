from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

class RAGChatbot:
    def __init__(self, api_key):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            api_key=api_key
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.chat_history = []
        self.vectorstore = None

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=75,
            length_function=len
        )

    def load_document(self, text, source="uploaded_document"):
        """Load and index a document with metadata"""
        doc = Document(page_content=text, metadata={"source": source})
        chunks = self.splitter.split_documents([doc])

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name="rag_chatbot"
        )

        return len(chunks)

    def retrieve(self, query, k=3, score_threshold=1.0):
        """Retrieve and re-rank relevant chunks"""
        if self.vectorstore is None:
            return []

        # Get more candidates than needed, then re-rank
        results = self.vectorstore.similarity_search_with_score(query, k=k*2)

        # Filter by threshold
        filtered = [(doc, score) for doc, score in results if score < score_threshold]

        # Re-rank — sort by score ascending (lower L2 = more similar)
        reranked = sorted(filtered, key=lambda x: x[1])

        # Deduplicate by chunk_id
        seen = set()
        unique = []
        for doc in [doc for doc, score in reranked[:k]]:
            cid = doc.metadata.get("chunk_id")
            if cid not in seen:
                seen.add(cid)
                unique.append(doc)

        return unique

    def chat(self, user_message):
        """Generate a response using RAG"""
        relevant_docs = self.retrieve(user_message)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        if context:
            system_prompt = """You are a strict document-based assistant.
            Answer ONLY using the exact information present in the document context below.
            If the specific information requested is not explicitly stated in the context,
            respond with exactly: 'I don't have that information in the document.'
            Do NOT infer, expand, or add any information beyond what is in the context.
            Do NOT use your own knowledge under any circumstances.

            Document Context:
            {context}"""
        else:
            system_prompt = """You are a strict document-based assistant.
            The retrieved context does not contain relevant information for this query.
            You MUST respond with exactly: 'I don't have that information in the document.'
            Do NOT add any additional information, explanations, or knowledge.
            Do NOT say 'however'. Do NOT answer from your own knowledge."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({
            "context": context,
            "history": self.chat_history,
            "input": user_message
        })

        self.chat_history.append(HumanMessage(content=user_message))
        self.chat_history.append(AIMessage(content=response))

        return response, relevant_docs

    def reset(self):
        """Reset conversation and document"""
        self.chat_history = []
        self.vectorstore = None