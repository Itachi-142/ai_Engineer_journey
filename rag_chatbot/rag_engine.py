from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
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

    def load_document(self, text):
        """Load and index a document"""
        doc = Document(page_content=text)
        chunks = self.splitter.split_documents([doc])

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name="rag_chatbot"
        )

        return len(chunks)

    def retrieve(self, query, k=3, score_threshold=1.0):
        """Retrieve relevant chunks only if similarity score is below threshold"""
        if self.vectorstore is None:
            return []

        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        for doc, score in results:
            print(f"Score: {score:.4f} | Chunk: {doc.page_content[:60]}...")

        # L2 distance — lower score = more similar. Filter irrelevant chunks.
        relevant = [doc for doc, score in results if score < score_threshold]

        return relevant

    def chat(self, user_message):
        """Generate a response using RAG"""
        relevant_docs = self.retrieve(user_message)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        if context:
            system_prompt = """You are a helpful assistant that answers 
            questions based on the provided document context.
            Use ONLY the context to answer. If the answer is not in 
            the context, say 'I don't have that information in the document.'
            Be concise and accurate.
            
            Document Context:
            {context}"""
        else:
            system_prompt = """You are a helpful assistant.
            Answer based ONLY on the document provided.
            You do not have enough context to answer this question.
            Say: 'I don't have that information in the document.'"""

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