import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import nest_asyncio

# Apply nest_asyncio to fix the event loop error in the Streamlit application
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

# Set the Google API key from environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Function to load the PDF and create a vector store
def get_vector_store(pdf_path="Human-Resource-Management.pdf"):
    if os.path.exists("faiss_index"):
        vector_store = FAISS.load_local("faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    else:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_documents(documents)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    return vector_store

# Function to create and return the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer. Your answer should be professional as you are HRM chatbot.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def main():
    st.set_page_config("HRM Chatbot")
    st.header("HRM Chatbot")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        vector_store = get_vector_store()
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True,
        )
        st.write("Reply: ", response["output_text"])

if __name__ == "__main__":
    main()
