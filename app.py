import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import Optional, List
import requests
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def get_pdf_text(pdf_docs: List[st.runtime.uploaded_file_manager.UploadedFile]) -> str:
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    logger.info(f"Extracted text: {text[:500]}...")  # Log the first 500 characters of extracted text
    return text

def get_text_chunks(text: str) -> List[str]:
    """Splits text into chunks of 10,000 characters with 1,000 character overlap."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    logger.info(f"Number of text chunks: {len(chunks)}")  # Log the number of chunks created
    return chunks

def get_vector_store(chunks: List[str], index_path: str = "faiss_index") -> None:
    """Generates embeddings for text chunks and saves them in a FAISS vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(index_path)
    logger.info(f"Vector store saved to {index_path}")

def get_conversational_chain() -> load_qa_chain:
    """Sets up the conversational chain with a specified prompt template."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n{context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.4)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

def get_url_content(url: str) -> Optional[str]:
    """Fetches the text content from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        logger.info(f"Fetched content from URL: {url}")
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching content from URL: {url}\n{e}")
        return None

def clear_chat_history() -> None:
    """Clears the chat history."""
    st.session_state.messages = [{"role": "assistant", "content": "You can upload multiple PDFs and ask me a question."}]

def perform_global_search(user_question: str) -> dict:
    """Performs a global search for a given user question using the generative model."""
    model = genai.GenerativeModel('gemini-pro')
    global_search_response = model.generate_content(user_question)
    return {"output_text": "".join(item.text for item in global_search_response)}

def user_input(user_question: str) -> dict:
    """Processes user input and returns the response from the conversational chain or global search."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)
    
    # Log retrieved documents for debugging
    logger.info(f"Retrieved documents for the question '{user_question}': {[doc.page_content[:500] for doc in docs]}")

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    if response['output_text'].lower() == 'answer is not available in the context':
        return perform_global_search(user_question)
    return response

def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")
    st.sidebar.title("Menu")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    
    try:
        pdf_docs = st.sidebar.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.sidebar.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.sidebar.success("PDFs processed successfully.")
    except Exception as e:
        st.sidebar.warning("Error processing PDF files. Please try again.")
    
    urls = st.sidebar.text_area("Enter multiple URLs (one per line):")
    if st.sidebar.button("Submit & Process URLs") and urls:
        with st.spinner("Processing URLs..."):
            url_contents = []
            for url in urls.splitlines():
                url_content = get_url_content(url)
                if url_content:
                    url_contents.append(url_content)
            all_text_chunks = [chunk for url_content in url_contents for chunk in get_text_chunks(url_content)]
            get_vector_store(all_text_chunks)
            st.sidebar.success("URLs processed successfully.")

    st.title("Chat with PDF files")
    st.write("Welcome to the chat!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "You can upload multiple PDFs and ask me a question."}]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                full_response = response.get('output_text', '')
                st.write(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
