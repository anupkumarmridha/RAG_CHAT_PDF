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
from typing import Optional
import requests

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# read all pdf files and return text 
def get_pdf_text(pdf_docs):
    """Extracts text from PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# split text into chunks
def get_text_chunks(text):
    """Splits text into chunks of 10,000 characters with 1,000 character overlap."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk and save them in a vector store
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.4)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def get_url_content(url: str) -> Optional[str]:
    """Fetches the text content from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from URL: {url}")
        print(e)
        return None

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "You can upload multiple pdfs and ask me a question"}]

def perform_global_search(user_question):
    # Perform global search here
    model = genai.GenerativeModel('gemini-pro')
    global_search_response = model.generate_content(user_question)
    text = ""
    for item in global_search_response:
        text += item.text
    return {"output_text": text}

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )
    # print(response["output_text"])
    if response['output_text'].lower() == 'answer is not available in the context':
        # Perform global search
        global_search_response = perform_global_search(user_question)
        # print(global_search_response)
        return global_search_response
    else:
        # print(response)
        return response


def main():
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="🤖"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        try:
            pdf_docs = st.file_uploader(
                "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
        except Exception as e:
            st.warning("Please upload a PDF file.")
        
        st.markdown("---")
        urls = st.text_area("Enter multiple URLs (one per line):")
        if st.button("Submit & Process URLs"):
            with st.spinner("Processing URLs..."):
                # Process multiple URLs
                url_contents = []
                for url in urls.splitlines():
                    url_content = get_url_content(url)
                    url_contents.append(url_content)

                # Combine text chunks from all URLs
                all_text_chunks = []
                for url_content in url_contents:
                    text_chunks = get_text_chunks(url_content)
                    all_text_chunks.extend(text_chunks)

                get_vector_store(all_text_chunks)
                st.success("URLs processed")
            
    # Main content area for displaying chat messages
    st.title("Chat with PDF files")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "You can upload multiple pdfs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
