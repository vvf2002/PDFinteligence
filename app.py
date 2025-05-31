import streamlit as st
import os
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from chat_handler import ChatHandler
from vector_store import VectorStore

# Page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Chat with Your PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables from .env file
load_dotenv()

def initialize_session_state():
    """Initialize session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_handler" not in st.session_state:
        st.session_state.chat_handler = None
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None

def main():
    initialize_session_state()
    
    st.title("📄 Chat with Your PDF")
    st.markdown("Upload a PDF document and ask questions about its content using AI!")
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📁 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to start chatting with its content"
        )
        
        if uploaded_file is not None:
            if st.session_state.pdf_name != uploaded_file.name:
                # New file uploaded, reset state
                st.session_state.chat_history = []
                st.session_state.pdf_processed = False
                st.session_state.vector_store = None
                st.session_state.chat_handler = None
                st.session_state.pdf_name = uploaded_file.name
                
                # Process the PDF
                with st.spinner("Processing PDF... This may take a moment."):
                    try:
                        # Initialize PDF processor
                        pdf_processor = PDFProcessor()
                        
                        # Extract text from PDF
                        text_content = pdf_processor.extract_text(uploaded_file)
                        
                        if not text_content.strip():
                            st.error("❌ No readable text found in the PDF. Please ensure the PDF contains text content.")
                            return
                        
                        # Create vector store
                        vector_store = VectorStore()
                        st.session_state.vector_store = vector_store.create_vector_store(text_content)
                        
                        # Initialize chat handler
                        st.session_state.chat_handler = ChatHandler(st.session_state.vector_store)
                        
                        st.session_state.pdf_processed = True
                        st.success("✅ PDF processed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing PDF: {str(e)}")
                        return
            
            # Display PDF info
            if st.session_state.pdf_processed:
                st.success(f"📄 **Current PDF:** {uploaded_file.name}")
                st.info(f"📊 **Size:** {uploaded_file.size:,} bytes")
        
        # API Key status
        st.header("🔑 API Configuration")
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            st.success("✅ Gemini API key configured")
        else:
            st.warning("⚠️ GEMINI_API_KEY environment variable not set. Please add it to your .env file.")
    
    # Main chat interface
    if not st.session_state.pdf_processed:
        st.info("👆 Please upload a PDF file from the sidebar to start chatting!")
        
        # Display example questions
        st.subheader("💡 Example Questions You Can Ask:")
        st.markdown("""
        - What is this document about?
        - Can you summarize the main points?
        - What are the key findings mentioned?
        - Tell me about the conclusions
        - What recommendations are provided?
        """)
        
    else:
        # Chat interface
        st.subheader(f"💬 Chat about: {st.session_state.pdf_name}")
        
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    if "sources" in message:
                        with st.expander("📚 Relevant Document Sections"):
                            for j, source in enumerate(message["sources"], 1):
                                st.markdown(f"**Section {j}:**")
                                st.markdown(f"```\n{source}\n```")
        
        # Chat input
        user_question = st.chat_input("Ask a question about your PDF...")
        
        if user_question:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_question)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response, sources = st.session_state.chat_handler.get_response(user_question)
                        
                        st.write(response)
                        
                        # Display sources if available
                        if sources:
                            with st.expander("📚 Relevant Document Sections"):
                                for i, source in enumerate(sources, 1):
                                    st.markdown(f"**Section {i}:**")
                                    st.markdown(f"```\n{source}\n```")
                        
                        # Add assistant message to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response,
                            "sources": sources
                        })
                        
                    except Exception as e:
                        error_msg = f"❌ Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg
                        })
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()
