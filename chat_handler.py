import os
from typing import Tuple, List
import streamlit as st
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class ChatHandler:
    """Handles chat interactions with the Gemini LLM"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        
        if not GEMINI_AVAILABLE:
            st.error("❌ Google Generative AI library not installed. Please install: pip install google-generativeai")
            self.client = None
        elif not self.api_key:
            st.warning("⚠️ GEMINI_API_KEY environment variable not set. Please set it to use the chat functionality.")
            self.client = None
        else:
            try:
                # Configure Gemini API
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e:
                st.error(f"❌ Error initializing Gemini client: {str(e)}")
                self.client = None
    
    def get_response(self, user_question: str) -> Tuple[str, List[str]]:
        """
        Get response from Gemini LLM based on user question and document context
        
        Args:
            user_question (str): User's question about the document
            
        Returns:
            Tuple[str, List[str]]: (response_text, relevant_chunks)
        """
        if not self.client:
            return "❌ Gemini API client not available. Please check your GEMINI_API_KEY.", []
        
        if not user_question.strip():
            return "Please ask a question about the document.", []
        
        try:
            # Find relevant document chunks
            relevant_chunks = self.vector_store.similarity_search(user_question, k=3)
            
            if not relevant_chunks:
                return "I couldn't find relevant information in the document to answer your question.", []
            
            # Prepare context from relevant chunks
            context = "\n\n".join([f"Section {i+1}:\n{chunk}" for i, (chunk, _) in enumerate(relevant_chunks)])
            
            # Create the prompt for Gemini
            prompt = f"""You are a helpful AI assistant that answers questions based on the provided document content. 

Guidelines:
- Only answer based on the information provided in the document sections below
- If the document doesn't contain enough information to answer the question, say so clearly
- Be concise but comprehensive in your answers
- Reference specific sections when possible
- If you're unsure about something, acknowledge the uncertainty

Document sections:
{context}

Question: {user_question}

Please provide a helpful answer based on the document content above."""

            # Make API call to Gemini
            response = self.client.generate_content(prompt)
            
            answer = response.text
            source_chunks = [chunk for chunk, _ in relevant_chunks]
            
            return answer, source_chunks
            
        except Exception as e:
            error_msg = f"Error getting response from Gemini: {str(e)}"
            st.error(error_msg)
            return error_msg, []
    
    def summarize_document(self) -> str:
        """
        Generate a summary of the entire document
        
        Returns:
            str: Document summary
        """
        if not self.client:
            return "❌ Gemini API client not available."
        
        try:
            # Get all chunks
            all_chunks = self.vector_store.get_all_chunks()
            
            if not all_chunks:
                return "No document content available for summarization."
            
            # Limit context to avoid token limits
            context = "\n\n".join(all_chunks[:5])  # Use first 5 chunks for summary
            
            prompt = f"""Please provide a comprehensive summary of the following document. 
            Include the main topics, key points, and important conclusions.
            
            Document content:
            {context}"""
            
            response = self.client.generate_content(prompt)
            
            return response.text
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if the chat handler is properly configured"""
        return self.client is not None
    
    def get_suggested_questions(self) -> List[str]:
        """
        Generate suggested questions based on document content
        
        Returns:
            List[str]: List of suggested questions
        """
        if not self.client or not self.vector_store.get_all_chunks():
            return [
                "What is this document about?",
                "Can you summarize the main points?",
                "What are the key findings?",
                "What conclusions are presented?"
            ]
        
        try:
            # Get a sample of the document
            chunks = self.vector_store.get_all_chunks()
            sample_content = "\n".join(chunks[:2])  # First 2 chunks
            
            prompt = f"""Based on this document excerpt, suggest 4-5 relevant questions that a reader might ask about the full document. 
            Make the questions specific and useful.
            
            Document excerpt:
            {sample_content[:2000]}
            
            Return only the questions, one per line, without numbering."""
            
            response = self.client.generate_content(prompt)
            
            questions = response.text.strip().split('\n')
            # Clean up questions
            questions = [q.strip().strip('- ').strip() for q in questions if q.strip()]
            
            return questions[:5]  # Return max 5 questions
            
        except Exception as e:
            # Return default questions if API call fails
            return [
                "What is this document about?",
                "Can you summarize the main points?",
                "What are the key findings?",
                "What conclusions are presented?"
            ]
