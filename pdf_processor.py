import io
import streamlit as st
try:
    import PyPDF2
except ImportError:
    try:
        import pdfplumber
    except ImportError:
        st.error("Please install PyPDF2 or pdfplumber: pip install PyPDF2 pdfplumber")

class PDFProcessor:
    """Handles PDF text extraction using available PDF libraries"""
    
    def __init__(self):
        self.available_libs = self._check_available_libraries()
    
    def _check_available_libraries(self):
        """Check which PDF libraries are available"""
        libs = []
        try:
            import PyPDF2
            libs.append('pypdf2')
        except ImportError:
            pass
        
        try:
            import pdfplumber
            libs.append('pdfplumber')
        except ImportError:
            pass
        
        return libs
    
    def extract_text(self, uploaded_file) -> str:
        """
        Extract text from uploaded PDF file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            str: Extracted text content
        """
        if not self.available_libs:
            raise Exception("No PDF processing library available. Please install PyPDF2 or pdfplumber.")
        
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Try pdfplumber first (generally better text extraction)
        if 'pdfplumber' in self.available_libs:
            return self._extract_with_pdfplumber(uploaded_file)
        elif 'pypdf2' in self.available_libs:
            return self._extract_with_pypdf2(uploaded_file)
        else:
            raise Exception("No suitable PDF library found")
    
    def _extract_with_pdfplumber(self, uploaded_file) -> str:
        """Extract text using pdfplumber"""
        import pdfplumber
        
        text_content = ""
        try:
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += f"\n--- Page {page_num + 1} ---\n"
                            text_content += page_text + "\n"
                    except Exception as e:
                        st.warning(f"Could not extract text from page {page_num + 1}: {str(e)}")
                        continue
        except Exception as e:
            raise Exception(f"Error reading PDF with pdfplumber: {str(e)}")
        
        return text_content.strip()
    
    def _extract_with_pypdf2(self, uploaded_file) -> str:
        """Extract text using PyPDF2"""
        import PyPDF2
        
        text_content = ""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n--- Page {page_num + 1} ---\n"
                        text_content += page_text + "\n"
                except Exception as e:
                    st.warning(f"Could not extract text from page {page_num + 1}: {str(e)}")
                    continue
                    
        except Exception as e:
            raise Exception(f"Error reading PDF with PyPDF2: {str(e)}")
        
        return text_content.strip()
    
    def get_document_info(self, uploaded_file) -> dict:
        """
        Get basic information about the PDF document
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            dict: Document information
        """
        info = {
            "filename": uploaded_file.name,
            "size": uploaded_file.size,
            "pages": 0,
            "text_length": 0
        }
        
        try:
            if 'pdfplumber' in self.available_libs:
                import pdfplumber
                uploaded_file.seek(0)
                with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                    info["pages"] = len(pdf.pages)
            elif 'pypdf2' in self.available_libs:
                import PyPDF2
                uploaded_file.seek(0)
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                info["pages"] = len(pdf_reader.pages)
                
            # Extract text to get length
            uploaded_file.seek(0)
            text = self.extract_text(uploaded_file)
            info["text_length"] = len(text)
            
        except Exception as e:
            st.warning(f"Could not extract document info: {str(e)}")
        
        return info
