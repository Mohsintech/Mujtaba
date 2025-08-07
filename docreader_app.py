import streamlit as st
import re
import nltk
from collections import Counter, defaultdict
import heapq
import numpy as np
from typing import List, Tuple
import requests
from urllib.parse import urlparse
import PyPDF2
import docx
import io
import time

# Download required NLTK data (only once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt')
    except:
        pass

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except:
        pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords')
    except:
        pass

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

class TextSummarizer:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            # Fallback stopwords list if NLTK stopwords are not available
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
                'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
                'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above',
                'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
                'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
                'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
            }
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespaces and special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', ' ', text)
        return text.strip()
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            # Fallback to simple sentence splitting if NLTK tokenizer fails
            sentences = re.split(r'[.!?]+', text)
        return [sent.strip() for sent in sentences if len(sent.strip()) > 10]
    
    def calculate_word_frequencies(self, text: str) -> dict:
        """Calculate word frequencies"""
        try:
            words = word_tokenize(text.lower())
        except LookupError:
            # Fallback to simple word splitting if NLTK tokenizer fails
            words = re.findall(r'\b\w+\b', text.lower())
        
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        return Counter(words)
    
    def score_sentences(self, sentences: List[str], word_freq: dict) -> dict:
        """Score sentences based on word frequencies and other factors"""
        sentence_scores = defaultdict(float)
        
        for i, sentence in enumerate(sentences):
            try:
                words = word_tokenize(sentence.lower())
            except LookupError:
                words = re.findall(r'\b\w+\b', sentence.lower())
            
            word_count = len([word for word in words if word.isalnum()])
            
            if word_count == 0:
                continue
            
            # Base score from word frequencies
            for word in words:
                if word in word_freq and word.isalnum():
                    sentence_scores[i] += word_freq[word]
            
            # Normalize by sentence length
            sentence_scores[i] = sentence_scores[i] / word_count
            
            # Position-based scoring (beginning and end are often important)
            total_sentences = len(sentences)
            if i < total_sentences * 0.3:  # First 30%
                sentence_scores[i] *= 1.2
            elif i > total_sentences * 0.7:  # Last 30%
                sentence_scores[i] *= 1.1
            
            # Length bonus for moderate-length sentences
            if 10 <= word_count <= 30:
                sentence_scores[i] *= 1.1
        
        return sentence_scores
    
    def extractive_summarize(self, text: str, num_sentences: int = 3) -> str:
        """Create extractive summary"""
        if not text or len(text.strip()) < 50:
            return "Text too short to summarize effectively."
        
        cleaned_text = self.clean_text(text)
        sentences = self.extract_sentences(cleaned_text)
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        word_freq = self.calculate_word_frequencies(cleaned_text)
        sentence_scores = self.score_sentences(sentences, word_freq)
        
        # Get top sentences
        top_sentences_idx = heapq.nlargest(num_sentences, sentence_scores.keys(), 
                                         key=lambda k: sentence_scores[k])
        
        # Sort by original order
        top_sentences_idx.sort()
        summary_sentences = [sentences[i] for i in top_sentences_idx]
        
        return ' '.join(summary_sentences)
    
    def abstractive_summarize(self, text: str, max_length: int = 150) -> str:
        """Simple abstractive summarization using key phrases"""
        if not text or len(text.strip()) < 50:
            return "Text too short to summarize effectively."
        
        # This is a simplified approach - in practice, you'd use transformer models
        sentences = self.extract_sentences(text)
        word_freq = self.calculate_word_frequencies(text)
        
        # Get most important words
        important_words = [word for word, freq in word_freq.most_common(20)]
        
        # Find sentences with most important words
        key_sentences = []
        for sentence in sentences[:10]:  # Look at first 10 sentences
            try:
                words = word_tokenize(sentence.lower())
            except LookupError:
                words = re.findall(r'\b\w+\b', sentence.lower())
            importance_score = sum(1 for word in words if word in important_words)
            key_sentences.append((importance_score, sentence))
        
        key_sentences.sort(reverse=True)
        
        # Build summary from key phrases
        summary_parts = []
        current_length = 0
        
        for score, sentence in key_sentences:
            if current_length + len(sentence) <= max_length:
                summary_parts.append(sentence)
                current_length += len(sentence)
            if len(summary_parts) >= 2:
                break
        
        return ' '.join(summary_parts) if summary_parts else sentences[0]

def extract_text_from_pdf(file) -> str:
    """Extract text from PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(file) -> str:
    """Extract text from DOCX"""
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def extract_text_from_url(url: str) -> str:
    """Extract text from URL (basic web scraping)"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Basic HTML tag removal
        text = re.sub(r'<[^>]+>', ' ', response.text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    except Exception as e:
        st.error(f"Error fetching URL: {str(e)}")
        return ""

def main():
    st.set_page_config(
        page_title="Advanced Text Summarizer",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .summary-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .input-box {
        background: #f8f9ff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e1e8ed;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📝 Advanced Text Summarizer</h1>
        <p>Extract key insights from any text using AI-powered summarization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize summarizer
    summarizer = TextSummarizer()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Input method selection
    input_method = st.sidebar.selectbox(
        "Choose input method:",
        ["Direct Text", "Upload File", "URL/Website"]
    )
    
    # Summarization settings
    st.sidebar.subheader("Summary Settings")
    summary_type = st.sidebar.selectbox(
        "Summary Type:",
        ["Extractive", "Abstractive"]
    )
    
    if summary_type == "Extractive":
        num_sentences = st.sidebar.slider("Number of sentences:", 1, 10, 3)
    else:
        max_length = st.sidebar.slider("Max summary length:", 50, 300, 150)
    
    # Advanced options
    with st.sidebar.expander("🔧 Advanced Options"):
        show_metrics = st.checkbox("Show text metrics", value=True)
        show_keywords = st.checkbox("Show key phrases", value=True)
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📄 Input")
        
        text_to_summarize = ""
        
        if input_method == "Direct Text":
            text_to_summarize = st.text_area(
                "Enter or paste your text here:",
                height=300,
                placeholder="Paste your text here to get a summary..."
            )
        
        elif input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['txt', 'pdf', 'docx'],
                help="Supported formats: TXT, PDF, DOCX"
            )
            
            if uploaded_file:
                file_extension = uploaded_file.name.lower().split('.')[-1]
                
                with st.spinner('Processing file...'):
                    if file_extension == 'pdf':
                        text_to_summarize = extract_text_from_pdf(uploaded_file)
                    elif file_extension == 'docx':
                        text_to_summarize = extract_text_from_docx(uploaded_file)
                    elif file_extension == 'txt':
                        text_to_summarize = uploaded_file.read().decode('utf-8')
                
                if text_to_summarize:
                    st.success(f"✅ File '{uploaded_file.name}' loaded successfully!")
                    with st.expander("Preview text"):
                        st.text_area("File content:", value=text_to_summarize[:1000] + "...", height=200, disabled=True)
        
        elif input_method == "URL/Website":
            url = st.text_input("Enter URL:", placeholder="https://example.com/article")
            
            if url:
                if st.button("🌐 Fetch from URL"):
                    with st.spinner('Fetching content...'):
                        text_to_summarize = extract_text_from_url(url)
                        if text_to_summarize:
                            st.success("✅ Content fetched successfully!")
                            with st.expander("Preview text"):
                                st.text_area("Fetched content:", value=text_to_summarize[:1000] + "...", height=200, disabled=True)
    
    with col2:
        st.subheader("✨ Summary & Analysis")
        
        if text_to_summarize and len(text_to_summarize.strip()) > 50:
            # Generate summary
            with st.spinner('Generating summary...'):
                if summary_type == "Extractive":
                    summary = summarizer.extractive_summarize(text_to_summarize, num_sentences)
                else:
                    summary = summarizer.abstractive_summarize(text_to_summarize, max_length)
            
            # Display summary
            st.markdown('<div class="summary-box">', unsafe_allow_html=True)
            st.markdown(f"**{summary_type} Summary:**")
            st.write(summary)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Metrics
            if show_metrics:
                original_words = len(text_to_summarize.split())
                summary_words = len(summary.split())
                compression_ratio = round((1 - summary_words/original_words) * 100, 1) if original_words > 0 else 0
                
                col_m1, col_m2, col_m3 = st.columns(3)
                
                with col_m1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>{original_words:,}</h3>
                        <p>Original Words</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>{summary_words}</h3>
                        <p>Summary Words</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>{compression_ratio}%</h3>
                        <p>Compression</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Key phrases
            if show_keywords:
                word_freq = summarizer.calculate_word_frequencies(text_to_summarize)
                top_words = [word for word, freq in word_freq.most_common(10)]
                
                st.subheader("🔑 Key Phrases")
                st.write(", ".join(top_words))
            
            # Download options
            st.subheader("💾 Download")
            
            summary_text = f"""
SUMMARY REPORT
==============
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
Summary Type: {summary_type}
Original Length: {len(text_to_summarize.split())} words
Summary Length: {len(summary.split())} words
Compression: {compression_ratio if show_metrics else 'N/A'}%

SUMMARY:
{summary}

ORIGINAL TEXT:
{text_to_summarize}
            """
            
            st.download_button(
                label="📄 Download Summary Report",
                data=summary_text,
                file_name=f"summary_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        else:
            st.info("👆 Enter some text to get started with summarization!")
            st.markdown("""
            **Supported Features:**
            - 📝 Direct text input
            - 📁 File upload (TXT, PDF, DOCX)
            - 🌐 Web content extraction
            - 🤖 Extractive & Abstractive summarization
            - 📊 Text analytics and metrics
            - 💾 Download summary reports
            """)

if __name__ == "__main__":
    main()
