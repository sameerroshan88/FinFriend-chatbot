from pymongo import MongoClient
import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import requests
from newsapi.newsapi_client import NewsApiClient
import google.generativeai as genai
import tempfile
import os
from io import BytesIO
import spacy
import re
import PyPDF2
from textblob import TextBlob
from googletrans import Translator
from gtts import gTTS
from gtts.lang import tts_langs
import datetime

# API Keys
NEWS_API_KEY = "63eff4841cc846d4b78184769bd799d8"
GOOGLE_API_KEY = "AIzaSyDyl3vZr00KysZI-TsxRNTLfAuKGOzvm1s"

# Initialize MongoDB client
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["Finchathistory"]
chat_collection = db["Chat history"]

# Load NLP Model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Initialize APIs
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')
translator = Translator()

# Audio recording parameters
SAMPLE_RATE = 44100
CHANNELS = 1
DTYPE = 'float32'

# List of major financial companies
COMPANY_LIST = [
    "Apple", "Microsoft", "Google", "Amazon", "Tesla", "Meta",
    "Netflix", "Nvidia", "JPMorgan", "Goldman Sachs", "Morgan Stanley",
    "Bank of America", "Citigroup", "Wells Fargo", "Berkshire Hathaway"
]

# List of finance-related keywords
FINANCE_KEYWORDS = [
    "finance", "stock", "investment", "market", "trading", "economy", 
    "bank", "crypto", "currency", "wealth", "shares", "interest rates", 
    "inflation", "financial news", "business", "Wall Street", "S&P 500", "Bitcoin"
]

# Complete list of all supported languages
SUPPORTED_LANGUAGES = {
    'English': 'en', 'Hindi': 'hi', 'Bengali': 'bn', 'Telugu': 'te',
    'Marathi': 'mr', 'Tamil': 'ta', 'Urdu': 'ur', 'Gujarati': 'gu',
    'Kannada': 'kn', 'Malayalam': 'ml', 'Odia': 'or', 'Punjabi': 'pa',
    'Assamese': 'as', 'Bhojpuri': 'bh', 'Kashmiri': 'ks', 'Sanskrit': 'sa',
    'Sindhi': 'sd', 'Maithili': 'mai', 'Santali': 'sat', 'Konkani': 'kok',
    'Dogri': 'doi', 'Manipuri': 'mni', 'Bodo': 'brx', 'Nepali': 'ne'
}

# Multilingual financial terms database (partial)
MULTILINGUAL_FINANCE_TERMS = {
    'stock': {
        'hi': 'शेयर', 'te': 'స్టాక్', 'ta': 'பங்கு', 'bn': 'স্টক', 
        'mr': 'स्टॉक', 'gu': 'સ્ટોક', 'kn': 'ಸ್ಟಾಕ್', 'ml': 'സ്റ്റോക്ക്'
    },
    'investment': {
        'hi': 'निवेश', 'te': 'పెట్టుబడి', 'ta': 'முதலீடு', 'bn': 'বিনিয়োগ',
        'mr': 'गुंतवणूक', 'gu': 'એન્વેસ્ટમેન્ટ', 'kn': 'ಹೂಡಿಕೆ', 'ml': 'നിക്ഷേപം'
    }
}

# Multilingual company names database (partial)
MULTILINGUAL_COMPANY_NAMES = {
    'Apple': {
        'hi': 'एप्पल', 'te': 'ఆపిల్', 'ta': 'ஆப்பிள்', 'bn': 'অ্যাপল',
        'mr': 'ॲपल', 'gu': 'એપલ', 'kn': 'ಆಪಲ್', 'ml': 'ആപ്പിൾ'
    },
    'Microsoft': {
        'hi': 'माइक्रोसॉफ्ट', 'te': 'మైక్రోసాఫ్ట్', 'ta': 'மைக்ரோசாப்ட்',
        'bn': 'মাইক্রোসফট', 'mr': 'मायक्रोसॉफ्ट', 'gu': 'માઇક્રોસોફ્ટ',
        'kn': 'ಮೈಕ್ರೋಸಾಫ್ಟ್', 'ml': 'മൈക്രോസോഫ്റ്റ്'
    }
}

# Session state initialization
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "selected_lang" not in st.session_state:
    st.session_state.selected_lang = 'en'
if "user_id" not in st.session_state:
    st.session_state.user_id = None

def record_audio(duration=10):
    st.write(f"🎤 Recording for {duration} seconds...")
    recording = sd.rec(int(duration * SAMPLE_RATE),
                      samplerate=SAMPLE_RATE,
                      channels=CHANNELS,
                      dtype=DTYPE)
    sd.wait()
    return recording

def save_recording(recording):
    with BytesIO() as buffer:
        sf.write(buffer, recording, SAMPLE_RATE, format='WAV')
        buffer.seek(0)
        return buffer.getvalue()

def transcribe_audio(audio_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        
        with open(temp_path, "rb") as f:
            audio = genai.upload_file(f, mime_type="audio/wav")
            response = model.generate_content(["Transcribe this financial discussion:", audio])
        os.unlink(temp_path)
        return response.text
    except Exception as e:
        st.error(f"❌ Error in transcription: {str(e)}")
        return None

def is_finance_related(query):
    if not query:
        return False
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in FINANCE_KEYWORDS):
        return True
    doc = nlp(query)
    for ent in doc.ents:
        if ent.text in COMPANY_LIST:
            return True
    return False

def analyze_sentiment(text):
    try:
        polarity = TextBlob(text).sentiment.polarity
        if polarity == 0 and not text.isascii():
            translated = translator.translate(text, dest='en').text
            polarity = TextBlob(translated).sentiment.polarity
        if polarity > 0.1:
            return "🟢 Positive"
        elif polarity < -0.1:
            return "🔴 Negative"
        else:
            return "⚪ Neutral"
    except Exception as e:
        st.error(f"Sentiment analysis error: {str(e)}")
        return "⚪ Unknown"

def translate_text(text, target_lang):
    try:
        if target_lang == 'en':
            return text
        translated = translator.translate(text, dest=target_lang)
        return translated.text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

def summarize_news(title, description, target_lang='en'):
    try:
        prompt = f"Create a concise 2-sentence summary in English of this financial news:\nTitle: {title}\nDescription: {description}"
        response = model.generate_content(prompt)
        english_summary = response.text.strip()
        if target_lang != 'en':
            return translate_text(english_summary, target_lang)
        return english_summary
    except Exception as e:
        st.error(f"Summarization error: {str(e)}")
        return translate_text("⚠ Summary not available.", target_lang)

def extract_financial_entities(query, lang_code='en'):
    if lang_code == 'en':
        doc = nlp(query)
        companies = [ent.text for ent in doc.ents if ent.text in MULTILINGUAL_COMPANY_NAMES]
        finance_terms = [term for term in FINANCE_KEYWORDS if term in query.lower()]
        return (companies, finance_terms)
    
    try:
        translated = translator.translate(query, dest='en').text
        doc = nlp(translated)
        companies = [ent.text for ent in doc.ents if ent.text in MULTILINGUAL_COMPANY_NAMES]
        native_finance_terms = []
        for eng_term, translations in MULTILINGUAL_FINANCE_TERMS.items():
            if lang_code in translations and translations[lang_code] in query:
                native_finance_terms.append(eng_term)
        finance_terms = list(set(
            [term for term in FINANCE_KEYWORDS if term in translated.lower()] +
            native_finance_terms
        ))
        return (companies, finance_terms)
    except Exception as e:
        st.error(f"Entity extraction error: {str(e)}")
        return ([], [])

def get_news_search_query(companies, finance_terms):
    if not companies and not finance_terms:
        return None
    if companies:
        return " OR ".join(companies)
    return " OR ".join(finance_terms[:3])

def fetch_financial_news(query, lang_code='en'):
    try:
        companies, finance_terms = extract_financial_entities(query, lang_code)
        search_query = get_news_search_query(companies, finance_terms)
        
        if not search_query:
            return translate_text("No financial entities found in query", lang_code)
        
        st.write(translate_text(f"🔍 Searching for: {search_query}", lang_code))
        
        articles = newsapi.get_everything(
            q=search_query,
            language='en',
            sort_by='relevancy',
            page_size=5
        ).get('articles', [])
        
        return [{
            'title': article.get('title', 'No title'),
            'description': article.get('description', 'No description'),
            'url': article.get('url', '#'),
            'source': article.get('source', {}).get('name', 'Unknown'),
            'publishedAt': article.get('publishedAt', 'Unknown date'),
            'sentiment': analyze_sentiment(f"{article.get('title', '')} {article.get('description', '')}")
        } for article in articles]
    except Exception as e:
        st.error(f"News fetch error: {str(e)}")
        return translate_text("Error fetching news", lang_code)

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def text_to_speech(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None

def save_chat_to_mongodb(user_id, user_message, bot_response):
    chat_document = {
        "user_id": user_id,
        "user_message": user_message,
        "bot_response": bot_response,
        "timestamp": datetime.datetime.now().isoformat(),
        "language": st.session_state.selected_lang
    }
    chat_collection.insert_one(chat_document)

def get_chat_history_from_mongodb(user_id):
    chats = chat_collection.find({"user_id": user_id}).sort("timestamp", -1)
    return list(chats)

def show_login_page():
    login_col1, login_col2, login_col3 = st.columns([1, 1.2, 1])
    with login_col2:
        st.title("FINFRIEND CHATBOT - Login")
        st.markdown("<div class='login-welcome-box'>Welcome back! Please log in.</div>", unsafe_allow_html=True)

        with st.form(key="login_form"):
            email = st.text_input("Email", placeholder="your.email@example.com", label_visibility="collapsed")
            password = st.text_input("Password", type="password", placeholder="••••••••", label_visibility="collapsed")
            submit = st.form_submit_button("Login")
            
            if submit:
                if email and password and "@" in email and len(password) >= 8:
                    st.session_state.logged_in = True
                    st.session_state.user_id = email
                    st.rerun()
                else:
                    st.error("Please enter a valid email and password (at least 8 characters)")

def show_main_app():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('image.png');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("📈 FINFRIEND CHATBOT")

    with st.sidebar:
        st.header("Settings")
        selected_lang_name = st.selectbox(
            "Select Language",
            list(SUPPORTED_LANGUAGES.keys()),
            index=list(SUPPORTED_LANGUAGES.values()).index(st.session_state.selected_lang)
        )
        st.session_state.selected_lang = SUPPORTED_LANGUAGES[selected_lang_name]
        
        st.header("Chat History")
        if not st.session_state.user_id:
            st.info("No user logged in")
        else:
            chats = get_chat_history_from_mongodb(st.session_state.user_id)
            if not chats:
                st.info("No chat history yet")
            else:
                for chat in chats:
                    st.markdown(f"<div class='chat-message-user'><b>You:</b> {chat['user_message']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='chat-message-bot'><b>Assistant:</b> {chat['bot_response']}</div>", unsafe_allow_html=True)
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.rerun()

    target_lang_code = st.session_state.selected_lang

    st.sidebar.header(translate_text("📂 Upload Financial Report", target_lang_code))
    uploaded_file = st.sidebar.file_uploader(
        translate_text("Upload a PDF file", target_lang_code), 
        type=["pdf"]
    )

    if uploaded_file:
        st.sidebar.success(translate_text("✅ File uploaded successfully!", target_lang_code))
        extracted_text = extract_text_from_pdf(uploaded_file)
        question_about_pdf = st.text_input(
            translate_text("Ask a question about the document:", target_lang_code),
            key="pdf_question"
        )

        if st.button(translate_text("Submit PDF Question", target_lang_code), key="submit_pdf"):
            if question_about_pdf:
                with st.spinner(translate_text("🤖 Analyzing document...", target_lang_code)):
                    english_question = translate_text(question_about_pdf, 'en')
                    response = model.generate_content(
                        f"Analyze this financial document and answer the question in English: {english_question}\n\n{extracted_text}"
                    )
                    translated_response = translate_text(response.text, target_lang_code)
                    
                    st.subheader(translate_text("💡 AI Response", target_lang_code))
                    st.write(translated_response)
                    
                    save_chat_to_mongodb(
                        st.session_state.user_id,
                        question_about_pdf,
                        translated_response
                    )
                    
                    audio_file = text_to_speech(translated_response, target_lang_code)
                    if audio_file:
                        st.audio(audio_file, format="audio/mp3")
                        os.unlink(audio_file)

    # Use a form for the query input
    with st.form(key="query_form"):
        user_query = st.text_input(
            translate_text("🔍 Enter a financial question or company name:", target_lang_code),
            key="user_query"
        )
        
        duration = st.slider(
            translate_text("🎤 Set Recording Duration (seconds)", target_lang_code), 
            5, 30, 10
        )
        
        record_button = st.form_submit_button(translate_text("🎙 Start Recording", target_lang_code))
        submit_button = st.form_submit_button(translate_text("Submit Query", target_lang_code))

        if record_button:
            recording = record_audio(duration)
            audio_bytes = save_recording(recording)
            st.audio(audio_bytes, format="audio/wav")
            with st.spinner(translate_text("⏳ Processing your query...", target_lang_code)):
                transcription = transcribe_audio(audio_bytes)
                if transcription:
                    st.subheader(translate_text("📢 You Said:", target_lang_code))
                    st.markdown(f"👉 {transcription}")
                    st.session_state.temp_query = transcription  # Store temporarily
                    user_query = transcription

        if submit_button and user_query:
            try:
                detected_lang = translator.detect(user_query).lang
                english_query = translate_text(user_query, 'en') if detected_lang != 'en' else user_query
            except:
                english_query = user_query

            if not is_finance_related(english_query):
                st.warning(translate_text("❌ The question is out of scope. Please ask about finance-related topics.", target_lang_code))
            else:
                companies, finance_terms = extract_financial_entities(user_query, target_lang_code)
                if companies or finance_terms:
                    st.subheader(translate_text("📌 Detected Entities", target_lang_code))
                    if companies:
                        st.write(translate_text(f"Companies: {', '.join(companies)}", target_lang_code))
                    if finance_terms:
                        st.write(translate_text(f"Finance terms: {', '.join(finance_terms)}", target_lang_code))

                st.subheader(translate_text(f"📰 Financial News for: {user_query}", target_lang_code))
                articles = fetch_financial_news(english_query, target_lang_code)

                if isinstance(articles, str):
                    st.warning(articles)
                elif articles:
                    response_text = ""
                    for article in articles:
                        with st.expander(f"{article['title']} ({article['source']})"):
                            st.write(f"{translate_text('Sentiment:', target_lang_code)} {article.get('sentiment', '⚪ Neutral')}")
                            st.write(f"{translate_text('Published:', target_lang_code)} {article['publishedAt']}")
                            st.write(f"{translate_text('Description:', target_lang_code)} {article['description']}")
                            summary = summarize_news(
                                article['title'],
                                article['description'],
                                target_lang_code
                            )
                            st.write(f"{translate_text('Summary:', target_lang_code)} {summary}")
                            summary_audio = text_to_speech(summary, target_lang_code)
                            if summary_audio:
                                st.audio(summary_audio, format="audio/mp3")
                                os.unlink(summary_audio)
                            st.markdown(f"[{translate_text('🔗 Read full article', target_lang_code)}]({article['url']})")
                        response_text += f"{article['title']}: {summary}\n"

                    save_chat_to_mongodb(
                        st.session_state.user_id,
                        user_query,
                        response_text
                    )
                else:
                    st.warning(translate_text("❌ No relevant finance news found.", target_lang_code))
                # The form will automatically clear the input when submitted

def main():
    if not st.session_state.logged_in:
        show_login_page()
    else:
        show_main_app()

if __name__ == "__main__":  # Fixed typo in original code (_name_ to __name__)
    try:
        main()
    finally:
        mongo_client.close()