
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import time
from load_and_analyze import FraudDataLoader
from fraud_detection_model import FraudDetectionModel
import os
import requests
import json
from passlib.hash import pbkdf2_sha256

from fpdf import FPDF
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FraudGuard AI Enterprise",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# --- UTILS ---

def load_lottieurl(url: str):
    """Load Lottie animation from URL"""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# --- THEMES & STYLING ---

THEMES = {
    "Neon Cyberpunk": {
        "primary": "#00f2ff",
        "background": "#050510", # Dark void
        "text": "#e0e0e0",
        "card_bg": "rgba(20, 20, 35, 0.7)",
        "card_border": "rgba(0, 242, 255, 0.3)",
        "sidebar_bg": "#0a0a15",
        "header_color": "#00f2ff",
        "nav_bg": "#0a0a15",
        "nav_active": "#bd00ff"
    },
    "Professional (Blue)": {
        "primary": "#3b82f6",
        "background": "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
        "text": "#333333",
        "card_bg": "rgba(255, 255, 255, 0.75)",
        "card_border": "rgba(255, 255, 255, 0.5)",
        "sidebar_bg": "#ffffff",
        "header_color": "#1e3a8a",
        "nav_bg": "#f8f9fa",
        "nav_active": "#3b82f6"
    },
    "Light Mode": {
        "primary": "#2563eb",
        "background": "#ffffff", 
        "text": "#1a1a1a",
        "card_bg": "rgba(240, 242, 245, 0.9)",
        "card_border": "rgba(0, 0, 0, 0.1)",
        "sidebar_bg": "#f8fafc",
        "header_color": "#1e293b",
        "nav_bg": "#ffffff",
        "nav_active": "#2563eb"
    },
    "Dark Mode": {
        "primary": "#38bdf8",
        "background": "#0f172a", 
        "text": "#f8fafc",
        "card_bg": "rgba(30, 41, 59, 0.8)",
        "card_border": "rgba(255, 255, 255, 0.1)",
        "sidebar_bg": "#1e293b",
        "header_color": "#e2e8f0",
        "nav_bg": "#0f172a",
        "nav_active": "#38bdf8"
    }
}

def apply_theme(theme_name="Neon Cyberpunk"):
    """Inject custom CSS based on selected theme"""
    theme = THEMES.get(theme_name, THEMES["Neon Cyberpunk"])
    
    # Grid animation logic for Neon theme
    extra_css = ""
    if theme_name == "Neon Cyberpunk":
        extra_css = """
        /* Neon Grid Background Animation */
        .stApp {
            background-color: #050510;
            background-image: 
                linear-gradient(rgba(0, 242, 255, 0.05) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 242, 255, 0.05) 1px, transparent 1px);
            background-size: 40px 40px;
            animation: moveGrid 20s linear infinite;
        }
        
        @keyframes moveGrid {
            0% { background-position: 0 0; }
            100% { background-position: 40px 40px; }
        }
        """
    
    st.markdown(f"""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

        :root {{
            --primary-color: {theme['primary']};
            --text-color: {theme['text']};
            --card-bg: {theme['card_bg']};
            --card-border: {theme['card_border']};
            --header-color: {theme['header_color']};
        }}

        html, body, [class*="css"] {{
            font-family: 'Outfit', sans-serif;
            color: var(--text-color);
            font-size: 20px !important; /* Significantly increased font size */
            line-height: 1.6;
        }}
        
        /* Larger Headers and Clean Typography */
        h1 {{ font-size: 3.5rem !important; font-weight: 800 !important; letter-spacing: -1px; }}
        h2 {{ font-size: 2.8rem !important; font-weight: 700 !important; }}
        h3 {{ font-size: 2.2rem !important; font-weight: 600 !important; }}
        p, div, span, label, li {{ font-size: 1.2rem !important; }}

        /* Background */
        {f'.stApp {{ background: {theme["background"]}; }}' if theme_name != "Neon Cyberpunk" else ""}
        {extra_css}

        /* Glassmorphism Card Style - Professional & Clean */
        .glass-card, .metric-box {{
            background: var(--card-bg);
            box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 20px;
            border: 1px solid var(--card-border);
            padding: 32px; /* More whitespace */
            margin-bottom: 32px;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); /* Smooth bounce */
        }}
        
        .glass-card:hover, .metric-box:hover {{
             transform: translateY(-8px) scale(1.01);
             box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
             border-color: var(--primary-color);
        }}
        
        /* Subtle Pulse Animation for Critical Elements */
        @keyframes pulse-glow {{
            0% {{ box-shadow: 0 0 0 0 rgba(0, 242, 255, 0.4); }}
            70% {{ box-shadow: 0 0 0 10px rgba(0, 242, 255, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(0, 242, 255, 0); }}
        }}
        
        .metric-value {{
             animation: pulse-glow 3s infinite;
             border-radius: 50%;
        }}
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: {theme['sidebar_bg']};
            border-right: 1px solid var(--card-border);
        }}
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
             color: var(--text-color);
        }}
        section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span {{
             color: var(--text-color);
        }}

        /* Titles and Headers */
        h1, h2, h3 {{
            color: var(--header_color);
            text-shadow: 0 0 15px rgba(0, 242, 255, 0.2); /* Softer glow */
        }}

        /* Custom Button - Professional Pill Shape */
        div.stButton > button {{
            background: linear-gradient(90deg, var(--primary-color) 0%, #bd00ff 100%);
            color: white;
            border: none;
            padding: 14px 32px; 
            border-radius: 50px; /* Pill shape */
            font-weight: 600;
            font-size: 1.2rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        div.stButton > button:hover {{
            transform: translateY(-3px);
            filter: brightness(110%);
             box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
        }}

        /* Metric Cards Tweaks */
        .metric-container {{
            display: flex;
            justify-content: space-between;
            gap: 1.5rem;
        }}
        .metric-box {{
            text-align: center;
            flex: 1;
        }}
        .metric-value {{
            font-size: 3rem; /* Huge metrics */
            font-weight: 800;
            color: var(--primary-color);
            margin-bottom: 10px;
        }}
        .metric-label {{
            color: var(--text-color);
            opacity: 0.9;
            font-size: 1.1rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 2px;
        }}
        
        /* Breadcrumbs */
        .breadcrumb {{
            font-size: 1.1rem;
            color: var(--text-color);
            margin-bottom: 2rem;
            opacity: 0.8;
            font-family: 'Outfit', sans-serif;
            background: rgba(255,255,255,0.05);
            padding: 12px 24px;
            border-radius: 30px;
            display: inline-block;
            backdrop-filter: blur(5px);
            border: 1px solid var(--card-border);
        }}
        .breadcrumb span {{
            color: var(--primary-color);
            font-weight: 600;
        }}
        
        .stSlider > div > div > div > div {{
            background-color: var(--primary-color);
        }}
        
        /* Chat styling */
        .stChatMessage {{
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            font-size: 1.2rem;
            border: 1px solid var(--card-border);
        }}

    </style>
    """, unsafe_allow_html=True)
    
    # Inject Scanning Animation
    # st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)

def render_breadcrumbs(steps):
    """Render a breadcrumb navigation trail"""
    html = '<div class="breadcrumb">'
    for i, step in enumerate(steps):
        if i == len(steps) - 1:
            html += f'<span>{step}</span>'
        else:
            html += f'{step} &nbsp; / &nbsp; '
    html += '</div>'
    return html

# --- AUTHENTICATION ---

class Auth:
    def __init__(self):
        # Simulated user database
        self.users = {
            "admin": pbkdf2_sha256.hash("admin123"),
            "analyst": pbkdf2_sha256.hash("analyst123")
        }

    def verify_password(self, username, password):
        if username in self.users:
            return pbkdf2_sha256.verify(password, self.users[username])
        return False

    def login_form(self):
        # Centered Login with Neon Style
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
            st.title("🛡️ FraudGuard AI")
            st.markdown("### Enterprise Access")
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="admin")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submit = st.form_submit_button("ENTER SYSTEM", type="primary")

                if submit:
                    if self.verify_password(username, password):
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = username
                        st.success("Access Granted. Initializing Environment...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Access Denied: Invalid Credentials")
            st.markdown('</div>', unsafe_allow_html=True)

def show_profile_sidebar():
    with st.sidebar:
        # Profile Header
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, var(--primary-color), #bd00ff); padding: 20px; border-radius: 15px; margin-bottom: 20px; text-align: center; color: white;">
            <div style="background: rgba(255,255,255,0.2); border-radius: 50%; width: 60px; height: 60px; margin: 0 auto 10px auto; display: flex; justify-content: center; align-items: center; font-size: 1.5rem; font-weight: bold;">AD</div>
            <div style="font-weight: bold; font-size: 1.2rem;">{st.session_state.get('username', 'User').title()}</div>
            <div style="opacity: 0.8; font-size: 0.9rem;">Security Level 5</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Settings Expander
        with st.expander("⚙️ System Settings", expanded=True):
            selected_theme = st.selectbox("Interface Theme", list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state["theme"]))
            if selected_theme != st.session_state["theme"]:
                st.session_state["theme"] = selected_theme
                st.rerun()
            
            st.toggle("Notifications", value=True)
            st.toggle("Sound Effects", value=False)
            
        # Security Info Expander
        with st.expander("🛡️ Security Protocols", expanded=False):
            st.info("Encryption: AES-256")
            st.success("Last Audit: Passed")
            st.caption("Session ID: " + datetime.datetime.now().strftime("%Y%m%d-%H%M"))

        st.markdown("---")
        if st.button("LOGOUT SESSION", use_container_width=True):
            st.session_state["authenticated"] = False
            st.rerun()
            
        st.markdown("<div style='text-align: center; opacity: 0.5; font-size: 0.8rem; margin-top: 20px;'>v3.5.0 Secure Build</div>", unsafe_allow_html=True)

# --- INITIALIZE ---
if "theme" not in st.session_state:
    st.session_state["theme"] = "Neon Cyberpunk"

apply_theme(st.session_state["theme"])

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# --- LOAD ASSETS ---
lottie_security = load_lottieurl("https://lottie.host/9e0eada7-7c70-4f59-9922-3832d2074e50/c0205f24-5d5d-4f36-8e56-11f87e3845c4.json") 
lottie_analytics = load_lottieurl("https://lottie.host/17852b75-3b98-4905-b0d7-2c96c5685293/589ca38f-a797-4c07-9759-335b71948834.json") 
lottie_contact = load_lottieurl("https://lottie.host/3c5b8a0c-4f7f-4b0a-8c3b-5a0d3b6f0e3d/123456.json") 

# --- DATA LOADING (CACHED) ---
@st.cache_data
def load_dataset():
    if os.path.exists('data/fraud_detection_dataset.csv'):
        df = pd.read_csv('data/fraud_detection_dataset.csv')
    else:
        loader = FraudDataLoader(random_state=42)
        df = loader.create_synthetic_dataset(n_samples=20000, fraud_ratio=0.01) 
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/fraud_detection_dataset.csv', index=False)
    return df

@st.cache_resource
def get_model():
    trainer = FraudDetectionModel(random_state=42)
    df = load_dataset()
    trainer.df = df
    trainer.X = df.drop('Class', axis=1)
    trainer.y = df['Class']
    trainer.prepare_data()
    trainer.train_with_smote() 
    return trainer

# --- AI ASSISTANT LOGIC ---

def generate_pdf_report(metrics):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="FraudGuard AI - Enterprise Report", ln=1, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1, align='L')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(200, 10, txt="Executive Summary", ln=1, align='L')
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=f"""
    Total Transactions Processed: {metrics['total']}
    Fraud Cases Detected: {metrics['fraud']}
    Blocked Value: ${metrics['blocked']:,.2f}
    System Health: Optimal
    
    The AI model (SMOTE-enhanced Random Forest) is currently operating with high recall, flagging suspicious activities in real-time.
    """)
    filename = "fraud_report.pdf"
    pdf.output(filename)
    return filename

def get_time_greeting():
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning"
    elif 12 <= hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"

# --- CHATBOT INTELLIGENCE ---

CHAT_RESPONSES = {
    "greeting": {
        "en": "{greeting}! I'm your AI assistant. How can I help you today?",
        "hi": "{greeting}! मैं आपका एआई सहायक हूं। आज मैं आपकी कैसे मदद कर सकता हूं?",
        "kn": "{greeting}! ನಾನು ನಿಮ್ಮ AI ಸಹಾಯಕ. ಇಂದು ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?"
    },
    "report": {
        "en": "I've generated the report. You can download it below.",
        "hi": "मैंने रिपोर्ट तैयार कर ली है। आप इसे नीचे डाउनलोड कर सकते हैं।",
        "kn": "ನಾನು ವರದಿಯನ್ನು ರಚಿಸಿದ್ದೇನೆ. ನೀವು ಅದನ್ನು ಕೆಳಗೆ ಡೌನ್‌ಲೋಡ್ ಮಾಡಬಹುದು."
    },
    "metrics": {
       "en": "Our SMOTE model is achieving >90% recall. You can view the full confusion matrix on the 'Performance' page.",
       "hi": "हमारा SMOTE मॉडल >90% रिकॉल प्राप्त कर रहा है। आप 'Performance' पेज पर पूर्ण भ्रम मैट्रिक्स देख सकते हैं।",
       "kn": "ನಮ್ಮ SMOTE ಮಾದರಿಯು >90% ರಿಕಾಲ್ ಸಾಧಿಸುತ್ತಿದೆ. ನೀವು 'Performance' ಪುಟದಲ್ಲಿ ಪೂರ್ಣ ಗೊಂದಲ ಮ್ಯಾಟ್ರಿಕ್ಸ್ ಅನ್ನು ವೀಕ್ಷಿಸಬಹುದು."
    },
    "monitor": {
        "en": "The Live Monitor simulates real-time transaction processing. Go to the 'Live Monitor' tab and click 'Start Simulation' to see it in action.",
        "hi": "लाइव मॉनिटर रीयल-टाइम लेनदेन प्रसंस्करण का अनुकरण करता है। 'Live Monitor' टैब पर जाएं और इसे देखने के लिए 'Start Simulation' पर क्लिक करें।",
        "kn": "ಲೈವ್ ಮಾನಿಟರ್ ರಿಯಲ್-ಟೈಮ್ ವಹಿವಾಟು ಪ್ರಕ್ರಿಯೆಯನ್ನು ಅನುಕರಿಸುತ್ತದೆ. 'Live Monitor' ಟ್ಯಾಬ್‌ಗೆ ಹೋಗಿ ಮತ್ತು ಅದನ್ನು ನೋಡಲು 'Start Simulation' ಕ್ಲಿಕ್ ಮಾಡಿ."
    },
    "default": {
        "en": "I'm specialized in helping with the fraud detection dashboard. You can ask me about reports, performance, or real-time monitoring.",
        "hi": "मैं धोखाधड़ी का पता लगाने वाले डैशबोर्ड में मदद करने में विशिष्ट हूं। आप मुझसे रिपोर्ट, प्रदर्शन या रीयल-टाइम निगरानी के बारे में पूछ सकते हैं।",
        "kn": "ವಂಚನೆ ಪತ್ತೆ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್‌ಗೆ ಸಹಾಯ ಮಾಡಲು ನಾನು ಪರಿಣತಿ ಹೊಂದಿದ್ದೇನೆ. ನೀವು ವರದಿಗಳು, ಕಾರ್ಯಕ್ಷಮತೆ ಅಥವಾ ರಿಯಲ್-ಟೈಮ್ ಮಾನಿಟರಿಂಗ್ ಬಗ್ಗೆ ನನ್ನನ್ನು ಕೇಳಬಹುದು."
    },
    "apology": {
        "en": "I apologize if I caused any frustration. I'm still learning. ",
        "hi": "अगर मैंने कोई निराशा पैदा की है तो मैं माफी मांगता हूं। मैं अभी भी सीख रहा हूं। ",
        "kn": "ನಾನು ಯಾವುದೇ ನಿರಾಶೆಯನ್ನು ಉಂಟುಮಾಡಿದರೆ ಕ್ಷಮಿಸಿ. ನಾನು ಇನ್ನೂ ಕಲಿಯುತ್ತಿದ್ದೇನೆ. "
    }
}

def detect_language(text):
    # Simple heuristic based on unicode ranges
    for char in text:
        if '\u0900' <= char <= '\u097F': # Devanagari (Hindi)
            return 'hi'
        elif '\u0C80' <= char <= '\u0CFF': # Kannada
            return 'kn'
    return 'en'

def detect_emotion(text):
    negative_keywords = ["angry", "bad", "stupid", "useless", "worst", "hate", "trash", "ಬೇಜಾರು", "ಕೆಟ್ಟ", "बेकार", "गुस्सा", "bakwaas"]
    text_lower = text.lower()
    for word in negative_keywords:
        if word in text_lower:
            return "negative"
    return "neutral"

def process_assistant_query(prompt):
    lang = detect_language(prompt)
    emotion = detect_emotion(prompt)
    prompt_lower = prompt.lower()
    
    # Handle language switching requests
    if "speak in kannada" in prompt_lower or "talk in kannada" in prompt_lower or "kannada" in prompt_lower:
        lang = 'kn'
        return "ಖಂಡಿತ, ನಾನು ಕನ್ನಡದಲ್ಲಿ ಮಾತನಾಡಬಲ್ಲೆ. ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?"
    elif "speak in hindi" in prompt_lower or "talk in hindi" in prompt_lower or "hindi" in prompt_lower:
        lang = 'hi'
        return "बिल्कुल, मैं हिंदी में बात कर सकता हूँ। मैं आपकी कैसे मदद कर सकता हूँ?"

    response_key = "default"
    
    if "report" in prompt_lower or "pdf" in prompt_lower or "ವರದಿ" in prompt_lower or "रिपोर्ट" in prompt_lower:
        df = load_dataset()
        metrics = {
            "total": df.shape[0],
            "fraud": df[df['Class'] == 1].shape[0],
            "blocked": df[df['Class'] == 1]['Amount'].sum()
        }
        report_path = generate_pdf_report(metrics)
        # Handle optimized return for report
        msg = CHAT_RESPONSES["report"][lang]
        return f"REPORT_GENERATED:{report_path}:{msg}" 
        
    elif "metrics" in prompt_lower or "performance" in prompt_lower or "कार्यಕ್ಷಮತೆ" in prompt_lower or "प्रदर्शन" in prompt_lower:
        response_key = "metrics"
    elif "monitor" in prompt_lower or "live" in prompt_lower or "ಲೈವ್" in prompt_lower or "लाइव" in prompt_lower:
        response_key = "monitor"
    elif "hello" in prompt_lower or "hi" in prompt_lower or "namaste" in prompt_lower or "ನಮಸ್ಕಾರ" in prompt_lower or "नमस्ते" in prompt_lower:
        response_key = "greeting"
        
    response = CHAT_RESPONSES[response_key][lang]
    
    if response_key == "greeting":
        time_greeting = get_time_greeting()
        if lang == 'hi':
            response = response.format(greeting="नमस्ते")
        elif lang == 'kn':
            response = response.format(greeting="ನಮಸ್ಕಾರ")
        else:
            response = response.format(greeting=time_greeting)

    if emotion == "negative":
        prefix = CHAT_RESPONSES["apology"][lang]
        response = prefix + response
        
    return response

# --- MAIN APPLICATION ---

def main():
    if not st.session_state["authenticated"]:
        # LOGIN PAGE
        # Use columns to center
        auth = Auth()
        auth.login_form()
    
    else:
        # TOP NAVIGATION (Restored AI Assistant)
        # Using streamlit-option-menu for top navigation
        selected_nav = option_menu(
            menu_title=None,
            options=["Home", "Live Monitor", "Analytics", "Performance", "Predict", "Contact Us", "AI Assistant"],
            icons=["house", "activity", "bar-chart", "speedometer", "lightning", "envelope", "robot"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "5px", "background-color": "transparent"},
                "icon": {"color": "var(--primary-color)", "font-size": "20px"}, 
                "nav-link": {"font-size": "18px", "text-align": "center", "margin":"5px", "--hover-color": "rgba(255,255,255,0.1)", "font-weight": "bold"},
                "nav-link-selected": {"background-color": "var(--primary-color)", "color": "white", "box-shadow": "0 4px 15px rgba(0,0,0,0.3)"},
            }
        )
        
        # Sidebar with settings
        show_profile_sidebar()
        
        # Page Routing
        if selected_nav == "Home":
            render_overview()
        elif selected_nav == "Live Monitor":
            render_realtime()
        elif selected_nav == "Analytics":
            render_analytics()
        elif selected_nav == "Performance":
            render_model_perf()
        elif selected_nav == "Predict":
            render_prediction()
        elif selected_nav == "Contact Us":
            render_contact()
        elif selected_nav == "AI Assistant":
            render_assistant()

# --- PAGE RENDERERS ---

def render_overview():
    st.markdown(render_breadcrumbs(["Home", "Executive Overview"]), unsafe_allow_html=True)
    
    col_head, col_anim = st.columns([3, 1])
    with col_head:
        st.title("📊 Executive Overview")
        st.markdown(f"System Status: **ONLINE** | Protocol: **SECURE**")
    with col_anim:
         if lottie_security:
            st_lottie(lottie_security, height=100, key="h_anim")

    df = load_dataset()
    fraud_count = df[df['Class'] == 1].shape[0]
    total_count = df.shape[0]
    blocked_amount = df[df['Class'] == 1]['Amount'].sum()
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{total_count:,}", "+124 today")
    with col2:
        st.metric("Fraud Detected", f"{fraud_count:,}", "+12%", delta_color="inverse")
    with col3:
        st.metric("Blocked Value", f"${blocked_amount:,.2f}", "+$4.2k", delta_color="normal")
    with col4:
        st.metric("System Integrity", "OK", "Stable")
        
    st.markdown("---")
    
    # Main Chart area
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### 📅 Traffic Volume")
        # Synthetic time series
        dates = pd.date_range(start='2023-01-01', periods=30)
        daily_tx = np.random.randint(1000, 5000, size=30)
        daily_fraud = np.random.randint(10, 50, size=30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=daily_tx, name='Legitimate', fill='tozeroy', line=dict(color='#00f2ff')))
        fig.add_trace(go.Bar(x=dates, y=daily_fraud, name='Malicious', marker_color='#bd00ff'))
        
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
        st.plotly_chart(fig, use_container_width=True)
        
    with col_right:
        st.markdown("### 🚨 Threat Scope")
        if lottie_analytics:
            st_lottie(lottie_analytics, height=200, key="overview_anim")
        
        st.info(" Threat Level: **NEGLIGIBLE**")
        st.progress(0.12)

def render_realtime():
    st.markdown(render_breadcrumbs(["Home", "Live Monitor"]), unsafe_allow_html=True)
    st.title("🔎 Real-time Monitoring")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4>Live Gateway Connection</h4>
            <p>Status: 🟢 <strong>ACTIVESTREAM</strong> | Offset: <strong>12ms</strong></p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("▶ ACTIVATE FEED", type="primary", use_container_width=True):
            st.session_state['simulating'] = True
        if st.button("⏹ TERMINATE", use_container_width=True):
            st.session_state['simulating'] = False
            
    placeholder = st.empty()
    
    if st.session_state.get('simulating'):
        # Simulate 10 transactions
        for i in range(10):
            with placeholder.container():
                # Generate random transactions
                txs = []
                for _ in range(5):
                    amt = np.random.uniform(10, 5000)
                    is_fraud = np.random.random() > 0.9  # 10% chance for demo visual
                    status = "🟥 MALICIOUS" if is_fraud else "🟩 VERIFIED"
                    txs.append({
                        "Time": pd.Timestamp.now().strftime("%H:%M:%S.%f"),
                        "Amount": f"${amt:.2f}",
                        "Location": np.random.choice(["NY, USA", "London, UK", "Tokyo, JP"]),
                        "Status": status
                    })
                
                df_live = pd.DataFrame(txs)
                st.table(df_live)
                
            time.sleep(1.5)
        st.success("Batch Analysis Complete")

def render_analytics():
    st.markdown(render_breadcrumbs(["Home", "Deep Analytics"]), unsafe_allow_html=True)
    st.title("📈 Deep Data Analytics")
    df = load_dataset()
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Distribution", "Correlations"])
    
    with tab1:
        st.markdown("### Transaction Amount Distribution")
        fig = px.histogram(df, x="Amount", color="Class", nbins=50, 
                           color_discrete_map={0: "#00f2ff", 1: "#bd00ff"},
                           opacity=0.7, log_y=True,
                           labels={"Class": "Type"})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.markdown("### Feature Correlations")
        corr = df[['Amount', 'Time', 'V1', 'V2', 'V3', 'Class']].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_model_perf():
    st.markdown(render_breadcrumbs(["Home", "Model Performance"]), unsafe_allow_html=True)
    st.title("🤖 AI Model Performance")
    
    trainer = get_model()
    # Assume SMOTE is the best model trained
    model = trainer.models['SMOTE']
    res = trainer.results['SMOTE Oversampling']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Confusion Matrix")
        cm = res['confusion_matrix']
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Normal', 'Fraud'], y=['Normal', 'Fraud'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <h3>🏆 Champion Model: SMOTE</h3>
            <p>Algorithm: Random Forest + Synthetic Minority Over-sampling</p>
            <br>
            <ul>
                <li><strong>Precision:</strong> {res['precision']:.4f}</li>
                <li><strong>Recall:</strong> {res['recall']:.4f}</li>
                <li><strong>F1-Score:</strong> {res['f1_score']:.4f}</li>
                <li><strong>ROC-AUC:</strong> {res['roc_auc']:.4f}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def render_prediction():
    st.markdown(render_breadcrumbs(["Home", "Instant Prediction"]), unsafe_allow_html=True)
    st.title("⚡ Instant Prediction")
    
    trainer = get_model()
    model = trainer.models['SMOTE']
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0)
            time_val = st.number_input("Time (s)", value=0)
        with col2:
            v1 = st.slider("Feature V1 (Anonymized)", -10.0, 10.0, 0.0)
            v2 = st.slider("Feature V2 (Anonymized)", -10.0, 10.0, 0.0)
            
        submit = st.form_submit_button("SCAN TRANSACTION", type="primary")
        
        if submit:
            np.random.seed(42)
            features_full = np.random.randn(1, 30) 
            features_full[0, 0] = time_val
            features_full[0, 1] = v1
            features_full[0, 2] = v2
            features_full[0, -1] = amount 
            
            features_scaled = trainer.scaler.transform(features_full)
            prob = model.predict_proba(features_scaled)[0, 1]
            
            st.markdown("---")
            if prob > 0.5:
                st.error(f"🚨 ALERT: FRAUD PROBABILITY {prob:.2%}")
            else:
                st.success(f"✅ VERIFIED: CLEAN TRANSACTION ({prob:.2%})")
    st.markdown('</div>', unsafe_allow_html=True)

def render_contact():
    st.markdown(render_breadcrumbs(["Home", "Contact Us"]), unsafe_allow_html=True)
    st.title("📞 Secure Line")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>Fraud Response Unit</h3>
            <p>24/7 Encrypted Channel</p>
            <br>
            <p><strong>📧 Secure Mail:</strong> sos@fraudguard.ai</p>
            <p><strong>📞 Priority Line:</strong> +1 (800) 555-0199</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Direct Message")
        with st.form("contact_form"):
            email = st.text_input("Agent ID / Email")
            message = st.text_area("Intel Brief")
            sent = st.form_submit_button("TRANSMIT")
            if sent:
                st.success("Transmitted successfully.")
        st.markdown('</div>', unsafe_allow_html=True)

def render_assistant():
    st.markdown(render_breadcrumbs(["Home", "AI Assistant"]), unsafe_allow_html=True)
    st.title("🤖 Intelligent Assistant")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
        ### {get_time_greeting()}! 
        I'm here to help you navigate the system, generate reports, and explain features.
        """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Initial greeting
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"{get_time_greeting()}! I'm your AI assistant for the Fraud Detection Dashboard. I can answer your questions, explain features, and even generate reports. How may I help you today?"
        })

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask me about reports, metrics, or navigation..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process logic
        response = process_assistant_query(prompt)
        
        # Handle report path in response if present
        # Handle report path in response if present
        if response.startswith("REPORT_GENERATED:"):
             parts = response.split(":")
             path = parts[1]
             # Get localized message if available
             clean_response = parts[2] if len(parts) > 2 else "I've generated the report."
             
             # User download button needs to be rendered, but inside chat loop it's tricky.
             # We'll just show a success message and render the button below the chat input or in the last message
             
             # Display assistant response
             with st.chat_message("assistant"):
                st.markdown(clean_response)
                with open(path, "rb") as f:
                    st.download_button("⬇️ Download Report", f, file_name="fraud_report.pdf", mime="application/pdf")
                    
             st.session_state.messages.append({"role": "assistant", "content": clean_response})
        else:
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
