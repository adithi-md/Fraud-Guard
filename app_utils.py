import streamlit as st
import json
import requests
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from passlib.hash import pbkdf2_sha256

# --- ASSETS & CONFIGURATION ---

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
    "Professional (Blue)": {
        "primary": "#3b82f6",
        "background": "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
        "text": "#333333",
        "card_bg": "rgba(255, 255, 255, 0.7)",
        "card_border": "rgba(255, 255, 255, 0.4)",
        "sidebar_bg": "#ffffff",
        "header_color": "#1e3a8a"
    },
    "Midnight (Dark)": {
        "primary": "#60a5fa",
        "background": "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
        "text": "#e2e8f0",
        "card_bg": "rgba(30, 41, 59, 0.7)",
        "card_border": "rgba(255, 255, 255, 0.1)",
        "sidebar_bg": "#0f172a",
        "header_color": "#f8fafc"
    },
    "Forest (Green)": {
        "primary": "#10b981",
        "background": "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)",
        "text": "#166534",
        "card_bg": "rgba(255, 255, 255, 0.8)",
        "card_border": "rgba(22, 101, 52, 0.2)",
        "sidebar_bg": "#f0fdf4",
        "header_color": "#14532d"
    }
}

def apply_theme(theme_name="Professional (Blue)"):
    """Inject custom CSS based on selected theme"""
    theme = THEMES.get(theme_name, THEMES["Professional (Blue)"])
    
    st.markdown(f"""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        :root {{
            --primary-color: {theme['primary']};
            --text-color: {theme['text']};
            --card-bg: {theme['card_bg']};
            --card-border: {theme['card_border']};
            --header-color: {theme['header_color']};
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: var(--text-color);
        }}

        /* Background */
        .stApp {{
            background: {theme['background']};
        }}

        /* Glassmorphism Card Style */
        .glass-card, .metric-box {{
            background: var(--card-bg);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border-radius: 12px;
            border: 1px solid var(--card-border);
            padding: 20px;
            margin-bottom: 20px;
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
            font-weight: 700;
        }}

        /* Custom Button */
        div.stButton > button {{
            background: var(--primary-color);
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }}
        div.stButton > button:hover {{
            transform: translateY(-2px);
            filter: brightness(110%);
        }}

        /* Metric Cards Tweaks */
        .metric-container {{
            display: flex;
            justify-content: space-between;
            gap: 1rem;
        }}
        .metric-box {{
            text-align: center;
            flex: 1;
            transition: transform 0.2s;
        }}
        .metric-box:hover {{
            transform: translateY(-5px);
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary-color);
        }}
        .metric-label {{
            color: var(--text-color);
            opacity: 0.8;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        /* Breadcrumbs */
        .breadcrumb {{
            font-size: 0.9rem;
            color: var(--text-color);
            margin-bottom: 1rem;
            opacity: 0.8;
        }}
        .breadcrumb span {{
            color: var(--primary-color);
            font-weight: 600;
        }}

    </style>
    """, unsafe_allow_html=True)

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
        st.markdown("<div style='text-align: center; margin-top: 50px;'>", unsafe_allow_html=True)
        st.title("🔒 Secure Access")
        st.markdown("Please verify your credentials to access the Fraud Console.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="e.g., admin")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submit = st.form_submit_button("Login to Dashboard")

                if submit:
                    if self.verify_password(username, password):
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = username
                        st.success("Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
        st.markdown("</div>", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def generate_metrics_html(title, value, delta, is_positive=True):
    delta_color = "green" if is_positive else "red"
    arrow = "↑" if is_positive else "↓"
    return f"""
    <div class="metric-box">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        <div style="color: {delta_color}; font-weight: 600; margin-top: 5px;">
            {arrow} {delta}
        </div>
    </div>
    """

def show_profile_sidebar():
    with st.sidebar:
        st.image("https://ui-avatars.com/api/?name=" + st.session_state.get("username", "User") + "&background=random", width=80)
        st.markdown(f"**Welcome, {st.session_state.get('username', 'User').title()}**")
        st.markdown("Fraud Analyst | Level 3 Access")
        st.markdown("---")
        if st.button("Logout"):
            st.session_state["authenticated"] = False
            st.rerun()

