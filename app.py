import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from groq import Groq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, f1_score, classification_report, mean_squared_error, r2_score, mean_absolute_error

# =========================
# PAGE CONFIG - MUST BE FIRST!
# =========================
st.set_page_config(page_title="DataGuardian AI", layout="wide", initial_sidebar_state="expanded")

# =========================
# API KEY SETUP
# =========================
load_dotenv()

# Get API key - try Streamlit secrets first, then environment variable
GROQ_API_KEY = None
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate API key
if not GROQ_API_KEY:
    st.error("🔑 **GROQ_API_KEY not found!**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 💻 For Local Development")
        st.code('GROQ_API_KEY=gsk_your_key_here', language="bash")
    with col2:
        st.markdown("### ☁️ For Streamlit Cloud")
        st.markdown("Add to Settings → Secrets:")
        st.code('GROQ_API_KEY = "gsk_your_key_here"', language="toml")
    st.info("🔗 Get your API key: https://console.groq.com/keys")
    st.stop()

# Initialize Groq client
try:
    client = Groq(api_key=GROQ_API_KEY)
    MODEL = "llama-3.3-70b-versatile"
except Exception as e:
    st.error(f"❌ Failed to initialize Groq: {e}")
    st.stop()

# =========================
# ENHANCED STYLING - BIGGER TITLE + NO ARROWS
# =========================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    /* Global Font */
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* FORCE WHITE BACKGROUND EVERYWHERE */
    .stApp {
        background-color: #ffffff !important;
    }
    
    .main {
        background-color: #ffffff !important;
    }
    
    .block-container {
        background-color: #ffffff !important;
        padding: 2rem !important;
        max-width: 1400px !important;
    }
    
    /* Main content area */
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
    }
    
    [data-testid="stHeader"] {
        background-color: rgba(255, 255, 255, 0) !important;
    }
    
    /* Remove keyboard shortcuts tooltip */
    [data-testid="stTooltipHoverTarget"],
    .stTooltipIcon,
    [data-testid="stHeaderActionElements"] button[title*="keyboard"] {
        display: none !important;
    }
    
    /* Hide keyboard shortcut indicators */
    [data-testid="stAppViewBlockContainer"]::before,
    .stApp::before {
        content: none !important;
        display: none !important;
    }
    
    /* Remove any tooltip content */
    [role="tooltip"] {
        display: none !important;
    }
    
    /* Remove sidebar */
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Enhanced Metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.25);
    }
    
    [data-testid="stMetric"] label {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #667eea !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Beautiful Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: #f8f9fa;
        padding: 12px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 12px 24px;
        color: #495057;
        font-weight: 600;
        font-size: 14px;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f8f9fa;
        transform: translateY(-2px);
        border-color: #667eea;
        color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.35) !important;
        border-color: #764ba2 !important;
    }
    
    /* Sidebar Enhancement */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] label {
        color: rgba(255,255,255,0.95) !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* Logo container */
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        background: white;
        border-radius: 50%;
        width: 80px;
        height: 80px;
        margin: 0 auto 20px auto;
        padding: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    .logo-container img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }
    
    /* Title Enhancement - MUCH BIGGER SIZE */
    h1, .main h1, [data-testid="stMarkdownContainer"] h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-weight: 800 !important;
        font-size: 4.5rem !important;
        letter-spacing: -1px !important;
        margin-bottom: 0.5rem !important;
        line-height: 1.2 !important;
    }
    
    /* Subtitle */
    h5 {
        color: #667eea !important;
        font-size: 1.3rem !important;
        font-weight: 500 !important;
        margin-top: 0 !important;
    }
    
    /* Subheaders */
    h2 {
        color: #2d3748 !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }
    
    h3 {
        color: #667eea !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
    }
    
    h4 {
        color: #495057 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    /* Regular text */
    p, li, span, div, label {
        color: #2d3748 !important;
        font-size: 15px !important;
    }
    
    /* Enhanced Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 15px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: #f8f9fa !important;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    
    /* DataFrames */
    [data-testid="stDataFrame"] {
        background: white !important;
    }
    
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        background: white !important;
    }
    
    .stDataFrame thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px !important;
        font-size: 14px !important;
    }
    
    /* Dividers */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 25px 0;
    }
    
    /* Success/Error/Info Messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%) !important;
        border-left: 4px solid #059669;
        border-radius: 10px;
        padding: 12px 16px;
        color: #065f46 !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #ffa8a8 0%, #ff6b6b 100%) !important;
        border-left: 4px solid #dc2626;
        border-radius: 10px;
        padding: 12px 16px;
        color: #991b1b !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #a8e6ff 0%, #6bb6ff 100%) !important;
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        padding: 12px 16px;
        color: #1e40af !important;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        background: #ffffff !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #667eea;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: white !important;
    }
    
    .stFileUploader {
        border-radius: 12px;
        border: 2px dashed #e9ecef;
        padding: 25px;
        background: #f8f9fa;
    }
    
    .stFileUploader:hover {
        border-color: #667eea;
        background: #ffffff;
    }
    
    /* Welcome Screen */
    .welcome-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .welcome-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
        border-color: #667eea;
    }
    
    .welcome-card h3 {
        font-size: 1.2rem !important;
        margin-bottom: 8px !important;
    }
    
    .welcome-card p {
        color: #6c757d !important;
        font-size: 14px !important;
    }
    
    /* Info Card */
    .info-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
        margin: 12px 0;
    }
    
    .info-card h4 {
        margin-bottom: 12px !important;
        font-size: 1.1rem !important;
    }
    
    /* Welcome message */
    .welcome-message {
        text-align: center;
        padding: 60px 20px;
        background: white !important;
    }
    
    .welcome-message h2 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        font-weight: 800;
        margin-bottom: 15px !important;
    }
    
    .welcome-message .subtitle {
        font-size: 1.3rem !important;
        color: #667eea !important;
        margin-top: 15px !important;
        font-weight: 600;
    }
    
    .welcome-message .instruction {
        font-size: 1.1rem !important;
        color: #6c757d !important;
        margin-top: 25px !important;
    }
    
    /* AGGRESSIVE FIX FOR EXPANDER ARROW TEXT */
    /* Hide ALL expander markers and pseudo-elements */
    details summary {
        list-style: none !important;
    }
    
    details summary::-webkit-details-marker {
        display: none !important;
    }
    
    details summary::marker {
        display: none !important;
        content: "" !important;
    }
    
    details summary::before {
        content: "" !important;
        display: none !important;
    }
    
    /* Hide streamlit expander arrow elements */
    .streamlit-expanderHeader::before,
    .streamlit-expanderHeader::after {
        content: "" !important;
        display: none !important;
    }
    
    /* Target the expander content area */
    .streamlit-expanderHeader {
        background: #f8f9fa !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        color: #2d3748 !important;
        font-weight: 600 !important;
        border: 1px solid #e9ecef !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: #ffffff !important;
        border-color: #667eea !important;
        color: #667eea !important;
    }
    
    /* Hide any text nodes starting with underscore */
    .streamlit-expanderHeader *[class*="_arrow"],
    .streamlit-expanderHeader *[class*="arrow"],
    [class*="_arrow"] {
        display: none !important;
        visibility: hidden !important;
        font-size: 0 !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
    }
    
    /* Force hide text content that contains arrow */
    .streamlit-expanderHeader span:not(:empty) {
        font-size: 14px !important;
    }
    
    /* Clean expander content */
    .streamlit-expanderContent {
        background: white !important;
        padding: 15px !important;
        border-radius: 0 0 10px 10px !important;
    }
    
    /* Remove keyboard shortcut overlays */
    [data-testid="stDecoration"],
    .element-container [title*="keyboard"],
    [aria-label*="keyboard"] {
        display: none !important;
    }
    
    /* Hide empty spans and weird artifacts */
    span:empty,
    div:empty {
        display: none !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f8f9fa;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Float animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }
    
    .float {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: white !important;
    }
    
    /* Input fields */
    input, textarea, select {
        background: white !important;
        color: #2d3748 !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: #2d3748 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================
# LOGIC & ANALYSIS
# =========================

def analyze_dataset(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=np.number)
    report = {
        "missing": df.isna().sum()[df.isna().sum() > 0].to_dict(),
        "outliers": {},
        "duplicates": int(df.duplicated().sum()),
        "numeric_cols": list(numeric_df.columns),
        "categorical_cols": list(df.select_dtypes(include=['object', 'category']).columns)
    }
    for col in numeric_df.columns:
        q1, q3 = numeric_df[col].quantile(0.25), numeric_df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = ((numeric_df[col] < (q1 - 1.5 * iqr)) | (numeric_df[col] > (q3 + 1.5 * iqr))).sum()
        if outliers > 0: report["outliers"][col] = int(outliers)
    return report

def detect_problem_type(y):
    """Detect if the problem is classification or regression"""
    unique_values = y.nunique()
    if y.dtype in ['object', 'category']:
        return 'classification'
    elif unique_values <= 10:
        return 'classification'
    else:
        return 'regression'

def smart_chat(df, user_input):
    missing_summary = df.isna().sum()
    missing_cols = missing_summary[missing_summary > 0].to_dict()
    duplicates = df.duplicated().sum()
    
    context = f"""
    Dataset Info:
    - Columns: {list(df.columns)}
    - Data types: {df.dtypes.to_dict()}
    - Shape: {df.shape}
    - Numeric columns: {list(df.select_dtypes(include=np.number).columns)}
    - Categorical columns: {list(df.select_dtypes(include=['object', 'category']).columns)}
    - Missing values: {missing_cols if missing_cols else "None"}
    - Duplicate rows: {duplicates}
    """
    
    prompt = f"""
    You are 'DataGuardian AI', an expert data science assistant.
    
    CRITICAL INSTRUCTIONS:
    
    **IMPORTANT**: The dataframe is ALREADY LOADED as 'df'. DO NOT use pd.read_csv() or load files.
    
    **DEFAULT BEHAVIOR**: 
    - Give TEXT-ONLY answers by default
    - DO NOT generate code or plots unless EXPLICITLY asked
    - Only provide visualizations when user specifically requests them (e.g., "show plot", "visualize this", "create a chart")
    
    1. **For Questions WITHOUT code/plot requests**:
       - Provide a clear, concise text answer
       - Use bullet points and numbered lists
       - Be direct and specific
       - At the end, ask: "Would you like me to create visualizations for this?"
       - DO NOT generate any code
    
    2. **For Questions WITH explicit code/plot requests** (e.g., "plot", "visualize", "show chart", "create graph"):
       - First give brief explanation
       - Then provide working Python code
       - Use the EXISTING 'df' variable
       - Set figure size to (6, 3.5) for compact display
       - Use vibrant palettes: 'viridis', 'plasma', 'rocket'
       - Always end with: st.pyplot(plt.gcf()); plt.clf()
    
    3. **Supervised Learning Questions**:
       - Provide step-by-step explanation first
       - Only generate code if explicitly asked to implement/demonstrate
       - Explain concepts clearly with examples
    
    4. **Response Structure**:
       - Keep responses concise and conversational
       - Use emojis sparingly for clarity
       - Avoid lengthy explanations unless requested
       - Always ask if user wants more details/visualizations
    
    User Question: {user_input}
    
    Dataset Context: {context}
    
    Remember: TEXT-ONLY by default. Code/plots only when explicitly requested!
    """
    
    response = client.chat.completions.create(
        model=MODEL, 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1500
    )
    return response.choices[0].message.content

# =========================
# UI LAYOUT
# =========================

st.title("✨ DataGuardian AI")
st.markdown("##### 🛡️ Intelligent Data Analysis & Model Training Platform")

with st.sidebar:
    st.markdown("""
        <div class="logo-container">
            <img src="https://cdn-icons-png.flaticon.com/512/2103/2103633.png" alt="Logo">
        </div>
    """, unsafe_allow_html=True)
    
    st.header("📤 Upload Center")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    st.markdown("**💡 Quick Tips:**")
    st.markdown("- 🤖 Ask about data insights")
    st.markdown("- 📊 Request plots when needed")
    st.markdown("- 🔍 Explore data quality")
    st.markdown("- 🚀 Train ML models")
    
    st.divider()
    if uploaded_file:
        st.success("✅ File Ready for Analysis")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    report = analyze_dataset(df)
    
    tabs = st.tabs(["💎 Data Preview", "🛡️ Quality Audit", "🧠 AI Assistant", "🚀 Model Playground"])

    # TAB 1: DATA PREVIEW
    with tabs[0]:
        st.subheader("📊 Data Overview")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📏 Rows", f"{df.shape[0]:,}")
        c2.metric("📋 Columns", df.shape[1])
        c3.metric("⚠️ Missing", sum(report["missing"].values()) if report["missing"] else 0)
        c4.metric("🔄 Duplicates", report["duplicates"])
        
        st.markdown("---")
        
        st.write("### 🔍 First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True, height=350)
        
        st.markdown("---")
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("""
            <div class='info-card'>
                <h4 style='color: #667eea; margin-bottom: 15px;'>🔢 Numeric Columns</h4>
            """, unsafe_allow_html=True)
            if report["numeric_cols"]:
                for col in report["numeric_cols"][:10]:
                    st.markdown(f"- **{col}** ({df[col].dtype})")
                if len(report["numeric_cols"]) > 10:
                    st.markdown(f"*...and {len(report['numeric_cols']) - 10} more*")
            else:
                st.write("No numeric columns found")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_info2:
            st.markdown("""
            <div class='info-card'>
                <h4 style='color: #764ba2; margin-bottom: 15px;'>📝 Categorical Columns</h4>
            """, unsafe_allow_html=True)
            if report["categorical_cols"]:
                for col in report["categorical_cols"][:10]:
                    unique_count = df[col].nunique()
                    st.markdown(f"- **{col}** ({unique_count} unique)")
                if len(report["categorical_cols"]) > 10:
                    st.markdown(f"*...and {len(report['categorical_cols']) - 10} more*")
            else:
                st.write("No categorical columns found")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.write("### 📈 Statistical Summary")
        if len(report["numeric_cols"]) > 0:
            with st.expander("📊 View Statistics"):
                st.dataframe(df[report["numeric_cols"]].describe(), use_container_width=True)

    # TAB 2: QUALITY AUDIT
    with tabs[1]:
        st.subheader("🔍 Data Quality Overview")
        
        q1, q2, q3 = st.columns(3)
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells * 100)
        
        q1.metric("Data Completeness", f"{completeness:.1f}%")
        q2.metric("Missing Cells", missing_cells)
        q3.metric("Duplicates", report["duplicates"])
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if report["missing"]:
                st.write("#### 📉 Missing Values")
                fig, ax = plt.subplots(figsize=(4.5, 2.5))
                missing_data = list(report["missing"].items())[:5]
                sns.barplot(
                    x=[v for k, v in missing_data], 
                    y=[k for k, v in missing_data], 
                    palette="rocket",
                    ax=ax
                )
                ax.set_xlabel("Count", fontsize=8)
                ax.set_ylabel("", fontsize=8)
                ax.set_title("Top 5 Missing", fontweight='bold', fontsize=9)
                ax.tick_params(labelsize=7)
                plt.tight_layout()
                st.pyplot(fig)
                plt.clf()
            else: 
                st.success("✅ No missing values!")
        
        with col2:
            st.write("#### 🥧 Completeness")
            fig, ax = plt.subplots(figsize=(4.5, 2.5))
            colors = ['#667eea', '#ff6b6b']
            explode = (0.05, 0) if missing_cells > 0 else (0, 0)
            ax.pie(
                [total_cells - missing_cells, missing_cells], 
                labels=['Complete', 'Missing'], 
                autopct='%1.1f%%',
                colors=colors,
                explode=explode,
                startangle=90,
                textprops={'fontsize': 7}
            )
            ax.set_title("Data Quality", fontweight='bold', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()
        
        if len(report["numeric_cols"]) > 1:
            with st.expander("📊 View Correlations & Distributions"):
                col3, col4 = st.columns(2)
                
                with col3:
                    st.write("**Correlation Heatmap**")
                    cols_to_show = report["numeric_cols"][:6]
                    corr = df[cols_to_show].corr()
                    fig, ax = plt.subplots(figsize=(4.5, 3.5))
                    sns.heatmap(
                        corr, 
                        annot=True, 
                        fmt='.2f', 
                        cmap='coolwarm', 
                        center=0,
                        square=True,
                        ax=ax,
                        annot_kws={'fontsize': 6}
                    )
                    ax.set_title("Correlations", fontsize=9)
                    ax.tick_params(labelsize=6)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.clf()
                
                with col4:
                    st.write("**Distribution**")
                    selected_col = st.selectbox("Column:", report["numeric_cols"], key="dist")
                    fig, ax = plt.subplots(figsize=(4.5, 3.5))
                    sns.histplot(df[selected_col].dropna(), kde=True, color='#667eea', bins=20, ax=ax)
                    ax.set_title(selected_col, fontsize=9)
                    ax.tick_params(labelsize=7)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.clf()
        
        if len(report["categorical_cols"]) > 0:
            with st.expander("📂 View Categorical Analysis"):
                cat_col = st.selectbox("Select column:", report["categorical_cols"], key="cat")
                fig, ax = plt.subplots(figsize=(9, 3))
                value_counts = df[cat_col].value_counts().head(8)
                sns.barplot(x=value_counts.values, y=value_counts.index, palette="viridis", ax=ax)
                ax.set_title(f"{cat_col} Distribution", fontsize=9)
                ax.tick_params(labelsize=7)
                plt.tight_layout()
                st.pyplot(fig)
                plt.clf()

    # TAB 3: AI ASSISTANT
    with tabs[2]:
        st.subheader("💬 Chat with your Data")
        st.markdown("*Ask questions naturally - I'll provide visualizations when you need them!*")
        
        if "messages" not in st.session_state: 
            st.session_state.messages = []
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): 
                st.markdown(msg["content"])

        if user_prompt := st.chat_input("💭 Ask anything about your data..."):
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"): 
                st.markdown(user_prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    resp = smart_chat(df, user_prompt)
                
                if "```python" in resp:
                    parts = resp.split("```python")
                    st.markdown(parts[0])
                    
                    for i in range(1, len(parts)):
                        code = parts[i].split("```")[0]
                        st.code(code, language="python")
                        try: 
                            exec(code, {
                                "df": df, 
                                "plt": plt, 
                                "sns": sns, 
                                "st": st, 
                                "np": np,
                                "pd": pd
                            })
                        except Exception as e: 
                            st.error(f"⚠️ Execution Error: {e}")
                        
                        if i < len(parts) - 1:
                            remaining = parts[i].split("```", 1)
                            if len(remaining) > 1:
                                st.markdown(remaining[1])
                else: 
                    st.markdown(resp)
            
            st.session_state.messages.append({"role": "assistant", "content": resp})

    # TAB 4: MODEL PLAYGROUND
    with tabs[3]:
        st.subheader("🚀 ML Model Playground")
        
        col_config1, col_config2, col_config3 = st.columns([2, 2, 1])
        
        with col_config1:
            target_col = st.selectbox("🎯 Select Target Variable", options=df.columns)
        
        with col_config2:
            if target_col:
                y_sample = df[target_col].dropna()
                auto_type = detect_problem_type(y_sample)
                problem_type = st.radio(
                    "📊 Problem Type",
                    options=["Classification", "Regression"],
                    index=0 if auto_type == "classification" else 1,
                    horizontal=True,
                    help=f"Auto-detected: {auto_type.capitalize()}"
                )
        
        with col_config3:
            test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
        
        st.markdown("---")
        
        col_model1, col_model2 = st.columns([3, 1])
        
        with col_model1:
            if problem_type == "Classification":
                st.write("#### 🎯 Classification Models")
                model_choice = st.selectbox(
                    "Select Model",
                    options=[
                        "Random Forest Classifier",
                        "Logistic Regression",
                        "Decision Tree Classifier",
                        "Gradient Boosting Classifier",
                        "Support Vector Machine (SVM)"
                    ]
                )
            else:
                st.write("#### 📈 Regression Models")
                model_choice = st.selectbox(
                    "Select Model",
                    options=[
                        "Random Forest Regressor",
                        "Linear Regression",
                        "Ridge Regression",
                        "Lasso Regression",
                        "Gradient Boosting Regressor",
                        "Support Vector Regressor (SVR)"
                    ]
                )
        
        with col_model2:
            st.write("#### ")
            st.write("")
        
        if st.button("🚀 Train Model", use_container_width=True):
            try:
                with st.spinner("🔄 Training model..."):
                    X = df.drop(columns=[target_col]).select_dtypes(include=np.number).fillna(0)
                    y = df[target_col]
                    
                    if X.shape[1] == 0:
                        st.error("❌ No numeric features available for training!")
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )
                        
                        if problem_type == "Classification":
                            if model_choice == "Random Forest Classifier":
                                model = RandomForestClassifier(n_estimators=100, random_state=42)
                            elif model_choice == "Logistic Regression":
                                model = LogisticRegression(max_iter=1000, random_state=42)
                            elif model_choice == "Decision Tree Classifier":
                                model = DecisionTreeClassifier(random_state=42)
                            elif model_choice == "Gradient Boosting Classifier":
                                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                            else:
                                model = SVC(kernel='rbf', random_state=42)
                            
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            
                            acc = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            
                            st.success(f"✅ {model_choice} trained successfully!")
                            
                            col_m1, col_m2, col_m3 = st.columns(3)
                            col_m1.metric("🎯 Accuracy", f"{acc*100:.2f}%")
                            col_m2.metric("📊 F1 Score", f"{f1:.3f}")
                            col_m3.metric("🧪 Test Samples", len(y_test))
                            
                            st.markdown("---")
                            
                            col_p1, col_p2 = st.columns(2)
                            
                            with col_p1:
                                if hasattr(model, 'feature_importances_'):
                                    st.write("#### 🌟 Feature Importance")
                                    feat_importances = pd.Series(
                                        model.feature_importances_, 
                                        index=X.columns
                                    ).nlargest(10)
                                    
                                    fig, ax = plt.subplots(figsize=(5.5, 3.5))
                                    feat_importances.plot(kind='barh', color="#667eea", ax=ax)
                                    ax.set_xlabel("Importance", fontsize=8)
                                    ax.set_title("Top 10 Features", fontsize=9)
                                    ax.tick_params(labelsize=7)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.clf()
                                else:
                                    st.info("Feature importance not available for this model")
                            
                            with col_p2:
                                st.write("#### 📋 Classification Report")
                                report_dict = classification_report(y_test, y_pred, output_dict=True)
                                report_df = pd.DataFrame(report_dict).transpose()
                                st.dataframe(report_df.style.background_gradient(cmap='Blues'), 
                                           use_container_width=True, height=300)
                        
                        else:
                            if model_choice == "Random Forest Regressor":
                                model = RandomForestRegressor(n_estimators=100, random_state=42)
                            elif model_choice == "Linear Regression":
                                model = LinearRegression()
                            elif model_choice == "Ridge Regression":
                                model = Ridge(alpha=1.0, random_state=42)
                            elif model_choice == "Lasso Regression":
                                model = Lasso(alpha=1.0, random_state=42)
                            elif model_choice == "Gradient Boosting Regressor":
                                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                            else:
                                model = SVR(kernel='rbf')
                            
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            st.success(f"✅ {model_choice} trained successfully!")
                            
                            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                            col_m1.metric("📊 R² Score", f"{r2:.4f}")
                            col_m2.metric("📉 RMSE", f"{rmse:.4f}")
                            col_m3.metric("📏 MAE", f"{mae:.4f}")
                            col_m4.metric("🧪 Test Samples", len(y_test))
                            
                            st.markdown("---")
                            
                            col_p1, col_p2 = st.columns(2)
                            
                            with col_p1:
                                st.write("#### 📈 Actual vs Predicted")
                                fig, ax = plt.subplots(figsize=(5.5, 3.5))
                                ax.scatter(y_test, y_pred, alpha=0.6, color='#667eea', edgecolors='k')
                                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                                       'r--', lw=2, label='Perfect Prediction')
                                ax.set_xlabel("Actual Values", fontsize=8)
                                ax.set_ylabel("Predicted Values", fontsize=8)
                                ax.set_title("Prediction Quality", fontsize=9)
                                ax.legend(fontsize=7)
                                ax.tick_params(labelsize=7)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.clf()
                            
                            with col_p2:
                                if hasattr(model, 'feature_importances_'):
                                    st.write("#### 🌟 Feature Importance")
                                    feat_importances = pd.Series(
                                        model.feature_importances_, 
                                        index=X.columns
                                    ).nlargest(10)
                                    
                                    fig, ax = plt.subplots(figsize=(5.5, 3.5))
                                    feat_importances.plot(kind='barh', color="#667eea", ax=ax)
                                    ax.set_xlabel("Importance", fontsize=8)
                                    ax.set_title("Top 10 Features", fontsize=9)
                                    ax.tick_params(labelsize=7)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.clf()
                                else:
                                    st.write("#### 📊 Residual Plot")
                                    residuals = y_test - y_pred
                                    fig, ax = plt.subplots(figsize=(5.5, 3.5))
                                    ax.scatter(y_pred, residuals, alpha=0.6, color='#764ba2', edgecolors='k')
                                    ax.axhline(y=0, color='r', linestyle='--', lw=2)
                                    ax.set_xlabel("Predicted Values", fontsize=8)
                                    ax.set_ylabel("Residuals", fontsize=8)
                                    ax.set_title("Residual Analysis", fontsize=9)
                                    ax.tick_params(labelsize=7)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.clf()
                        
                        st.markdown("---")
                        with st.expander("ℹ️ Model Information"):
                            st.write(f"**Model Type:** {problem_type}")
                            st.write(f"**Algorithm:** {model_choice}")
                            st.write(f"**Training Samples:** {len(X_train)}")
                            st.write(f"**Test Samples:** {len(X_test)}")
                            st.write(f"**Features Used:** {X.shape[1]}")
                            st.write(f"**Feature Names:** {', '.join(X.columns[:10])}" + 
                                   ("..." if X.shape[1] > 10 else ""))
                
            except Exception as e:
                st.error(f"❌ Model training failed: {e}")
                st.info("💡 Make sure the target column is appropriate for the selected problem type.")

else:
    st.markdown("""
    <div class='welcome-message'>
        <h2>👋 Welcome to DataGuardian AI!</h2>
        <p class='subtitle'>✨ Your intelligent companion for data analysis</p>
        <p class='instruction'>👈 Upload a CSV file to begin</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='welcome-card float'>
            <h3 style='color: #667eea;'>💎 Data Preview</h3>
            <p>Explore with interactive tables</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='welcome-card float' style='animation-delay: 0.2s;'>
            <h3 style='color: #764ba2;'>🛡️ Quality Audit</h3>
            <p>Visualize data health</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='welcome-card float' style='animation-delay: 0.4s;'>
            <h3 style='color: #667eea;'>🧠 AI Assistant</h3>
            <p>Chat naturally with data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='welcome-card float' style='animation-delay: 0.6s;'>
            <h3 style='color: #764ba2;'>🚀 ML Training</h3>
            <p>Train multiple models</p>
        </div>
        """, unsafe_allow_html=True)
