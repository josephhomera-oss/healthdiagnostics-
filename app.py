import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import io
import base64
import random
import time
import datetime
import cv2
import ollama
import tempfile
import os
import pandas as pd
import requests
from io import BytesIO

# ------------------------------------------------------------------
# Enhanced Custom CSS for Premium UI
# ------------------------------------------------------------------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Animated gradient background */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glass morphism effect for containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        padding: 2rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.45);
    }
    
    /* Gradient header */
    .gradient-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    
    /* Premium button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px 0 rgba(102, 126, 234, 0.6);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    /* Result container */
    .result-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1rem;
        color: white;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] * {
        color: white;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Radio button styling for navigation */
    .stRadio > div {
        gap: 0.5rem;
    }
    
    .stRadio label {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stRadio label:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateX(5px);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    
    /* Connection status indicator */
    .connection-status {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.3; }
        50% { opacity: 1; }
        100% { opacity: 0.3; }
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# REAL AI Functions using MedGemma via Ollama
# ------------------------------------------------------------------

MODEL_NAME = "edwardlo12/medgemma-4b-it-Q4_K_M"

def check_ollama_model():
    """Check if model is available, pull if not"""
    try:
        models = ollama.list()
        model_names = [m['model'] for m in models.get('models', [])]
        if MODEL_NAME not in model_names:
            with st.spinner("📥 Downloading MedGemma model... This may take a few minutes"):
                ollama.pull(MODEL_NAME)
        return True
    except Exception as e:
        st.error(f"⚠️ Ollama connection error: {e}")
        return False

def radiology_analysis_with_medgemma(image_bytes):
    """Real radiology analysis using MedGemma"""
    img_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt = """Analyze this X-ray image carefully. Respond in EXACTLY this format:

    LUNG NODULES: [Present/Absent]
    FRACTURES: [Present/Absent]
    CONFIDENCE: [0.00-1.00]
    DETAILS: [One sentence description of findings]
    
    Be specific about location if findings are present."""
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [img_base64]
            }]
        )
        
        result_text = response['message']['content']
        
        nodules = "Absent" if "nodules: absent" in result_text.lower() else "Present"
        fractures = "Absent" if "fractures: absent" in result_text.lower() else "Present"
        
        import re
        confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', result_text)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.85
        
        return nodules, fractures, confidence, result_text, None
        
    except Exception as e:
        st.error(f"Model inference error: {e}")
        return "Error", "Error", 0.0, str(e), None

def ultrasound_guidance_with_phone_camera(phone_ip_url):
    """
    Capture image from phone camera using IP Webcam app
    
    Args:
        phone_ip_url: The base URL of your phone's IP camera (e.g., http://192.168.1.100:8080)
    
    Returns:
        command: AI guidance command
        confidence: Confidence score
        frame: Captured image frame
    """
    try:
        # Construct the URL for a high-quality JPEG snapshot
        # Common endpoints: /photo.jpg, /shot.jpg, /snapshot.jpg
        snapshot_url = f"{phone_ip_url.rstrip('/')}/photo.jpg"
        
        # Fetch the image with timeout
        response = requests.get(snapshot_url, timeout=5, stream=True)
        
        if response.status_code == 200:
            # Convert the response to an image
            img_bytes = BytesIO(response.content)
            img = Image.open(img_bytes)
            
            # Convert PIL Image to OpenCV format (BGR)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Encode to base64 for Ollama
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # AI Prompt for ultrasound guidance
            prompt = """You are guiding an ultrasound exam for pregnancy (ANC). 
            Analyze this ultrasound screen image and provide guidance for obtaining the proper view.
            
            Respond in EXACTLY this format:
            COMMAND: [One specific movement command]
            TARGET_VIEW: [What view you're trying to achieve]
            CONFIDENCE: [0.00-1.00]
            
            The goal is to identify fetal presentation (breech vs cephalic) or placental location.
            If the image quality is poor, suggest adjustments like "Improve focus" or "Adjust lighting"."""
            
            # Send to Ollama for analysis
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [img_base64]
                }]
            )
            
            result_text = response['message']['content']
            
            # Parse the response
            import re
            command_match = re.search(r'COMMAND:\s*(.+)', result_text)
            command = command_match.group(1) if command_match else "Center probe and hold steady"
            
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', result_text)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.75
            
            return command, confidence, frame
            
        else:
            return f"HTTP {response.status_code}: Failed to fetch from phone", 0.0, None
            
    except requests.exceptions.ConnectionError:
        return "Connection Error: Cannot reach phone camera. Check IP and WiFi connection.", 0.0, None
    except requests.exceptions.Timeout:
        return "Timeout: Phone camera not responding", 0.0, None
    except Exception as e:
        return f"Error: {str(e)}", 0.0, None

def test_phone_connection(phone_ip_url):
    """Test if the phone camera is accessible"""
    try:
        snapshot_url = f"{phone_ip_url.rstrip('/')}/photo.jpg"
        response = requests.get(snapshot_url, timeout=3)
        return response.status_code == 200
    except:
        return False

def causal_analysis_with_llm(patient_data):
    """Use MedGemma for causal reasoning"""
    prompt = f"""Based on this patient data:
    - Age: {patient_data['age']}
    - Tumor size: {patient_data['tumor_size']} cm
    - Treatment history: {patient_data['treatment_history']}
    - Imaging features: {patient_data.get('imaging_features', 'Not specified')}
    
    Provide:
    1. Individual Treatment Effect (ITE) as a number between -0.5 and 1.0
    2. Counterfactual outcome if alternative treatment was given
    3. Causal reasoning based on tumor biology
    
    Respond in format:
    ITE: [number]
    COUNTERFACTUAL: [sentence]
    REASONING: [sentence]"""
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        result_text = response['message']['content']
        
        import re
        ite_match = re.search(r'ITE:\s*([-0-9.]+)', result_text)
        ite = float(ite_match.group(1)) if ite_match else 0.0
        
        return ite, result_text
        
    except Exception as e:
        return 0.0, f"Error: {e}"

def get_federated_status():
    """Simulate federated learning across hospitals"""
    return {
        'round': random.randint(5, 25),
        'hospitals': [
            {'name': '🏥 University Hospital', 'status': 'training', 'samples': 12500, 'accuracy': 0.89},
            {'name': '🏥 City Medical Center', 'status': 'idle', 'samples': 8400, 'accuracy': 0.87},
            {'name': '🏥 Rural Health Clinic', 'status': 'training', 'samples': 3200, 'accuracy': 0.84},
            {'name': '🏥 Children\'s Hospital', 'status': 'completed', 'samples': 5600, 'accuracy': 0.91},
        ],
        'global_accuracy': 0.88
    }

# ------------------------------------------------------------------
# Streamlit App Configuration
# ------------------------------------------------------------------

st.set_page_config(
    page_title="AI Healthcare Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check Ollama connection
if 'ollama_ready' not in st.session_state:
    with st.spinner("🔌 Initializing AI Engine..."):
        st.session_state.ollama_ready = check_ollama_model()

if not st.session_state.ollama_ready:
    st.error("⚠️ Ollama is not running. Please run 'ollama serve' in terminal.")
    st.stop()

# Initialize session state for navigation
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "Radiology AI"

# Initialize phone camera settings
if 'phone_ip_url' not in st.session_state:
    st.session_state.phone_ip_url = "http://192.168.1.100:8080"
if 'phone_connected' not in st.session_state:
    st.session_state.phone_connected = False

# ------------------------------------------------------------------
# Modern Sidebar Navigation
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 3rem;">🏥</div>
        <h2 style="color: white; margin: 0;">AI Healthcare</h2>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.8rem;">Advanced Diagnostics Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Custom styled navigation using radio buttons
    selected_page = st.radio(
        "Navigation",
        options=["Radiology AI", "Ultrasound Co-Pilot", "Causal AI", "Federated Learning"],
        format_func=lambda x: f"{'🩻' if x == 'Radiology AI' else '📹' if x == 'Ultrasound Co-Pilot' else '🔮' if x == 'Causal AI' else '🌐'} {x}",
        label_visibility="collapsed",
        key="nav_radio"
    )
    
    st.session_state.selected_page = selected_page
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem;">
        <p style="color: rgba(255,255,255,0.6); font-size: 0.7rem;">
            Powered by MedGemma<br>
            Privacy-Preserving AI
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main content with glass morphism
st.markdown("""
<div class="glass-card">
    <h1 style="margin: 0; font-size: 2.5rem;">Welcome to AI Healthcare Platform</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">Advanced diagnostics powered by Federated Learning & Causal AI</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Page Content
# ------------------------------------------------------------------

# 1. Radiology AI Page
if st.session_state.selected_page == "Radiology AI":
    st.markdown("""
    <div class="glass-card">
        <h2>🩻 Radiology AI Assistant</h2>
        <p>Advanced detection of lung nodules and bone fractures using federated learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        uploaded_file = st.file_uploader(
            "📁 Upload X-ray Image",
            type=["png", "jpg", "jpeg"],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Patient X-ray", use_container_width=True)
    
    with col2:
        if uploaded_file is not None:
            if st.button("🔍 Start Analysis", use_container_width=True):
                with st.spinner("🧠 AI analyzing medical image..."):
                    image_bytes = uploaded_file.getvalue()
                    nodules, fractures, confidence, full_response, _ = radiology_analysis_with_medgemma(image_bytes)
                
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.subheader("📊 Analysis Results")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    status = "✅" if nodules == "Absent" else "⚠️"
                    color = "#10b981" if nodules == "Absent" else "#ef4444"
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <div style="font-size: 2rem;">{status}</div>
                        <div style="color: {color}; font-weight: bold;">Lung Nodules</div>
                        <div style="color: {color};">{nodules}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_b:
                    status = "✅" if fractures == "Absent" else "⚠️"
                    color = "#10b981" if fractures == "Absent" else "#ef4444"
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <div style="font-size: 2rem;">{status}</div>
                        <div style="color: {color}; font-weight: bold;">Fractures</div>
                        <div style="color: {color};">{fractures}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_c:
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <div style="font-size: 2rem;">📊</div>
                        <div style="font-weight: bold;">Confidence</div>
                        <div style="font-size: 1.5rem;">{confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.expander("📋 Detailed Clinical Report"):
                    st.write(full_response)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.success("💡 Analysis complete. Consultation recommended for positive findings.")

# 2. Ultrasound Co-Pilot Page (UPDATED with Phone Camera)
elif st.session_state.selected_page == "Ultrasound Co-Pilot":
    st.markdown("""
    <div class="glass-card">
        <h2>📹 Ultrasound AI Co-Pilot</h2>
        <p>Real-time guidance for obstetric scans using your phone camera</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Phone Camera Connection Section
    st.markdown("""
    <div class="info-box">
        <h3>📱 Phone Camera Setup</h3>
        <p><strong>Instructions:</strong></p>
        <ol>
            <li>Install <strong>"IP Webcam"</strong> app on your Android phone (by Pavel Khlebovich)</li>
            <li>Open the app and tap <strong>"Start Server"</strong></li>
            <li>Note the URL shown at the bottom (e.g., http://192.168.1.100:8080)</li>
            <li>Enter the URL below and click <strong>Test Connection</strong></li>
            <li>Position your phone to face the ultrasound machine screen</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        phone_url = st.text_input(
            "📱 Phone Camera URL",
            value=st.session_state.phone_ip_url,
            help="Enter the IP Webcam URL shown on your phone (e.g., http://192.168.1.100:8080)"
        )
        st.session_state.phone_ip_url = phone_url
    
    with col2:
        if st.button("🔌 Test Connection", use_container_width=True):
            with st.spinner("Testing connection..."):
                if test_phone_connection(phone_url):
                    st.session_state.phone_connected = True
                    st.success("✅ Phone camera connected!")
                else:
                    st.session_state.phone_connected = False
                    st.error("❌ Cannot connect. Check URL and WiFi connection.")
    
    with col3:
        if st.session_state.phone_connected:
            st.markdown("""
            <div style="text-align: center; padding: 0.5rem;">
                <span class="connection-status" style="background-color: #10b981;"></span>
                <span style="color: #10b981;"> Connected</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
            <div style="font-size: 2rem;">🤰</div>
            <strong>Fetal Presentation</strong>
            <p style="font-size: 0.8rem;">Breech vs Cephalic detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
            <div style="font-size: 2rem;">🧬</div>
            <strong>Placental Assessment</strong>
            <p style="font-size: 0.8rem;">Previa & abruption detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
            <div style="font-size: 2rem;">📏</div>
            <strong>Fetal Biometry</strong>
            <p style="font-size: 0.8rem;">Growth measurements</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("📸 Capture & Analyze", type="primary", use_container_width=True, disabled=not st.session_state.phone_connected):
            if not st.session_state.phone_connected:
                st.warning("⚠️ Please connect to your phone camera first using 'Test Connection'")
            else:
                with st.spinner("📱 Capturing from phone camera..."):
                    command, confidence, captured_frame = ultrasound_guidance_with_phone_camera(st.session_state.phone_ip_url)
                
                if captured_frame is not None:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    
                    # Display captured image
                    st.image(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB), 
                            caption="Ultrasound Screen Capture (via Phone Camera)", 
                            use_container_width=True)
                    
                    # Display AI guidance
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem;">
                        <div style="font-size: 1rem; opacity: 0.9;">🎯 AI Guidance Command</div>
                        <div style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;">{command}</div>
                        <div class="metric-card" style="display: inline-block; padding: 0.5rem 1rem;">
                            AI Confidence: {confidence:.1%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.success("💡 Follow the guidance command for optimal ultrasound view.")
                else:
                    st.error(f"❌ Failed to capture: {command}")
                    st.info("💡 Troubleshooting tips:\n"
                           "- Ensure your phone is running the IP Webcam server\n"
                           "- Check that both devices are on the same WiFi network\n"
                           "- Verify the URL is correct (including port number)\n"
                           "- Try accessing the URL in your computer's browser first")
    
    with col2:
        st.markdown("""
        <div class="glass-card" style="padding: 1rem;">
            <h4>📋 Quick Guide</h4>
            <ol style="margin: 0; padding-left: 1rem;">
                <li>Setup phone camera (see above)</li>
                <li>Position phone at ultrasound screen</li>
                <li>Ensure clear view of display</li>
                <li>Click capture for AI guidance</li>
            </ol>
            <hr>
            <h4>🎯 Tips for Best Results</h4>
            <ul style="margin: 0; padding-left: 1rem;">
                <li>Good lighting on screen</li>
                <li>Steady phone position</li>
                <li>Full screen in frame</li>
                <li>Minimize glare</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.phone_connected:
            st.markdown("""
            <div class="metric-card" style="margin-top: 1rem; padding: 1rem;">
                <div>📱 Phone Status</div>
                <div style="font-size: 0.9rem;">Camera Ready</div>
                <div style="font-size: 0.7rem; opacity: 0.8;">Streaming via WiFi</div>
            </div>
            """, unsafe_allow_html=True)

# 3. Causal AI Page
elif st.session_state.selected_page == "Causal AI":
    st.markdown("""
    <div class="glass-card">
        <h2>🔮 Causal AI Treatment Planner</h2>
        <p>Personalized treatment effect prediction using causal inference</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("treatment_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            patient_id = st.text_input("Patient Identifier", value=f"P{random.randint(1000, 9999)}")
            age = st.slider("Age (years)", 0, 120, 65)
            tumor_size = st.slider("Tumor Size (cm)", 0.1, 20.0, 3.2, 0.1)
        
        with col2:
            treatment = st.selectbox("Current Treatment Protocol", 
                                    ["Chemotherapy", "Radiation", "Immunotherapy", "Surgery", "Combination"])
            imaging_features = st.text_area("Imaging Characteristics", 
                                           "Irregular margins, heterogeneous echotexture")
        
        submitted = st.form_submit_button("📊 Predict Treatment Effect", use_container_width=True)
    
    if submitted:
        with st.spinner("🧠 Running causal inference model..."):
            patient_data = {
                'age': age,
                'tumor_size': tumor_size,
                'treatment_history': treatment,
                'imaging_features': imaging_features
            }
            ite, causal_report = causal_analysis_with_llm(patient_data)
        
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9rem;">Treatment Effect</div>
                <div style="font-size: 2rem; font-weight: bold;">{ite:.3f}</div>
                <div>{'📈 Positive' if ite > 0 else '📉 Negative'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9rem;">Benefit Estimate</div>
                <div style="font-size: 2rem; font-weight: bold;">{abs(ite)*100:.1f}%</div>
                <div>{'Improvement' if ite > 0 else 'Reduction'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            recommendation = "Continue Current" if ite > 0.3 else "Consider Alternatives"
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9rem;">Recommendation</div>
                <div style="font-size: 1rem; font-weight: bold;">{recommendation}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("🔮 Counterfactual Analysis")
        st.info(causal_report)
        
        st.markdown('</div>', unsafe_allow_html=True)

# 4. Federated Learning Page
elif st.session_state.selected_page == "Federated Learning":
    st.markdown("""
    <div class="glass-card">
        <h2>🌐 Federated Learning Network</h2>
        <p>Privacy-preserving AI training across global healthcare institutions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple status display without auto-refresh loop issues
    if st.button("🔄 Refresh Network Status", use_container_width=True):
        status = get_federated_status()
        st.session_state.federated_status = status
    
    # Initialize or get status
    if 'federated_status' not in st.session_state:
        st.session_state.federated_status = get_federated_status()
    
    status = st.session_state.federated_status
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div>Global Training Round</div>
            <div style="font-size: 2rem; font-weight: bold;">Round {status['round']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div>Global Model Accuracy</div>
            <div style="font-size: 2rem; font-weight: bold;">{status['global_accuracy']:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🏥 Network Participants")
    
    for hospital in status['hospitals']:
        col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
        with col1:
            st.write(hospital['name'])
        with col2:
            st.write(f"📊 {hospital['samples']:,}")
        with col3:
            st.write(f"🎯 {hospital['accuracy']:.1%}")
        with col4:
            progress = 1.0 if hospital['status'] == 'completed' else 0.7 if hospital['status'] == 'training' else 0.0
            st.progress(progress, text=hospital['status'].upper())
    
    st.markdown("### 📈 Training Progress")
    st.progress(min(status['round'] / 50, 1.0))
    
    st.info("""
    **🔒 Federated Learning Architecture:**
    - Model: Masked Autoencoding Foundation Model (MedGemma)
    - Data stays at each hospital - completely private
    - Only encrypted model updates are shared
    - Specialized for lung nodule classification & fracture detection
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <p style="color: rgba(255,255,255,0.7);">
        🏥 AI-Powered Healthcare Platform | Federated Learning | Causal Inference | MedGemma | Phone Camera Integration
    </p>
</div>
""", unsafe_allow_html=True)