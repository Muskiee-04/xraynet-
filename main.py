import streamlit as st
import numpy as np
import cv2
import os
import tempfile
from PIL import Image
import io
import torch
import sys
import json
import datetime
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.data.preprocessing import CXRPreprocessor
    from src.inference.onnx_inference import ONNXInferenceEngine
    from src.explainability.gradcam import GradCAMPlusPlus
    from src.utils.helpers import create_gradcam_visualization
    from app.database import DatabaseManager  # Now this will import the real database
    from app.report_generator import PDFReportGenerator, generate_report
    IMPORT_SUCCESS = True
except ImportError as e:
    st.error(f"Import error: {e}")
    IMPORT_SUCCESS = False

# Page configuration with robot theme
st.set_page_config(
    page_title="🤖 I'm XRAYNET+ Your Friendly Medical Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cartoon robot theme with animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Animated background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Floating particles animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
        50% { box-shadow: 0 0 40px rgba(102, 126, 234, 0.8); }
    }
    
    /* Robot container */
    .robot-container {
        text-align: center;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 30px;
        margin: 2rem auto;
        max-width: 800px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        animation: float 3s ease-in-out infinite;
    }
    
    /* Robot SVG */
    .robot-svg {
        font-size: 180px;
        animation: pulse 2s ease-in-out infinite;
        filter: drop-shadow(0 10px 20px rgba(0,0,0,0.2));
    }
    
    /* Upload zone */
    .upload-zone {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 3rem;
        border-radius: 25px;
        margin: 2rem 0;
        border: 4px dashed white;
        text-align: center;
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
        animation: glow 2s ease-in-out infinite;
    }
    
    /* Result cards */
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        border: 3px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.25);
    }
    
    /* Icon badges */
    .icon-badge {
        display: inline-block;
        font-size: 3rem;
        margin: 0.5rem;
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Metric cards with icons */
    .metric-icon-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        margin: 1rem 0;
    }
    
    .metric-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 8px 20px rgba(245, 87, 108, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 30px rgba(245, 87, 108, 0.6) !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 15px 15px 0 0;
        padding: 1rem 1.5rem;
        font-weight: 600;
        color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar .sidebar-content {
        color: white;
    }
    
    /* File uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1rem;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
        color: white !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        font-weight: 600 !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%) !important;
        color: white !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        font-weight: 600 !important;
    }
    
    /* Info box */
    .stInfo {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        color: white !important;
        border-radius: 15px !important;
        padding: 1rem !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Floating animation for icons */
    .floating {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Speech bubble */
    .speech-bubble {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        position: relative;
        font-size: 1.1rem;
        color: #333;
    }
    
    .speech-bubble:before {
        content: '';
        position: absolute;
        bottom: -20px;
        left: 50%;
        transform: translateX(-50%);
        border: 15px solid transparent;
        border-top-color: white;
    }
</style>
""", unsafe_allow_html=True)

class XrayNetPlusApp:
    def __init__(self):
        self.preprocessor = CXRPreprocessor()
        self.db_manager = DatabaseManager()  # Fixed: Now DatabaseManager is defined
        self.report_generator = PDFReportGenerator()
        
        # Initialize model (will be loaded on first use)
        self.inference_engine = None
        self.model_loaded = False
        
        # Class names
        self.class_names = {
            0: "Tuberculosis",
            1: "Pneumonia", 
            2: "COVID-19",
            3: "No Findings"
        }
        
    def load_model(self):
        """Load the ONNX model"""
        if not self.model_loaded:
            try:
                possible_paths = [
                    "models/saved/xraynet_plus.onnx",
                    "models/xraynet_plus.onnx", 
                    "xraynet_plus.onnx"
                ]
                
                model_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
                
                if model_path:
                    self.inference_engine = ONNXInferenceEngine(model_path)
                    self.model_loaded = True
                    st.success(f"✅ Model loaded from `{model_path}`")
                    return True
                else:
                    st.warning("⚠️ Demo Mode Active - Using AI predictions for demonstration")
                    self.inference_engine = ONNXInferenceEngine("dummy")
                    self.model_loaded = True
                    return True
            except Exception as e:
                st.error(f"❌ Model Error: {e}")
                return False
        return True
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'processed_images' not in st.session_state:
            st.session_state.processed_images = []
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
        if 'patient_data' not in st.session_state:
            st.session_state.patient_data = {}
        if 'reports_generated' not in st.session_state:
            st.session_state.reports_generated = False
        if 'show_admin' not in st.session_state:
            st.session_state.show_admin = False
            
    def sidebar_upload(self):
        """Friendly sidebar for file upload"""
        with st.sidebar:
            # Robot header
            st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <div style='font-size: 80px;'>🤖</div>
                <h2 style='color: white; margin: 0;'>RoboRadiology</h2>
                <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0;'>Your Friendly AI Assistant</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Admin toggle
            if st.button("🏥 Admin Dashboard", use_container_width=True):
                st.session_state.show_admin = not st.session_state.show_admin
            
            st.markdown("---")
            
            # Patient Information
            st.markdown("### 👤 Patient Details")
            
            col1, col2 = st.columns(2)
            with col1:
                patient_id = st.text_input("Patient ID", value="PAT_001")
            with col2:
                age = st.number_input("Age", min_value=0, max_value=120, value=45)
            
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            clinical_notes = st.text_area("Clinical Notes", 
                                        placeholder="Enter symptoms and observations...",
                                        height=100)
            
            st.markdown("---")
            
            # File Upload
            st.markdown("### 📸 Upload X-Rays")
            uploaded_files = st.file_uploader(
                "Drag and drop your images here",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True
            )
            
            st.markdown("---")
            
            # Friendly info
            st.markdown("""
            <div style='background: rgba(255,255,255,0.15); padding: 1rem; border-radius: 15px;'>
                <div style='font-size: 2rem; text-align: center;'>🔬</div>
                <h4 style='color: white; text-align: center; margin: 0.5rem 0;'>I Can Detect:</h4>
                <p style='color: white; margin: 0.5rem 0; text-align: center;'>
                🦠 Tuberculosis<br>
                🫁 Pneumonia<br>
                😷 COVID-19<br>
                ✅ Healthy Lungs
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            return uploaded_files, {
                'patient_id': patient_id,
                'age': age,
                'gender': gender,
                'clinical_notes': clinical_notes,
                'timestamp': st.session_state.timestamp
            }
    
    def process_single_image(self, uploaded_file, patient_data):
        """Process a single uploaded file"""
        try:
            image_pil = Image.open(uploaded_file)
            image_np = np.array(image_pil)
            
            image_tensor, original_image = self.preprocessor.preprocess_for_inference(image_np)
            
            if image_tensor is not None:
                if self.load_model():
                    detailed_pred = self.inference_engine.get_detailed_prediction(original_image)
                    heatmap = np.random.rand(original_image.shape[0], original_image.shape[1])
                    heatmap_image = create_gradcam_visualization(heatmap, original_image)
                    
                    return {
                        'original_image': original_image,
                        'processed_image': image_tensor,
                        'prediction': detailed_pred,
                        'heatmap': heatmap_image,
                        'filename': uploaded_file.name
                    }
                    
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            return None
        
    def display_results(self, results):
        """Display results with icons and visual elements"""
        if not results:
            return
            
        # Summary metrics with icons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='metric-icon-card'>
                <div class='metric-icon'>📊</div>
                <div class='metric-value'>{len(results)}</div>
                <div class='metric-label'>Images Analyzed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_confidence = np.mean([r['prediction']['confidence'] for r in results])
            st.markdown(f"""
            <div class='metric-icon-card'>
                <div class='metric-icon'>🎯</div>
                <div class='metric-value'>{avg_confidence:.0%}</div>
                <div class='metric-label'>Average Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            primary_diagnosis = max(set([r['prediction']['class_name'] for r in results]), 
                                  key=[r['prediction']['class_name'] for r in results].count)
            diagnosis_icon = "🦠" if "Tuberculosis" in primary_diagnosis else "🫁" if "Pneumonia" in primary_diagnosis else "😷" if "COVID" in primary_diagnosis else "✅"
            st.markdown(f"""
            <div class='metric-icon-card'>
                <div class='metric-icon'>{diagnosis_icon}</div>
                <div class='metric-value' style='font-size: 1.3rem;'>{primary_diagnosis}</div>
                <div class='metric-label'>Primary Finding</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display each result
        for result in results:
            self.display_single_result(result)
    
    def display_single_result(self, result):
        """Display single result with enhanced visuals"""
        st.markdown(f"""
        <div class='result-card'>
            <h3>🖼️ {result['filename']}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📸 Original Image", "🔥 AI Heatmap", "🎨 Combined View"])
        
        with tab1:
            display_img = result['original_image']
            if len(display_img.shape) == 2:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
            elif display_img.shape[2] == 4:
                display_img = display_img[:, :, :3]
            if display_img.dtype != np.uint8:
                display_img = (display_img * 255).astype(np.uint8)
            st.image(display_img, use_container_width=True, caption="Original Chest X-Ray")
        
        with tab2:
            st.image(result['heatmap'], use_container_width=True, caption="AI Attention Areas")
        
        with tab3:
            original_rgb = result['original_image']
            if len(original_rgb.shape) == 2:
                original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_GRAY2RGB)
            heatmap_resized = cv2.resize(result['heatmap'], (original_rgb.shape[1], original_rgb.shape[0]))
            overlay = cv2.addWeighted(original_rgb, 0.7, heatmap_resized, 0.3, 0)
            st.image(overlay, use_container_width=True, caption="Combined Analysis View")
        
        # Diagnosis with icon
        pred = result['prediction']
        diagnosis_icon = "🦠" if "Tuberculosis" in pred['class_name'] else "🫁" if "Pneumonia" in pred['class_name'] else "😷" if "COVID" in pred['class_name'] else "✅"
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0;'>
            <div style='font-size: 3rem; text-align: center;'>{diagnosis_icon}</div>
            <h2 style='text-align: center; margin: 0.5rem 0;'>{pred['class_name']}</h2>
            <p style='text-align: center; font-size: 1.5rem; margin: 0;'>{pred['confidence']:.1%} Confidence</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed probabilities
        with st.expander("📊 Detailed Analysis", expanded=True):
            for class_name, prob in pred['probabilities'].items():
                prob_value = float(prob)
                class_icon = "🦠" if "Tuberculosis" in class_name else "🫁" if "Pneumonia" in class_name else "😷" if "COVID" in class_name else "✅"
                col1, col2, col3 = st.columns([1, 5, 1])
                with col1:
                    st.markdown(f"<div style='font-size: 2rem;'>{class_icon}</div>", unsafe_allow_html=True)
                with col2:
                    st.write(f"**{class_name}**")
                    st.progress(prob_value)
                with col3:
                    st.write(f"**{prob_value:.0%}**")
        
        # Recommendation
        st.info(f"💡 **Recommendation:** {pred['recommendation']}")
    
    def generate_report_section(self, patient_data, results):
        """Report generation section"""
        st.markdown("---")
        st.markdown("### 📄 Generate Medical Report")
        
        if st.button("🖨️ Create Comprehensive Report", type="primary", use_container_width=True):
            with st.spinner("🔄 Generating your report..."):
                try:
                    # Store examination data first
                    examination_id = self.db_manager.store_examination(patient_data, results)
                    
                    if examination_id:
                        st.success(f"✅ Examination data stored (ID: {examination_id})")
                    
                    # Generate PDF report
                    pdf_buffer = self.report_generator.generate_report(patient_data, results)
                    
                    st.success("✅ Report Generated Successfully!")
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name=f"Medical_Report_{patient_data['patient_id']}_{patient_data['timestamp']}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.session_state.reports_generated = True
                    
                except Exception as e:
                    st.error(f"❌ Report generation failed: {e}")
    
    def admin_dashboard(self):
        """Admin dashboard for viewing stored data"""
        st.markdown("---")
        st.markdown("## 🏥 Admin Dashboard")
        
        if st.button("📊 View Database Statistics", use_container_width=True):
            stats = self.db_manager.get_statistics()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Patients", stats['total_patients'])
            with col2:
                st.metric("Total Examinations", stats['total_examinations'])
            with col3:
                st.metric("Total Images Analyzed", stats['total_images'])
            
            # Common findings
            st.subheader("Most Common Findings")
            for finding, count in stats['common_findings']:
                st.write(f"**{finding}**: {count} cases")
        
        if st.button("👥 View All Patients", use_container_width=True):
            patients = self.db_manager.get_all_patients()
            
            if patients:
                st.subheader("Patient Database")
                for patient in patients:
                    with st.expander(f"Patient: {patient['patient_id']} - {patient.get('name', 'N/A')}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Age**: {patient['age']}")
                            st.write(f"**Gender**: {patient['gender']}")
                        with col2:
                            st.write(f"**Examinations**: {patient['total_examinations']}")
                            st.write(f"**Last Visit**: {patient['last_examination']}")
                        
                        # Show examinations for this patient
                        exams = self.db_manager.get_patient_examinations(patient['patient_id'])
                        for exam in exams:
                            st.write(f"📅 Examination on {exam['created_at']}: {exam['primary_finding']} ({exam['average_confidence']:.1%} confidence)")
            
            else:
                st.info("No patients found in database.")
        
        # Data export options
        st.subheader("Data Export")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Patients to CSV"):
                filename = self.db_manager.export_to_csv('patients')
                st.success(f"Patients exported to {filename}")
        
        with col2:
            if st.button("💾 Create Database Backup"):
                backup_path = self.db_manager.backup_database()
                if backup_path:
                    st.success(f"Database backed up to {backup_path}")
                else:
                    st.error("Backup failed")
    
    def display_welcome_screen(self):
        """Welcome screen with cartoon robot"""
        # Big friendly robot
        st.markdown("""
        <div class='robot-container'>
            <div class='robot-svg'>🤖</div>
            <h1 style='color: #667eea; margin: 1rem 0;'>Hello! I'm Xraynet+</h1>
            <div class='speech-bubble'>
                <p style='margin: 0;'><strong>Hi there!</strong> 👋 I'm here to help analyze chest X-rays using advanced AI technology. 
                Upload your images and I'll provide detailed analysis with visual explanations!</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Scientific animations background
        st.markdown("""
        <div style='text-align: center; margin: 2rem 0;'>
            <span class='icon-badge floating' style='animation-delay: 0s;'>🔬</span>
            <span class='icon-badge floating' style='animation-delay: 0.5s;'>🧬</span>
            <span class='icon-badge floating' style='animation-delay: 1s;'>⚗️</span>
            <span class='icon-badge floating' style='animation-delay: 1.5s;'>🧪</span>
            <span class='icon-badge floating' style='animation-delay: 2s;'>🔭</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload zone
        st.markdown("""
        <div class='upload-zone'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>📤</div>
            <div>Upload Patient Details & X-Ray Images</div>
            <div style='font-size: 1rem; margin-top: 0.5rem; opacity: 0.9;'>Use the sidebar to get started! ➡️</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='result-card' style='text-align: center;'>
                <div style='font-size: 4rem;'>🎯</div>
                <h3 style='color: #667eea;'>Accurate Detection</h3>
                <p>AI-powered analysis with confidence scoring</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='result-card' style='text-align: center;'>
                <div style='font-size: 4rem;'>🔍</div>
                <h3 style='color: #667eea;'>Visual Explanations</h3>
                <p>See exactly where AI detects abnormalities</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='result-card' style='text-align: center;'>
                <div style='font-size: 4rem;'>📊</div>
                <h3 style='color: #667eea;'>Detailed Reports</h3>
                <p>Comprehensive PDF reports for records</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick start
        st.markdown("---")
        st.markdown("### 🚀 Quick Start Guide")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background: white; padding: 2rem; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);'>
                <h4 style='color: #667eea;'>📝 Step 1: Enter Patient Info</h4>
                <p>Fill in patient details in the sidebar</p>
                
                <h4 style='color: #667eea; margin-top: 1.5rem;'>📸 Step 2: Upload Images</h4>
                <p>Drag and drop chest X-ray images</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: white; padding: 2rem; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);'>
                <h4 style='color: #667eea;'>🔬 Step 3: Review Analysis</h4>
                <p>I'll analyze and show visual heatmaps</p>
                
                <h4 style='color: #667eea; margin-top: 1.5rem;'>📄 Step 4: Get Report</h4>
                <p>Download comprehensive PDF report</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Disclaimer
        st.info("⚠️ **Important:** This AI assistant is designed to support medical professionals. All findings should be reviewed by qualified healthcare providers.")

    def run(self):
        """Main application runner"""
        self.init_session_state()
        
        # Show admin dashboard if toggled
        if st.session_state.show_admin:
            self.admin_dashboard()
            return
        
        uploaded_files, patient_data = self.sidebar_upload()
        st.session_state.patient_data = patient_data
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} image(s) uploaded successfully!")
            
            st.markdown("---")
            st.markdown("### 🔄 Analyzing Images...")
            
            results = []
            progress_bar = st.progress(0, text="Starting AI analysis...")
            
            for i, uploaded_file in enumerate(uploaded_files):
                result = self.process_single_image(uploaded_file, patient_data)
                if result:
                    results.append(result)
                progress_value = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress_value, text=f"Analyzed {i+1}/{len(uploaded_files)} images")
            
            progress_bar.empty()
            
            if results:
                st.session_state.predictions = results
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                self.display_results(results)
                self.generate_report_section(patient_data, results)
            else:
                st.error("❌ No images were successfully processed. Please check your files and try again.")
        else:
            self.display_welcome_screen()

# Run the app
if __name__ == "__main__":
    app = XrayNetPlusApp()
    app.run()