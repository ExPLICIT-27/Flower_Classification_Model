import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
from torchvision.models import ResNet50_Weights
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64
import warnings
warnings.filterwarnings("ignore")

# Configure page
st.set_page_config(
    page_title="Flower Classifier AI",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        transform: translateY(0);
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
    }
    
    .confidence-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        text-align: center;
        transform: translateY(0);
        transition: all 0.3s ease;
    }
    
    .confidence-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
    }
    
    .info-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    .upload-section {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 2px dashed rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 1rem 0;
        color: white;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: rgba(255, 255, 255, 0.5);
        background: rgba(255, 255, 255, 0.15);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stProgress > div > div {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        border-radius: 10px;
    }
    
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: white;
    }
    
    .stMetric {
        background: transparent;
    }
    
    .stMetric > div {
        background: transparent;
    }
    
    .flower-emoji {
        font-size: 2rem;
        margin: 0 0.5rem;
    }
    
    .status-success {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .status-error {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .sidebar .info-card h3 {
        color: #ffffff;
        margin-bottom: 1rem;
    }
    
    .sidebar .info-card p {
        color: #f0f0f0;
        margin-bottom: 0.5rem;
    }
    
    .plotly-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Define flower classes - Update these based on your actual training data
FLOWER_CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Flower emojis for visual appeal
FLOWER_EMOJIS = {
    'daisy': '🌼',
    'dandelion': '🌻',
    'roses': '🌹',
    'sunflowers': '🌻',
    'tulips': '🌷'
}

class FlowerClassifier:
    def __init__(self):
        self.model = None
        self.device = None
        self.model_loaded = False
        
    def load_model(self, model_path="version_2_resnet50.pth"):
        """Load the pre-trained model"""
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Create the same model architecture as in training
            self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            
            # Modify the final layer to match training setup
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, len(FLOWER_CLASSES))
            )
            
            # Load the trained weights
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.to(self.device)
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for prediction using the same transforms as training"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply the same test transforms as in training
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        transformed = transform(image=img_array)
        img_tensor = transformed['image'].unsqueeze(0)
        
        return img_tensor
    
    def predict(self, image):
        """Make prediction on the image"""
        if not self.model_loaded:
            return None, None, None
        
        try:
            img_tensor = self.preprocess_image(image)
            img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Get all probabilities for visualization
                all_probs = probabilities.cpu().numpy()[0]
                
            predicted_class = FLOWER_CLASSES[predicted.item()]
            confidence_score = confidence.item()
            
            return predicted_class, confidence_score, all_probs
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            return None, None, None

@st.cache_resource
def get_classifier():
    """Cache the classifier instance"""
    return FlowerClassifier()

def create_confidence_chart(probabilities, predicted_class):
    """Create an interactive confidence chart"""
    colors = ['#ff6b6b' if cls == predicted_class else '#4facfe' for cls in FLOWER_CLASSES]
    
    fig = go.Figure(data=[
        go.Bar(
            x=FLOWER_CLASSES,
            y=probabilities,
            marker_color=colors,
            text=[f'{prob:.1%}' for prob in probabilities],
            textposition='auto',
            textfont=dict(size=14, color='white', family='Poppins'),
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': '🎯 Prediction Confidence Scores',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': 'white', 'family': 'Poppins'}
        },
        xaxis_title="Flower Types",
        yaxis_title="Confidence Score",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Poppins'),
        yaxis=dict(tickformat='.0%', gridcolor='rgba(255,255,255,0.1)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌸 AI Flower Classifier</h1>
        <p>Powered by ResNet50 Deep Learning Model</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize classifier
    classifier = get_classifier()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="info-card">
            <h3>🤖 Model Information</h3>
            <p><strong>Architecture:</strong> ResNet50 Transfer Learning</p>
            <p><strong>Classes:</strong> 5 Flower Types</p>
            <p><strong>Input Size:</strong> 224×224 pixels</p>
            <p><strong>Framework:</strong> PyTorch</p>
            <p><strong>Accuracy:</strong> High precision classification</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>🌺 Supported Flowers</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for flower in FLOWER_CLASSES:
            emoji = FLOWER_EMOJIS.get(flower, '🌸')
            st.markdown(f"{emoji} **{flower.capitalize()}**")
        
        st.markdown("""
        <div class="info-card">
            <h3>💡 Tips for Best Results</h3>
            <p>• Use clear, well-lit images</p>
            <p>• Ensure flower is the main subject</p>
            <p>• Avoid blurry or distant shots</p>
            <p>• JPG, PNG formats supported</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load model
    if not classifier.model_loaded:
        with st.spinner("🔄 Loading AI Model..."):
            success = classifier.load_model()
            if success:
                st.markdown("""
                <div class="status-success">
                    ✅ Model loaded successfully! Ready for predictions.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-error">
                    ❌ Failed to load model. Please check if 'version_2_resnet50.pth' exists.
                </div>
                """, unsafe_allow_html=True)
                st.stop()
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class="upload-section">
            <h3>📸 Upload Your Flower Image</h3>
            <p>Drag and drop or click to browse</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a flower for AI classification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)
            
            # Image info
            st.markdown(f"""
            <div class="metric-container">
                <p><strong>Image Size:</strong> {image.size[0]} × {image.size[1]} pixels</p>
                <p><strong>Format:</strong> {image.format}</p>
                <p><strong>Mode:</strong> {image.mode}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Classify button
            if st.button("🔍 Classify Flower", key="classify_btn"):
                with st.spinner("🧠 AI is analyzing your image..."):
                    # Animated progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        if i < 30:
                            st.empty()
                    
                    # Make prediction
                    predicted_class, confidence, all_probs = classifier.predict(image)
                    
                    if predicted_class is not None:
                        # Store results in session state
                        st.session_state.prediction_made = True
                        st.session_state.predicted_class = predicted_class
                        st.session_state.confidence = confidence
                        st.session_state.all_probs = all_probs
                        st.session_state.upload_time = uploaded_file.name
                        
                        # Clear progress bar
                        progress_bar.empty()
                        
                        st.markdown("""
                        <div class="status-success">
                            🎉 Classification Complete!
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="status-error">
                            ❌ Classification failed. Please try again.
                        </div>
                        """, unsafe_allow_html=True)
    
    with col2:
        # Display results
        if hasattr(st.session_state, 'prediction_made') and st.session_state.prediction_made:
            predicted_class = st.session_state.predicted_class
            confidence = st.session_state.confidence
            all_probs = st.session_state.all_probs
            
            # Main prediction
            flower_emoji = FLOWER_EMOJIS.get(predicted_class, '🌸')
            st.markdown(f"""
            <div class="prediction-card">
                <h2>🎯 Predicted Flower</h2>
                <div style="font-size: 4rem; margin: 1rem 0;">{flower_emoji}</div>
                <h1 style="font-size: 2.5rem; margin: 0;">{predicted_class.upper()}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence score
            confidence_color = "#4CAF50" if confidence > 0.8 else "#FF9800" if confidence > 0.6 else "#F44336"
            st.markdown(f"""
            <div class="confidence-card">
                <h3>🎯 Confidence Score</h3>
                <h1 style="font-size: 3rem; color: {confidence_color};">{confidence:.1%}</h1>
                <p>{"Very High" if confidence > 0.8 else "High" if confidence > 0.6 else "Moderate"} Confidence</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence chart
            st.markdown('<div class="plotly-container">', unsafe_allow_html=True)
            fig = create_confidence_chart(all_probs, predicted_class)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Detailed results
            st.markdown("### 📊 Detailed Classification Results")
            
            # Sort probabilities for better display
            sorted_indices = np.argsort(all_probs)[::-1]
            
            for i, idx in enumerate(sorted_indices):
                flower = FLOWER_CLASSES[idx]
                prob = all_probs[idx]
                emoji = FLOWER_EMOJIS.get(flower, '🌸')
                
                is_prediction = (flower == predicted_class)
                
                st.markdown(f"""
                <div class="metric-container">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.2rem;">{emoji} <strong>{flower.capitalize()}</strong></span>
                        <span style="font-size: 1.5rem; color: {'#4CAF50' if is_prediction else '#ffffff'};">
                            {prob:.1%} {'✨' if is_prediction else ''}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
        else:
            st.markdown("""
            <div class="info-card">
                <h3>🎯 Ready for Classification!</h3>
                <p style="font-size: 1.1rem; line-height: 1.6;">
                    Upload a flower image on the left and click 'Classify Flower' to see:
                </p>
                <ul style="text-align: left; padding-left: 2rem;">
                    <li>🌸 Flower type identification</li>
                    <li>🎯 Confidence percentage</li>
                    <li>📊 Detailed probability scores</li>
                    <li>📈 Interactive visualization</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.8); margin-top: 2rem;">
        <p style="font-size: 1.1rem;">
            Author : Nandu Mahesh, do star the repo if you liked it :)
        </p>
        <p style="font-size: 0.9rem; opacity: 0.7;">
            Streamlit UI powered by Claude :)
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()