import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn

mps_available = torch.backends.mps.is_available()
if mps_available:
    device = torch.device("mps")


class PClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(1),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_my_model():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model = PClassifierModel()
    model.load_state_dict(torch.load("PCLNetwork.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_my_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

st.set_page_config(page_title="Model Inference", page_icon="🔮")

st.title("Instant Model Inference")
st.write("Upload a single local image file to pass it through your 90% accurate trained model weights.")

uploaded_file = st.file_uploader("Drop target chest radiography scan here...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Chest X-Ray", use_container_width=True)
    
    with st.spinner("🤖 Analysing X-ray contrast gradients..."):
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            
    class_map = {0: "NORMAL (Healthy Lungs)", 1: "PNEUMONIA DETECTED"}
    confidence_score = probabilities[0, prediction].item() * 100
    if prediction == 1:
        st.error(f"Result: {class_map[prediction]} (Confidence: {confidence_score}%)")
    else:
        st.success(f"Result: {class_map[prediction]} (Confidence: {confidence_score:.2f}%)")
