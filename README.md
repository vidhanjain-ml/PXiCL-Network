# 🩻 Pulmonary Diagnostic Automation Pipeline
An optimized Convolutional Neural Network (CNN) pipeline engineered from scratch in PyTorch to classify binary chest radiography scans. Accelerated natively via Apple Silicon hardware cores.

## 📊 Core Performance Metrics
* **Final Test Set Accuracy:** `90.00%` (Verified via independent split testing)
* **Area Under ROC Curve (AUC):** `0.9412` (Exceptional class discrimination)
* **Inference Pipeline Runtime:** Under 2 seconds (Using optimized caching)

## 🛠️ System Architecture Specifications
The classifier executes a streamlined 2D spatial convolution pipeline optimized to track density, contrasts, and fluid opacities:
* **Input Layer Dimensions:** 224x224 Pixels (Forced 3-Channel RGB Normalization)
* **Optimization Head:** Adam Optimizer (lr = 0.00005) with hard Gradient Norm Clipping (\(max\_norm = 1.0\))
* **Numerical Protection:** Integrated `nn.BatchNorm2d` matrices to eliminate exploding gradients.

## 📁 Repository Structure
```text
Pneumonia Classifier/
├── .streamlit/
│   └── config.toml      # Slate Noir Theme configuration parameters
├── network.py           # Core neural network architecture class blueprint
├── main.py              # Training loop execution script
├── app.py               # Multipage Streamlit landing app framework
├── pneumonia_model.pth # 90% Accurate pre-trained model weights (under 50 MB)
├── requirements.txt     # Global Python installation dependencies
└── pages/
    ├── 1_Predict.py     # Mac GPU-accelerated image prediction panel
    └── 2_Analytics.py   # Training history line chart logging dashboards
```

## 🚀 Instant Local Installation Setup
To reproduce or review this dashboard architecture inside a clean local virtual environment, execute the following commands in your Mac terminal:

```bash
# 1. Clone this blueprint repository
git clone https://github.com
cd Pneumonia-Classifier

# 2. Build and boot up your local virtual environment wrapper
python3 -m venv .venv
source .venv/bin/activate

# 3. Batch install all engineering requirements automatically
pip install -r requirements.txt

# 4. Fire up the dashboard server interface
streamlit run app.py
```
