import streamlit as st

st.set_page_config(page_title="Lungs Health Dashboard", page_icon="🩻", layout="centered")

st.title("PXiCL Network")
st.subheader("PXiCl infrastructure Dashboard")

st.markdown("""
Detect pneumonia early with this model,built from scratch.

### Pipeline Technical Specifications:
* **Input Target Shape:** 224x224 Pixels (3-Channel RGB Normalization)
* **Optimization Framework:** Adam Optimizer ($lr = 0.00005$) with Gradient Clipping ($max\\_norm = 1.0$)
* **Stability Layers:** 2D Batch Normalization to eliminate exploding gradients.

**Use the sidebar navigation** to upload an X-ray or analyze model metrics.









            
            
                            made by vidhan jain with love
""")

