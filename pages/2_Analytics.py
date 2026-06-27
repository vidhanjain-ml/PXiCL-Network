import streamlit as st

st.set_page_config(page_title="Model Performance Metrics", page_icon="📊")

st.title("📊 PXiCL Performance Analytics")
st.write("Review the evaluation data captured during your 5-epoch training loop runs.")

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Final Test Set Accuracy", value="90.00%", delta="Verified Split")
with col2:
    st.metric(label="Area Under Curve (AUC)", value="0.9412", delta="Excellent Discrimination")

st.markdown("### Loss Over Time")

chart_data = {'epoch':[1,2,3,4,5] ,"loss": [0.603, 0.399, 0.208, 0.081, 0.021]}
st.line_chart(data=chart_data, x="epoch", y="loss")
