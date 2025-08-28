import streamlit as st

def display_training_section():
    st.markdown('<div class="glass-card"><h3>🧠 Model Training</h3>', unsafe_allow_html=True)
    # Placeholder for training progress/metrics
    st.markdown('<div class="metric-glass">Training loss/accuracy chart will appear here.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
