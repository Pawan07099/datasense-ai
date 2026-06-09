import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="DataSense AI",
    page_icon="🧠",
    layout="wide"
)

# Title
st.title("🧠 DataSense AI — Intelligent Data Analysis Agent")
st.subheader("Upload any CSV and get instant AI-powered insights!")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.success("✅ File uploaded successfully!")
    
    # Show basic info
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    
    # Show data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10))
    
    # Show summary statistics
    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe())
    
    # Auto visualization
    st.subheader("📈 Auto Visualizations")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Correlation Heatmap**")
            fig, ax = plt.subplots()
            sns.heatmap(df[numeric_cols].corr(), 
                       annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.write("**Distribution Plot**")
            fig, ax = plt.subplots()
            df[numeric_cols[0]].hist(ax=ax, bins=20, color='steelblue')
            ax.set_title(f'Distribution of {numeric_cols[0]}')
            st.pyplot(fig)
    
    # AI Insights
    st.subheader("🤖 AI Insights")
    st.info(f"""
    📌 **Dataset Overview:**
    - Your dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns**
    - **{len(numeric_cols)} numeric columns** found: {', '.join(numeric_cols[:5])}
    - **Missing values:** {df.isnull().sum().sum()} total
    
    📌 **Key Observations:**
    - Highest value column: **{df[numeric_cols].max().idxmax() if numeric_cols else 'N/A'}**
    - Most correlated columns detected in heatmap above
    
    📌 **Recommendation:**
    - Clean missing values before analysis
    - Check correlation heatmap for feature relationships
    """)

else:
    st.info("👆 Please upload a CSV file to get started!")