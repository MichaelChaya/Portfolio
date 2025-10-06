import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Michaël Chaya-Moghrabi, Data Science Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and force light mode
st.markdown("""
<style>
    /* Force light mode - disable dark mode completely */
    .stApp {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    .stApp > header {
        background-color: #ffffff !important;
    }
    
    .stApp > div > div > div > div {
        background-color: #ffffff !important;
    }
    
    /* Override any dark mode styles */
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
    }
    
    /* Main content styling */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .project-card {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }
    .impact-metric {
        text-align: center;
        padding: 1rem;
        background-color: #e3f2fd;
        border-radius: 8px;
        margin: 0.5rem;
    }
    
    /* Force all text to be dark */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #262730 !important;
    }
    
    /* Force all backgrounds to be light */
    .stSelectbox > div > div, .stTextInput > div > div, .stTextArea > div > div {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">Michaël Chaya-Moghrabi, Data Science Portfolio</h1>', unsafe_allow_html=True)
    st.markdown("### Professional Data Science Projects with Real Business Impact")
    
    # Introduction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Welcome to my data science portfolio! This collection showcases two comprehensive projects 
        that demonstrate the full machine learning pipeline - from data collection and analysis to 
        model deployment and business impact measurement.
        
        Each project addresses real-world business challenges and provides actionable insights that 
        drive decision-making and revenue growth.
        """)
    
    with col2:
        st.info("""
        **Navigate using the sidebar** 👈
        
        Explore each project to see:
        - Interactive dashboards
        - Live predictions
        - Business impact metrics
        - Technical implementation
        """)
    
    # Portfolio Overview Metrics
    st.markdown("## Portfolio Impact Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Projects Completed", "2", "100%")
    with col2:
        st.metric("Data Points Analyzed", "25K+", "↗️")
    with col3:
        st.metric("Revenue Impact", "18-20%", "📈")
    with col4:
        st.metric("Technologies Used", "10+", "🔧")
    
    # Project Cards
    st.markdown("## Featured Projects")
    
    # Project 1: AirBnB Price Predictor
    st.markdown("""
    <div class="project-card">
        <h3>🏠 Berlin AirBnB Price Predictor & Market Analyzer</h3>
        <p><strong>Business Challenge:</strong> Help hosts optimize pricing and identify profitable neighborhoods</p>
        <p><strong>Solution:</strong> ML-powered price prediction with interactive market analysis</p>
        <p><strong>Impact:</strong> Demonstrates ~18% revenue optimization potential using representative market data</p>
        <p><strong>Tech Stack:</strong> Python, Scikit-learn, Streamlit, Folium, Pandas</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Project 2: Job Market Intelligence
    st.markdown("""
    <div class="project-card">
        <h3>💼 Automated Job Market Intelligence Tool</h3>
        <p><strong>Business Challenge:</strong> Navigate competitive job market with data-driven insights</p>
        <p><strong>Solution:</strong> Automated job scraping and skill demand analysis</p>
        <p><strong>Impact:</strong> Demonstrates 15-20% salary optimization potential using representative market data</p>
        <p><strong>Tech Stack:</strong> Python, Selenium, Plotly, Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Sources Section
    st.markdown("## 📊 Data Sources & Methodology")
    
    st.markdown("""
    ### Data Collection Strategy
    Both projects utilize a combination of real-world data sources and representative market data 
    to ensure reliable, up-to-date insights while maintaining robust performance across all environments.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🏠 AirBnB Price Predictor**

        **Primary Data Sources:**
        - **InsideAirbnb.com**: Real public AirBnB listing data for Berlin (No API key required)
          - Updated quarterly with real listing data
          - Multiple fallback dates for reliability
          - Direct CSV access from data.insideairbnb.com

        **Data Features:**
        - Property characteristics (room type, minimum nights, availability)
        - Location data (latitude, longitude, neighborhood districts)
        - Market indicators (pricing, reviews, availability patterns)
        - Host information (review counts, listing metrics)

        **Update Frequency:** Quarterly updates from InsideAirbnb, with automatic fallback to previous datasets
        """)
    
    with col2:
        st.markdown("""
        **💼 Job Market Intelligence**

        **Primary Data Sources:**
        - **Adzuna API**: Real job market data from Germany (Integrated!)
          - Live job postings with skill requirements
          - Real salary data from actual listings
          - Job counts by city and role type
        - **GitHub API**: Supplementary skill popularity metrics (No API key required)
          - Repository activity as demand indicators
          - Active development for growth trends
        - **Eurostat & Destatis**: Official German labor statistics
          - Salary benchmarks by city and experience level
          - Cost of living indices from federal sources

        **Data Features:**
        - Skill demand extracted from 100+ real job descriptions
        - Real salary data aggregated from actual job listings
        - Live job counts for major German cities
        - Experience-level breakdowns with official statistics

        **Update Frequency:** Real-time data from Adzuna API with robust fallback mechanisms
        """)
    
    st.markdown("""
    ### Data Quality & Validation
    - **Real API integration** with InsideAirbnb.com, Adzuna API, and GitHub API
    - **Live job market data** from Adzuna for German data science positions
    - **Official statistics** from Eurostat and German Federal Statistics Office
    - **Automated fallback** to previous datasets when APIs are unavailable
    - **Validation checks** ensure data completeness and accuracy
    - **Error handling** maintains stability across all environments
    """)
    
    # Technical Skills Overview
    st.markdown("## Technical Skills Demonstrated")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Machine Learning**
        - Regression & Classification
        - Feature Engineering
        - Model Evaluation
        - Hyperparameter Tuning
        """)
    
    with col2:
        st.markdown("""
        **Data Engineering**
        - Web Scraping
        - API Integration
        - Data Cleaning
        - ETL Pipelines
        """)
    
    with col3:
        st.markdown("""
        **Visualization & Deployment**
        - Interactive Dashboards
        - Geographic Visualization
        - Real-time Updates
        - Production Deployment
        """)
    
    # Contact Information
    st.markdown("---")
    st.markdown("## Connect With Me")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("📧 **Email:** michaelchayamoghrabi@gmail.com")
    with col2:
        st.markdown("💼 **LinkedIn:** [linkedin.com/in/michael-chaya-moghrabi](https://www.linkedin.com/in/michael-chaya-moghrabi/)")
    with col3:
        st.markdown("🐙 **GitHub:** [github.com/MichaelChaya](https://github.com/MichaelChaya)")

if __name__ == "__main__":
    main()
