# Data Science Portfolio

A comprehensive data science portfolio showcasing two production-ready projects with real business impact. This portfolio demonstrates advanced data science skills through interactive web applications built with Streamlit.

- 📊 **Portfolio app online**: https://portfolio-michael-chaya-moghrabi-data-science.streamlit.app/

## 🚀 Quick Start

### Run Locally
```bash
# Clone the repository
git clone https://github.com/MichaelChaya/DataSciencePortfolio.git
cd DataSciencePortfolio

# Install dependencies
pip install -r requirements.txt

# Run the portfolio
streamlit run Portfolio_overview.py
```

Then open your browser to `http://localhost:8501`

## 📊 Projects Overview

### 1. 🏠 AirBnB Price Predictor & Market Analyzer
- **What it does**: Predicts AirBnB listing prices and analyzes market trends
- **Key Features**: 
  - Interactive price prediction model
  - Market analysis with visualizations
  - Geographic mapping of price distribution
  - Revenue optimization insights
- **Tech Stack**: Python, Scikit-learn, Streamlit, Plotly, Pandas
- **Data**: AirBnB listings, neighborhood data, market trends

### 2. 💼 Job Market Intelligence Tool
- **What it does**: Analyzes job market trends and provides career insights
- **Key Features**:
  - Real-time skill demand analysis
  - Salary insights by location and role
  - Personalized job matching
  - Market reports and trends
  - Dynamic data refresh functionality
- **Tech Stack**: Python, Selenium, Plotly, Streamlit, Pandas
- **Data**: Job postings, salary data, market trends

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+ (tested with Python 3.10)
- pip package manager

### Detailed Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/MichaelChaya/DataSciencePortfolio.git
   cd DataSciencePortfolio
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the portfolio**
   ```bash
   streamlit run Portfolio_overview.py
   ```

5. **Access the application**
   - Open your browser to `http://localhost:8501`
   - Navigate through projects using the sidebar

## 📁 Project Structure

```
DataSciencePortfolio/
├── Portfolio_overview.py          # Main portfolio dashboard
├── pages/                         # Individual project pages
│   ├── 1_🏠_AirBnB_Price_Predictor.py
│   └── 3_💼_Job_Market_Intelligence.py
├── utils/                         # Utility functions
│   ├── data_helpers.py
│   ├── ml_models.py
│   └── visualization.py
├── requirements.txt               # Python dependencies
├── pyproject.toml                # Project configuration
├── README.md                     # This file
└── LICENSE                       # MIT License
```

## 🔧 Key Features

### AirBnB Price Predictor
- **Interactive Price Prediction**: Input property details to get instant price estimates
- **Market Analysis**: Comprehensive neighborhood and market trend analysis
- **Geographic Visualization**: Interactive maps showing price distribution across Berlin
- **Revenue Optimization**: Data-driven insights for maximizing host revenue
- **Data Sources**: Real AirBnB data with fallback to representative sample data

### Job Market Intelligence
- **Skill Demand Analysis**: Real-time tracking of in-demand skills and technologies
- **Salary Insights**: Location-based salary analysis and negotiation tools
- **Job Matching**: Personalized job matching based on skills and preferences
- **Market Reports**: Downloadable text reports with market insights
- **Dynamic Updates**: Refresh functionality for real-time data updates
- **Data Sources**: Job postings and market data with representative sample data

## 📊 Data Sources & Methodology

### AirBnB Project
- **Primary Data**: InsideAirbnb.com public dataset for Berlin
- **Geographic Data**: OpenStreetMap for neighborhood boundaries
- **Market Data**: Real estate trends and economic indicators
- **Fallback**: Representative sample data for demonstration purposes
- **Update Frequency**: Data refreshed on each application run

### Job Market Project
- **Primary Data**: Web scraping from job boards (LinkedIn, Indeed, etc.)
- **Salary Data**: Market research and salary benchmarking
- **Skills Data**: Technology trend analysis and demand metrics
- **Fallback**: Representative sample data for demonstration purposes
- **Update Frequency**: Dynamic refresh functionality available

## 🎯 Technical Skills Demonstrated

- **Machine Learning**: Regression models, feature engineering, model evaluation
- **Data Engineering**: Web scraping, API integration, data validation
- **Visualization**: Interactive dashboards, geographic mapping, statistical charts
- **Web Development**: Streamlit applications, responsive design, user interaction
- **Data Quality**: Error handling, data validation, fallback mechanisms
- **Software Engineering**: Modular code, documentation, version control

## 📈 Project Statistics

- **2 Production-Ready Applications**
- **10+ Technologies Used**
- **Comprehensive Error Handling**
- **Interactive User Interfaces**
- **Real-time Data Processing**

## 🚀 Deployment Options

### Local Development
```bash
streamlit run Portfolio_overview.py
```

### Cloud Deployment
The application is ready for deployment on:
- **Streamlit Cloud** (recommended for easy deployment)
- **Heroku**
- **AWS/GCP/Azure**
- **Docker containers**

### Docker Deployment
```bash
# Build the image
docker build -t data-science-portfolio .

# Run the container
docker run -p 8501:8501 data-science-portfolio
```

### Streamlit Cloud Deployment
1. Push your code to GitHub
2. Connect your GitHub repository to Streamlit Cloud
3. Deploy with one click - no additional configuration needed

## 📝 License

This project is licensed under the Apache License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/MichaelChaya/DataSciencePortfolio/issues).

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!

## 📞 Contact

**Michaël Chaya-Moghrabi**
- 📧 Email: michaelchayamoghrabi@gmail.com
- 💼 LinkedIn: [linkedin.com/in/michael-chaya-moghrabi](https://www.linkedin.com/in/michael-chaya-moghrabi/)
- 🐙 GitHub: [github.com/MichaelChaya](https://github.com/MichaelChaya)

---

**Built with ❤️ by Michaël Chaya-Moghrabi**

