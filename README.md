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
- **Tech Stack**: Python, Adzuna API, GitHub API, Plotly, Streamlit, Pandas
- **Data**: Real job postings via Adzuna API, salary data, market trends

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+ (tested with Python 3.10)
- pip package manager
- (Optional) Adzuna API credentials for live job market data

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

4. **Configure API Keys (Optional but Recommended)**

   For live job market data from Adzuna API:

   a. Get free API credentials from [Adzuna Developer Portal](https://developer.adzuna.com/)

   b. Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```

   c. Edit `.env` and add your credentials:
   ```
   ADZUNA_APP_ID=your_app_id_here
   ADZUNA_API_KEY=your_api_key_here
   ```

   **Note**: The application works without API keys using curated fallback data.

5. **Run the portfolio**
   ```bash
   streamlit run Portfolio_overview.py
   ```

6. **Access the application**
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
- **Primary Data**: InsideAirbnb.com public dataset for Berlin (No API key required)
  - Real AirBnB listing data updated quarterly
  - Multiple fallback dates for reliability
- **Geographic Data**: Latitude/longitude coordinates and neighborhood clustering
- **Market Data**: Price trends and availability patterns
- **Update Frequency**: Automatic fallback to most recent available dataset

### Job Market Project
- **Primary Data**: Adzuna API for real German job market data (API key required)
  - 100+ real job postings analyzed per session
  - Live salary data from actual listings
  - Real-time job counts by city
- **Supplementary Data**: GitHub API for skill popularity metrics (No API key required)
- **Official Statistics**: Eurostat & Destatis for salary benchmarks and cost of living
- **Fallback**: Curated market research data when APIs unavailable
- **Update Frequency**: Real-time with dynamic refresh functionality

## 🎯 Technical Skills Demonstrated

- **Machine Learning**: Regression models, feature engineering, model evaluation
- **Data Engineering**: RESTful API integration, data validation, ETL pipelines
- **Visualization**: Interactive dashboards, geographic mapping, statistical charts
- **Web Development**: Streamlit applications, responsive design, user interaction
- **Data Quality**: Error handling, data validation, fallback mechanisms
- **Software Engineering**: Modular code, documentation, version control, secure credential management

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
1. Push your code to GitHub (`.env` is automatically excluded via `.gitignore`)
2. Connect your GitHub repository to Streamlit Cloud
3. **Configure Secrets** in Streamlit Cloud dashboard:
   - Go to App Settings → Secrets
   - Add your API credentials:
     ```toml
     ADZUNA_APP_ID = "your_app_id_here"
     ADZUNA_API_KEY = "your_api_key_here"
     ```
4. Deploy - Streamlit will automatically use the secrets as environment variables

**Note**: The app works without API keys using fallback data, but real-time job market data requires Adzuna API credentials.

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

