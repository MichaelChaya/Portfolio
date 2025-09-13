import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import folium
from datetime import datetime, timedelta

def create_price_distribution_chart(df, price_column='price', title="Price Distribution"):
    """Create price distribution histogram"""
    try:
        # Clean price data
        if df[price_column].dtype == 'object':
            prices = pd.to_numeric(df[price_column].str.replace('$', '').str.replace('€', '').str.replace(',', ''), errors='coerce')
        else:
            prices = df[price_column].copy()
        
        if hasattr(prices, 'dropna'):
            prices = prices.dropna()
        else:
            prices = pd.Series(prices).dropna()
        
        fig = px.histogram(
            x=prices,
            nbins=50,
            title=title,
            labels={'x': 'Price', 'y': 'Count'},
            color_discrete_sequence=['#1f77b4']
        )
        
        fig.update_layout(
            xaxis_title="Price (€)",
            yaxis_title="Number of Listings",
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating price distribution chart: {e}")
        return go.Figure()

def create_correlation_heatmap(df, title="Correlation Heatmap"):
    """Create correlation heatmap for numeric columns"""
    try:
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return go.Figure()
        
        correlation_matrix = numeric_df.corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title=title,
            color_continuous_scale='RdBu_r'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating correlation heatmap: {e}")
        return go.Figure()

def create_time_series_chart(df, date_column, value_column, title="Time Series"):
    """Create time series line chart"""
    try:
        df_clean = df.copy()
        df_clean[date_column] = pd.to_datetime(df_clean[date_column])
        df_clean = df_clean.sort_values(date_column)
        
        fig = px.line(
            df_clean,
            x=date_column,
            y=value_column,
            title=title,
            markers=True
        )
        
        fig.update_layout(
            xaxis_title=date_column.replace('_', ' ').title(),
            yaxis_title=value_column.replace('_', ' ').title()
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating time series chart: {e}")
        return go.Figure()

def create_scatter_plot(df, x_column, y_column, color_column=None, size_column=None, title="Scatter Plot"):
    """Create interactive scatter plot"""
    try:
        fig = px.scatter(
            df,
            x=x_column,
            y=y_column,
            color=color_column,
            size=size_column,
            title=title,
            hover_data=df.columns.tolist()
        )
        
        fig.update_layout(
            xaxis_title=x_column.replace('_', ' ').title(),
            yaxis_title=y_column.replace('_', ' ').title()
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
        return go.Figure()

def create_bar_chart(df, x_column, y_column, color_column=None, title="Bar Chart"):
    """Create bar chart"""
    try:
        fig = px.bar(
            df,
            x=x_column,
            y=y_column,
            color=color_column,
            title=title
        )
        
        fig.update_layout(
            xaxis_title=x_column.replace('_', ' ').title(),
            yaxis_title=y_column.replace('_', ' ').title(),
            xaxis_tickangle=-45
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        return go.Figure()

def create_geographic_map(df, lat_column='latitude', lon_column='longitude', 
                         popup_columns=None, color_column=None, center_coords=None):
    """Create interactive geographic map with markers"""
    try:
        # Set default center coordinates (Berlin)
        if center_coords is None:
            center_coords = [52.5200, 13.4050]
        
        # Create base map
        m = folium.Map(
            location=center_coords,
            zoom_start=11,
            tiles='OpenStreetMap'
        )
        
        # Add markers
        for idx, row in df.iterrows():
            try:
                lat = float(row[lat_column])
                lon = float(row[lon_column])
                
                # Validate coordinates
                if pd.isna(lat) or pd.isna(lon) or lat < -90 or lat > 90 or lon < -180 or lon > 180:
                    continue
                
                # Create popup text
                if popup_columns:
                    popup_text = "<br>".join([f"{col}: {row[col]}" for col in popup_columns if col in row])
                else:
                    popup_text = f"Location: {lat:.4f}, {lon:.4f}"
                
                # Determine marker color
                if color_column and color_column in row:
                    if pd.isna(row[color_column]):
                        color = 'gray'
                    else:
                        value = float(row[color_column])
                        if value > df[color_column].median():
                            color = 'red'
                        else:
                            color = 'blue'
                else:
                    color = 'blue'
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    popup=popup_text,
                    color=color,
                    fill=True,
                    fillOpacity=0.6
                ).add_to(m)
                
            except Exception as e:
                continue  # Skip invalid rows
        
        return m
        
    except Exception as e:
        print(f"Error creating geographic map: {e}")
        return None

def create_sentiment_gauge(sentiment_score, title="Sentiment Score"):
    """Create sentiment gauge chart"""
    try:
        # Normalize score to 0-100 range
        if sentiment_score < 0:
            normalized_score = 50 + (sentiment_score * 50)  # -1 to 0 becomes 0 to 50
        else:
            normalized_score = 50 + (sentiment_score * 50)  # 0 to 1 becomes 50 to 100
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = normalized_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "gray"},
                    {'range': [50, 75], 'color': "lightgreen"},
                    {'range': [75, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        return fig
        
    except Exception as e:
        print(f"Error creating sentiment gauge: {e}")
        return go.Figure()

def create_feature_importance_chart(importance_df, title="Feature Importance"):
    """Create feature importance horizontal bar chart"""
    try:
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title=title,
            color='importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Importance Score",
            yaxis_title="Features"
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating feature importance chart: {e}")
        return go.Figure()

def create_skill_demand_chart(skills_df, x_col='Demand_2024', y_col='Growth_Rate', 
                             size_col='Avg_Salary_EUR', title="Skill Demand Analysis"):
    """Create skill demand bubble chart"""
    try:
        fig = px.scatter(
            skills_df,
            x=x_col,
            y=y_col,
            size=size_col,
            hover_name='Skill',
            title=title,
            color=size_col,
            color_continuous_scale='Viridis',
            size_max=60
        )
        
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title()
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating skill demand chart: {e}")
        return go.Figure()

def create_multi_line_chart(df, x_column, y_columns, title="Multi-line Chart"):
    """Create multi-line chart for comparing trends"""
    try:
        fig = go.Figure()
        
        for y_col in y_columns:
            if y_col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df[x_column],
                    y=df[y_col],
                    mode='lines+markers',
                    name=y_col.replace('_', ' ').title()
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_column.replace('_', ' ').title(),
            yaxis_title="Value",
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating multi-line chart: {e}")
        return go.Figure()

def create_comparison_chart(df, categories, values, title="Comparison Chart"):
    """Create comparison bar chart"""
    try:
        fig = px.bar(
            x=categories,
            y=values,
            title=title,
            color=values,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Value",
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating comparison chart: {e}")
        return go.Figure()

def create_donut_chart(df, names_column, values_column, title="Distribution"):
    """Create donut chart for categorical data"""
    try:
        fig = px.pie(
            df,
            names=names_column,
            values=values_column,
            title=title,
            hole=0.4
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig
        
    except Exception as e:
        print(f"Error creating donut chart: {e}")
        return go.Figure()

def create_dashboard_layout(charts_list, rows=2, cols=2):
    """Create dashboard layout with multiple charts"""
    try:
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[chart.get('title', f'Chart {i+1}') for i, chart in enumerate(charts_list)]
        )
        
        for i, chart_config in enumerate(charts_list):
            row = i // cols + 1
            col = i % cols + 1
            
            if row <= rows and col <= cols:
                chart = chart_config['chart']
                
                # Add traces from the chart to subplot
                for trace in chart.data:
                    fig.add_trace(trace, row=row, col=col)
        
        fig.update_layout(height=600 * rows, showlegend=False)
        
        return fig
        
    except Exception as e:
        print(f"Error creating dashboard layout: {e}")
        return go.Figure()

def apply_custom_theme(fig, theme='plotly_white'):
    """Apply custom theme to plotly figure"""
    try:
        fig.update_layout(
            template=theme,
            font=dict(family="Arial, sans-serif", size=12),
            title_font=dict(size=16, family="Arial, sans-serif"),
            colorway=px.colors.qualitative.Set2
        )
        
        return fig
        
    except Exception as e:
        print(f"Error applying custom theme: {e}")
        return fig
