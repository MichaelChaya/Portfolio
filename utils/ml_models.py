import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

class PricePredictionModel:
    """Price prediction model for AirBnB listings"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        try:
            # Define features
            numeric_features = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'availability_365']
            categorical_features = ['room_type', 'neighbourhood']
            
            # Handle missing values
            imputer_numeric = SimpleImputer(strategy='median')
            imputer_categorical = SimpleImputer(strategy='most_frequent')
            
            df_processed = df.copy()
            
            # Process numeric features
            for feature in numeric_features:
                if feature in df_processed.columns:
                    df_processed[feature] = imputer_numeric.fit_transform(df_processed[[feature]]).flatten()
            
            # Process categorical features
            for feature in categorical_features:
                if feature in df_processed.columns:
                    df_processed[feature] = imputer_categorical.fit_transform(df_processed[[feature]]).flatten()
                    
                    # Label encode
                    if feature not in self.label_encoders:
                        self.label_encoders[feature] = LabelEncoder()
                        df_processed[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(df_processed[feature])
                    else:
                        df_processed[f'{feature}_encoded'] = self.label_encoders[feature].transform(df_processed[feature])
            
            # Select final features
            final_features = []
            for feature in numeric_features:
                if feature in df_processed.columns:
                    final_features.append(feature)
            
            for feature in categorical_features:
                encoded_feature = f'{feature}_encoded'
                if encoded_feature in df_processed.columns:
                    final_features.append(encoded_feature)
            
            self.feature_names = final_features
            return df_processed[final_features]
            
        except Exception as e:
            print(f"Error in feature preparation: {e}")
            return pd.DataFrame()
    
    def train(self, df, target_column='price'):
        """Train the price prediction model"""
        try:
            # Clean target variable
            df = df.copy()
            if target_column in df.columns:
                df[target_column] = pd.to_numeric(df[target_column].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
                df = df.dropna(subset=[target_column])
            
            # Prepare features
            X = self.prepare_features(df)
            y = df[target_column]
            
            if X.empty or y.empty:
                return False, "No valid data for training"
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self.is_trained = True
            return True, f"Model trained successfully. R² = {r2:.3f}, RMSE = {rmse:.2f}"
            
        except Exception as e:
            return False, f"Error training model: {e}"
    
    def predict(self, input_data):
        """Make price prediction"""
        try:
            if not self.is_trained:
                return None, "Model not trained yet"
            
            # Prepare input data
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = input_data.copy()
            
            X = self.prepare_features(input_df)
            
            if X.empty:
                return None, "No valid features for prediction"
            
            # Scale and predict
            X_scaled = self.scaler.transform(X)
            prediction = self.model.predict(X_scaled)
            
            return prediction[0] if len(prediction) == 1 else prediction, "Success"
            
        except Exception as e:
            return None, f"Error making prediction: {e}"
    
    def get_feature_importance(self):
        """Get feature importance from trained model"""
        try:
            if not self.is_trained:
                return pd.DataFrame()
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return pd.DataFrame()

class SentimentAnalysisModel:
    """Sentiment analysis model for text data"""
    
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.vectorizer = None
        self.is_trained = False
    
    def analyze_sentiment_textblob(self, text):
        """Analyze sentiment using TextBlob"""
        try:
            from textblob import TextBlob
            blob = TextBlob(str(text))
            sentiment_data = blob.sentiment
            
            polarity = sentiment_data.polarity  # -1 to 1
            subjectivity = sentiment_data.subjectivity  # 0 to 1
            
            # Convert to categories
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'confidence': abs(polarity)
            }
            
        except Exception as e:
            return {
                'sentiment': 'neutral',
                'polarity': 0,
                'subjectivity': 0,
                'confidence': 0
            }
    
    def batch_analyze_sentiment(self, texts):
        """Analyze sentiment for multiple texts"""
        results = []
        
        for text in texts:
            result = self.analyze_sentiment_textblob(text)
            results.append(result)
        
        return pd.DataFrame(results)

class JobMatchingModel:
    """Job matching model based on skills"""
    
    def __init__(self):
        self.skill_weights = {}
        self.market_data = pd.DataFrame()
    
    def calculate_match_score(self, user_skills, job_requirements):
        """Calculate job match score based on skills overlap"""
        try:
            if not user_skills or not job_requirements:
                return 0
            
            user_skills_lower = set([skill.lower().strip() for skill in user_skills])
            job_requirements_lower = set([skill.lower().strip() for skill in job_requirements])
            
            # Calculate overlap
            matches = len(user_skills_lower.intersection(job_requirements_lower))
            total_required = len(job_requirements_lower)
            
            if total_required == 0:
                return 0
            
            basic_score = (matches / total_required) * 100
            
            # Apply skill weights if available
            if self.skill_weights:
                weighted_score = 0
                total_weight = 0
                
                for skill in job_requirements_lower:
                    weight = self.skill_weights.get(skill, 1.0)
                    if skill in user_skills_lower:
                        weighted_score += weight
                    total_weight += weight
                
                if total_weight > 0:
                    weighted_score = (weighted_score / total_weight) * 100
                    return (basic_score + weighted_score) / 2
            
            return basic_score
            
        except Exception as e:
            print(f"Error calculating match score: {e}")
            return 0
    
    def set_skill_weights(self, weights_dict):
        """Set weights for different skills based on market demand"""
        self.skill_weights = {skill.lower(): weight for skill, weight in weights_dict.items()}
    
    def recommend_skills(self, user_skills, target_roles, market_data):
        """Recommend skills to learn based on target roles and market demand"""
        try:
            recommendations = []
            
            user_skills_lower = set([skill.lower().strip() for skill in user_skills])
            
            # Analyze target roles
            all_required_skills = set()
            for role, requirements in target_roles.items():
                role_skills = set([skill.lower().strip() for skill in requirements])
                all_required_skills.update(role_skills)
            
            # Find missing skills
            missing_skills = all_required_skills - user_skills_lower
            
            # Rank by market demand if data available
            if not market_data.empty and 'Skill' in market_data.columns:
                market_skills = market_data.set_index(market_data['Skill'].str.lower())
                
                for skill in missing_skills:
                    skill_info = {'skill': skill, 'priority': 'medium'}
                    
                    if skill in market_skills.index:
                        skill_data = market_skills.loc[skill]
                        demand = skill_data.get('Demand_2024', 50)
                        growth = skill_data.get('Growth_Rate', 0)
                        salary = skill_data.get('Avg_Salary_EUR', 50000)
                        
                        # Calculate priority score
                        priority_score = (demand * 0.4) + (growth * 0.4) + (salary / 1000 * 0.2)
                        
                        if priority_score > 80:
                            skill_info['priority'] = 'high'
                        elif priority_score > 50:
                            skill_info['priority'] = 'medium'
                        else:
                            skill_info['priority'] = 'low'
                        
                        skill_info.update({
                            'demand': demand,
                            'growth_rate': growth,
                            'avg_salary': salary,
                            'priority_score': priority_score
                        })
                    
                    recommendations.append(skill_info)
                
                # Sort by priority score
                recommendations.sort(key=lambda x: x.get('priority_score', 50), reverse=True)
            
            else:
                recommendations = [{'skill': skill, 'priority': 'unknown'} for skill in missing_skills]
            
            return recommendations[:10]  # Top 10 recommendations
            
        except Exception as e:
            print(f"Error generating skill recommendations: {e}")
            return []

def create_price_prediction_pipeline():
    """Create a complete price prediction pipeline"""
    return PricePredictionModel()

def create_sentiment_analysis_pipeline():
    """Create a complete sentiment analysis pipeline"""
    return SentimentAnalysisModel()

def create_job_matching_pipeline():
    """Create a complete job matching pipeline"""
    return JobMatchingModel()

def evaluate_model_performance(y_true, y_pred, model_type='regression'):
    """Evaluate model performance"""
    try:
        if model_type == 'regression':
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            return {
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2,
                'Mean Absolute Error': np.mean(np.abs(y_true - y_pred))
            }
            
        elif model_type == 'classification':
            accuracy = accuracy_score(y_true, y_pred)
            
            return {
                'Accuracy': accuracy,
                'Classification Report': classification_report(y_true, y_pred)
            }
            
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return {}
