import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from lime.lime_text import LimeTextExplainer

# Add hero image
st.image('images/hero.jpg', use_column_width=True)

# Load time series data
df_resampled = pd.read_csv('data/reviews_processed.csv')
df_resampled.set_index('date', inplace=True)

# Add app title
st.title('Time Series Analysis Customer Reviews for Sandbar')

# User input for the time frame selection and sentiment analysis
st.subheader('Select a Time Frame')
time_frame = st.slider('Time Frame (Months)',
                       min_value=1,
                       max_value=(len(df_resampled)),
                       step=3)

# Resample data according to user-selected time frame
resampled_data = df_resampled['stars'].rolling(window=time_frame).mean()

# Plot the time series data
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_resampled.index,
    y=df_resampled['stars'],
    mode='lines',
    name='Monthly Average'
))

# Add 3-month moving average
fig.add_trace(go.Scatter(
    x=df_resampled.index,
    y=resampled_data,
    mode='lines',
    name=f'{time_frame}-Month Moving Average'
))

# Add title and labels
fig.update_layout(
    title=f'Average Star Rating Over Time with {time_frame}-Monthly Moving Average',
    xaxis_title='Time',
    yaxis_title='Average Star Rating',
)

# Display plot
st.plotly_chart(fig, use_container_width=True)

### Sentiment Analysis ###

# Load naive bayes model and TF-IDF vectorizer
naiveBayesModel = joblib.load('models/naive_bayes_model.pkl')
vectorizerTFIdf = joblib.load('models/vectorizer.pkl')

# Instantiate VADER
vader = SentimentIntensityAnalyzer()

# Instantiate the LIME text explainer
lime_explainer = LimeTextExplainer(class_names=['Positive', 'Neutral', 'Negative'])

