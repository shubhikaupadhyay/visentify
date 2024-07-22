import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import BartTokenizer, BartForConditionalGeneration
import plotly.graph_objects as go
from textblob import TextBlob
import plotly.express as px

st.set_page_config(page_title = "Visentify",
                   page_icon = "🎭", 
                   layout = "wide")


# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('amazon_alexa.tsv', sep='\t')
    return data

# Sentiment Analysis Function
def analyze_sentiment(text):
    blob = TextBlob(text)
    subjectivity = blob.subjectivity
    polarity = blob.polarity
    if polarity < 0:
        sentiment = 'Negative'
    elif polarity == 0:
        sentiment = 'Neutral'
    else:
        sentiment = 'Positive'
    return subjectivity, polarity, sentiment

# Word Cloud Function
def generate_wordcloud(text):
    wc = WordCloud(background_color='white', max_words=50)
    wordcloud = wc.generate(text)
    return wordcloud

# Sidebar
st.sidebar.title("Data Information")
data = load_data()
# st.sidebar.subheader("Dataset Info")
st.sidebar.write(f"Dataset Shape: {data.shape}")
st.sidebar.subheader("Preview Data")
st.sidebar.write(data.head())
st.sidebar.subheader("Data Types")
st.sidebar.write(data.dtypes)

# Main Page
st.title("Visentify - Visualizing Sentiment Analysis")
st.markdown("---")



col1, col2 = st.columns([1,1])
with col1: 
	# Percentage wise distribution of ratings
	st.subheader("Percentage wise Distribution of Rating")
	rating_counts = data['rating'].value_counts(normalize=True) * 100
	fig = px.pie(
		names=rating_counts.index,
		values=rating_counts.values,
		color_discrete_sequence=px.colors.qualitative.Set3,
	)
	fig.update_traces(textinfo='percent+label', textposition='inside', insidetextorientation='radial', textfont_size=20,)
	fig.update_layout(
		showlegend=False,
		plot_bgcolor='rgba(0,0,0,0)',  
		paper_bgcolor='rgba(0,0,0,0)', 
		height=500,		
		width= 650
	)
	st.plotly_chart(fig)
 
with col2:   
	# Word Cloud for the whole data
	st.subheader("Word Cloud for All the Reviews")
	all_reviews_text = ' '.join(data['verified_reviews'])
	wordcloud = WordCloud(background_color='rgba(255, 255, 255, 0)', width=800, height=400).generate(all_reviews_text)

	# Plot Word Cloud
	fig = px.imshow(wordcloud.to_array(), binary_string=True)
	fig.update_layout(
		xaxis=dict(visible=False),
		yaxis=dict(visible=False),
		plot_bgcolor='rgba(0,0,0,0)',  
		paper_bgcolor='rgba(0,0,0,0)', 
	)
	st.plotly_chart(fig)


col1, col2 = st.columns([1,1])
with col1:
	# Sentiment analysis over time
	# Apply sentiment analysis function to each review and update the DataFrame
	data['Subjectivity'], data['Polarity'], data['Sentiment'] = zip(*data['verified_reviews'].apply(analyze_sentiment))
	# Convert 'date' column to datetime format
	data['date'] = pd.to_datetime(data['date'])
	st.subheader("Sentiment Analysis Over Time")
	# Calculate smoothed sentiment
	smoothed_sentiment = data.groupby('date')['Polarity'].mean().rolling(window=3, min_periods=1).mean()
	# Create a Plotly Express line plot
	fig = px.line(x=smoothed_sentiment.index, y=smoothed_sentiment.values, markers=True)
	fig.update_layout(
		xaxis=dict(title="Date"),
		yaxis=dict(title="Smoothed Average Polarity"),
		plot_bgcolor='rgba(0,0,0,0)', 
		paper_bgcolor='rgba(0,0,0,0)',
		height=550,
        width=600
	)
	st.plotly_chart(fig)


with col2:
	# Bar graph to visualize the total counts of each variation
	st.subheader("Variation Distribution Count")
	variation_counts = data['variation'].value_counts()
	fig = px.bar(variation_counts, orientation='h', color=variation_counts.index)
	fig.update_layout(
		showlegend=False,
		xaxis_title="Count",
		yaxis_title="Variation",
		plot_bgcolor='rgba(0,0,0,0)', 
		paper_bgcolor='rgba(0,0,0,0)',
		height=550,		
		width=600
	)
	st.plotly_chart(fig)


# # Review Summarization

# # Assuming you have a dataframe named 'data' containing the Amazon Alexa reviews
# variant_dataframes = {}
# for variant in data['variation'].unique():
#     variant_dataframes[variant] = data[data['variation'] == variant]

# from gensim.summarization import summarize

# # Generate summaries for each variant's dataframe
# for variant, df in variant_dataframes.items():
#     variant_dataframes[variant]['summary'] = df['verified_reviews'].apply(summarize)

# import streamlit as st

# # Dropdown to select variant
# selected_variant = st.selectbox("Select Variant", list(variant_dataframes.keys()))

# if selected_variant:
#     selected_summary = variant_dataframes[selected_variant]['summary']
#     st.write(selected_summary)
