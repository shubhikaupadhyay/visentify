import pandas as pd
import pytest
from app import analyze_sentiment, generate_wordcloud, load_data  

def test_load_data():
    data = load_data()
    assert isinstance(data, pd.DataFrame), "Data should be a DataFrame"
    assert 'verified_reviews' in data.columns, "'verified_reviews' column is missing"
    assert 'rating' in data.columns, "'rating' column is missing"
    assert 'date' in data.columns, "'date' column is missing"

def test_analyze_sentiment():
    text = "I love this product!"
    subjectivity, polarity, sentiment = analyze_sentiment(text)
    assert isinstance(subjectivity, float), "Subjectivity should be a float"
    assert isinstance(polarity, float), "Polarity should be a float"
    assert sentiment in ['Positive', 'Neutral', 'Negative'], "Sentiment should be one of 'Positive', 'Neutral', 'Negative'"

def test_generate_wordcloud():
    text = "word cloud example test"
    wordcloud = generate_wordcloud(text)
    assert wordcloud is not None, "Wordcloud should be generated"
    assert len(wordcloud.words_) > 0, "Wordcloud should contain words"

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'verified_reviews': ["I love this product!", "This is a terrible product."],
        'rating': [5, 1],
        'date': ['2023-01-01', '2023-01-02'],
        'variation': ['Color1', 'Color2']
    })
    data['date'] = pd.to_datetime(data['date'])
    return data

def test_sentiment_analysis_over_time(sample_data):
    data = sample_data
    data['Subjectivity'], data['Polarity'], data['Sentiment'] = zip(*data['verified_reviews'].apply(analyze_sentiment))
    assert 'Subjectivity' in data.columns, "'Subjectivity' column should be present"
    assert 'Polarity' in data.columns, "'Polarity' column should be present"
    assert 'Sentiment' in data.columns, "'Sentiment' column should be present"
    assert data['Sentiment'].iloc[0] == 'Positive', "The first sentiment should be 'Positive'"
    assert data['Sentiment'].iloc[1] == 'Negative', "The second sentiment should be 'Negative'"
