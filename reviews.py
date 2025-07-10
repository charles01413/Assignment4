import spacy
from textblob import TextBlob

# 🔧 Load spaCy model
nlp = spacy.load("en_core_web_sm")

# 📋 Sample reviews
reviews = [
    "The Apple iPhone 13 is impressive.",
    "Samsung Galaxy feels fast and stylish.",
    "Sony headphones are overpriced for the value."
]

# 📌 NER + Sentiment Analysis
for review in reviews:
    doc = nlp(review)
    print(f"Review: {review}")

    print("Entities:")
    for ent in doc.ents:
        print(f" - {ent.text} ({ent.label_})")

    sentiment = TextBlob(review).sentiment.polarity
    sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
    print(f"Sentiment: {sentiment_label}")
    print("—" * 40)
