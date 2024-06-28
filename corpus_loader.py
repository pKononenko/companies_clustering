import nltk


# Function to check if NLTK data is already downloaded
def check_nltk_data():
    try:
        nltk.data.find("corpora/stopwords")
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("stopwords")
        nltk.download("wordnet")
