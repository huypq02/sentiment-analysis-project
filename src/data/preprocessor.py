import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def ensure_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


class Preprocessor:
    def __init__(self):
        ensure_nltk_resources()

    def preprocess(self, text):
        """Implement stop words, lowercase, punctuation removal and tokenization"""
        # Lowercase and Punctuation Removal
        text = re.sub("[^\w\s\-]", " ", text).lower()

        # Define English Stop word
        en_stopwords = stopwords.words("english")

        # Tokenization
        tokens = word_tokenize(text)

        # Remove the English stopwords
        tokens = [token for token in tokens if token not in en_stopwords]
        return tokens
