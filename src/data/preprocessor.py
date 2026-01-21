import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def ensure_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
        nltk.download("punkt_tab")

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
        text = re.sub(r"[^\w\s\-]", " ", text).lower()

        # Define English Stop word but exclude negation words (critical for sentiment)
        en_stopwords = set(stopwords.words("english"))
        # Remove negation words from stopwords - these are CRITICAL for sentiment
        negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 
                         'nowhere', 'nor', "don't", "doesn't", "didn't", "won't", 
                         "wouldn't", "shouldn't", "can't", "couldn't", "isn't", 
                         "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't"}
        en_stopwords = en_stopwords - negation_words

        # Tokenization
        tokens = word_tokenize(text)

        # Remove the English stopwords (but keep negations)
        tokens = [token for token in tokens if token not in en_stopwords]
        return tokens
