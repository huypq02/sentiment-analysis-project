from data_loader import DataLoader
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class Preprocessor():
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
    
    def preprocess(self, text):
        """Implement stop words, lowercase, punctual removal and tokenization"""
        # Lowercase and Punctual Removal
        text = re.sub("[^\w\s\-]", " ", text).lower()

        # Define English Stop word
        en_stopwords = stopwords.words('english')

        # Tokenization
        tokens = word_tokenize(text)

        # Remove the English topwords
        tokens = [token for token in tokens if token not in en_stopwords]
        return tokens