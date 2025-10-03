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

        # Stop word removal
        en_stopwords = stopwords.words('english')

        # Both English and Vietnamese Tokenization
        tokens = word_tokenize(text)

        # Remove both English and Vietnamese stopwords
        tokens = [token for token in tokens if token not in en_stopwords and token not in vi_stopwords]
        return tokens