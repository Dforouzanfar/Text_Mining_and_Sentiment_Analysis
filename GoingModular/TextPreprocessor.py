import re
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')


# TextPreprocessor Class
class TextPreprocessor:
  def __init__(self):
    self.stop_words = set(stopwords.words("english"))
    self.stemmer = SnowballStemmer("english")

  def preprocess(self,
                 text: str) -> list:
    """
    Preprocess a single text entry.
    """
    text = text.lower()
    text = re.sub(r"#\w+|http[s]?://\S+|\brt\b", "", text)  # Remove hashtags, URLs, 'rt'
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"\d+", "", text)  # Remove digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphanumeric chars
    text = contractions.fix(text)  # Expand contractions
    text = " ".join(text.split())  # Remove extra whitespace
    tokens = word_tokenize(text)
    return [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
