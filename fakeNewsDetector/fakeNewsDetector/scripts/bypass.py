import ssl
import nltk

# Bypass SSL verification (temporary fix)
ssl._create_default_https_context = ssl._create_unverified_context

# Download stopwords
nltk.download('stopwords')
