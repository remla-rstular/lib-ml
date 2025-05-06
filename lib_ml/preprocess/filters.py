import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


def sanitize_text(text: str) -> str:
    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s']", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def advanced_text_preprocessing(text: str):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))

    negation_words = {"no", "not", "nor", "neither", "never", "none"}
    stop_words = stop_words - negation_words
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word, pos="v") for word in tokens]  # 'v' for verbs
    tokens = [lemmatizer.lemmatize(word, pos="n") for word in tokens]  # 'n' for nouns

    # Handle common restaurant-specific terms
    tokens = [word.replace("'s", "") for word in tokens]  # Remove possessive

    # Reconstruct text
    return " ".join(tokens)


def restaurant_specific_preprocessing(text):
    # Standardize common terms
    text = re.sub(r"\bburger(s)?\b", "burger", text)
    text = re.sub(r"\bpizza(s)?\b", "pizza", text)

    # Handle emphasis (repeated characters)
    text = re.sub(r"(\w)\1{2,}", r"\1", text)  # "goooood" -> "good"

    # Handle negation patterns
    text = re.sub(r"\b(?:not|never|no)\b\s\b(\w+)", r"not_\1", text)

    return text
