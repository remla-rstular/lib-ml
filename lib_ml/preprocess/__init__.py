from sklearn.base import BaseEstimator, TransformerMixin

from lib_ml.preprocess.filters import (
    advanced_text_preprocessing,
    restaurant_specific_preprocessing,
    sanitize_text,
)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        enable_sanitize: bool = True,
        enable_advanced: bool = True,
        enable_restaurant: bool = True,
    ):
        self.enable_sanitize = enable_sanitize
        self.enable_advanced = enable_advanced
        self.enable_restaurant = enable_restaurant

    def fit(self, _X, _y=None):
        return self

    def transform(self, X):
        processed_texts = []
        for text in X:
            if self.enable_sanitize:
                text = sanitize_text(text)
            if self.enable_advanced:
                text = advanced_text_preprocessing(text)
            if self.enable_restaurant:
                text = restaurant_specific_preprocessing(text)
            processed_texts.append(text)
        return processed_texts


def process_text(
    text: list[str],
    enable_sanitize: bool = True,
    enable_advanced: bool = True,
    enable_restaurant: bool = True,
) -> list[str]:
    preprocessor = TextPreprocessor(
        enable_sanitize=enable_sanitize,
        enable_advanced=enable_advanced,
        enable_restaurant=enable_restaurant,
    )
    return preprocessor.transform(text)
