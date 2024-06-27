# text_processing.pyx
import re

def remove_punctuation(text: str) -> str:
    pattern = re.compile(r"[^\w\s]")
    return pattern.sub("", text)

def lemmatize_text(tokens, stop_words, lemmatizer):
    return [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
