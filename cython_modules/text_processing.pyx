# text_processing.pyx
'''import re

def remove_punctuation(text: str) -> str:
    pattern = re.compile(r"[^\w\s]")
    return pattern.sub("", text)

def lemmatize_text(tokens, stop_words, lemmatizer):
    return [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]'''
    
# text_processing.pyx
import re
import cython
from libc.stdlib cimport malloc, free
from libc.string cimport strlen
from nltk.stem import WordNetLemmatizer

@cython.boundscheck(False)
@cython.wraparound(False)
def remove_punctuation(text: str) -> str:
    cdef bytes b_text = text.encode('utf-8')  # Convert to bytes
    cdef int length = len(b_text)
    cdef char* c_text = <char*>malloc(length + 1)
    if not c_text:
        raise MemoryError()

    cdef int i = 0
    cdef int j = 0
    try:
        for i in range(length):
            if b_text[i:i+1].isalnum() or b_text[i:i+1].isspace():
                c_text[j] = b_text[i]
                j += 1
        c_text[j] = '\0'
        result = c_text[:j].decode('utf-8')
    finally:
        free(c_text)
    return result

'''@cython.boundscheck(False)
@cython.wraparound(False)
def lemmatize_text(list tokens, set stop_words, WordNetLemmatizer lemmatizer):
    cdef list lemmatized_tokens = []
    for word in tokens:
        if word not in stop_words:
            lemmatized_tokens.append(lemmatizer.lemmatize(word))
    return lemmatized_tokens'''

'''def lemmatize_text(tokens, stop_words, lemmatizer):
    return [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]'''

@cython.boundscheck(False)
@cython.wraparound(False)
def lemmatize_text(list tokens, set stop_words, lemmatizer):
    cdef list lemmatized_tokens = []
    for word in tokens:
        if word not in stop_words:
            lemmatized_tokens.append(lemmatizer.lemmatize(word))
    return lemmatized_tokens

