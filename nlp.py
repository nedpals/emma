import spacy

try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model 'en_core_web_sm' loaded.")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please run:")
    print("python -m spacy download en_core_web_sm")
    nlp = None

def extract_keywords(text: str, use_fallback = False, include_verb = False) -> list[str]:
    """Extracts keywords from user input using spaCy (or fallback)."""
    if not text:
        return []

    if not nlp:
        if use_fallback:
            # Basic fallback if spaCy is not available
            return [word.lower() for word in text.split() if len(word) > 3]
        else:
            return []

    doc = nlp(text)
    token_types = ["NOUN", "PROPN"]
    if include_verb:
        token_types.append("VERB")

    keywords = [
        token.lemma_.lower()   # Use lemma, convert to lowercase
        for token in doc
        if not token.is_stop   # Exclude stop words (like 'the', 'is', 'in')
        and not token.is_punct # Exclude punctuation
        and token.pos_ in token_types
        and len(token.lemma_) > 2
    ]

    return list(set(keywords))
    