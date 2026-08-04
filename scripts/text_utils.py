import re


def clean_review_text(text: str) -> str:
    text = str(text)
    text = text.lower()
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
