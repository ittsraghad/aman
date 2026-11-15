"""Text cleaning. Source data contains markup that would pollute embeddings."""
import re

# MITRE descriptions embed references like "(Citation: Mandiant APT1)"
_CITATION = re.compile(r"\(Citation:[^)]*\)")
# Markdown links: [text](https://...) -> text
_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://|/)[^)]*\)")
# <code>...</code> and other simple tags
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = _CITATION.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _TAG.sub("", text)
    text = text.replace("\r\n", "\n")
    text = _MULTI_SPACE.sub(" ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()
