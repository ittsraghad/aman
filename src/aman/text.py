"""Tokenizing, language detection and Arabic->English query expansion.

Security IDs (CVE-2024-3094, T1003.001) are kept as single tokens so keyword
search can match them exactly. Arabic questions are expanded with English
security terms because most of our sources are English.
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"cve-\d{4}-\d{4,}|t\d{4}(?:\.\d{3})?|[a-z0-9]+|[\u0621-\u063a\u0641-\u064a]+")
CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}", re.I)
ATTACK_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.I)
_ARABIC_CHARS = re.compile(r"[\u0600-\u06ff]")
_DIACRITICS = re.compile(r"[\u064b-\u0652\u0640]")

STOPWORDS = set("""a an the and or of to in on for with by is are was were be been it its this that these those
what which who how why when where do does did can could should would may might as at from into about than then
i you we they he she them us our your their not no yes if so such using use used between difference
common techniques main""".split()) | set("""ما ماهو ماهي هو هي هذا هذه ذلك في من على الى إلى عن مع كيف لماذا هل او أو و ان إن يمكن
الفرق بين اشرح شرح وش ايش""".split())


def normalize_arabic(text: str) -> str:
    text = _DIACRITICS.sub("", text)
    text = re.sub("[إأآ]", "ا", text)
    return text.replace("ة", "ه").replace("ى", "ي")


def detect_language(text: str) -> str:
    ar = len(_ARABIC_CHARS.findall(text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return "ar" if ar > latin else "en"


def _stem(tok: str) -> str:
    if tok.startswith(("cve-",)) or re.fullmatch(r"t\d{4}(\.\d{3})?", tok):
        return tok
    if _ARABIC_CHARS.match(tok):
        return tok[2:] if tok.startswith("ال") and len(tok) > 4 else tok
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("es") and tok[-3] in "sxz":
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    raw = _TOKEN.findall(normalize_arabic(text.lower()))
    if not keep_stopwords:
        raw = [t for t in raw if t not in STOPWORDS]   # filter BEFORE stemming
    return [_stem(t) for t in raw]


# Arabic (normalized) -> English search terms. Substring match, so "الحقن" also matches "حقن".
GLOSSARY = {
    "حقن": "injection sql", "تصيد": "phishing", "احتيال": "phishing social engineering",
    "ثغره": "vulnerability", "ثغرات": "vulnerability", "هجوم": "attack", "هجمات": "attack",
    "كلمه المرور": "password authentication", "كلمات المرور": "password authentication",
    "مصادقه": "authentication", "ضعيفه": "weak", "ضعف": "weak",
    "بيانات الاعتماد": "credential", "اعتماد": "credential",
    "منع": "prevent mitigation", "وقايه": "prevent mitigation", "حمايه": "prevent mitigation",
    "تخفيف": "mitigation", "كشف": "detect detection", "خطوره": "severity cvss",
    "برمجه نصيه": "cross-site scripting xss", "عبر المواقع": "cross-site scripting xss",
    "تزوير الطلبات": "csrf cross-site request forgery", "قاعده بيانات": "database",
    "برمجيات قديمه": "outdated software vulnerable components", "تحديث": "patch update",
    "فدييه": "ransomware", "ثغره يوم الصفر": "zero-day", "تخمين": "brute force",
    "القوه الغاشمه": "brute force", "تسريب": "leak dump", "ذاكره": "memory",
    "احمي": "prevent mitigation protect", "اصلح": "patch fix mitigation", "اعالج": "mitigation fix",
    "اتجنب": "prevent avoid", "اوقف": "block prevent", "اتحمي": "prevent mitigation protect",
    "اكتشف": "detect detection", "شبكه": "network", "مشبوه": "suspicious", "نشاط": "activity",
    "تنفيذ اوامر": "remote code execution", "باب خلفي": "backdoor",
}


def expand_query(query: str) -> str:
    """Return the query plus English expansions when it contains Arabic."""
    if not _ARABIC_CHARS.search(query):
        return query
    norm = normalize_arabic(query)
    extra = [en for ar, en in GLOSSARY.items() if normalize_arabic(ar) in norm]
    return query + " " + " ".join(extra) if extra else query
