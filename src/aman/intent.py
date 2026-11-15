"""Rule-based intent routing + safety guardrails.

Runs BEFORE retrieval, so greetings and abuse never reach the LLM (saves quota, avoids odd answers).
Intents: greeting | thanks | about | offensive | security (default -> RAG).
"""
from __future__ import annotations

import re

from aman.text import normalize_arabic

_GREET = re.compile(r"^(hi|hello|hey|hiya|salam|salaam|good (morning|evening)|السلام عليكم|سلام|مرحبا|مرحبتين|هلا|اهلا|"
                    r"يا هلا|هاي|هلو|صباح الخير|مساء الخير|حياك|حياكم)\b", re.I)
_THANKS = re.compile(r"(^|\s)(thanks|thank you|thx|شكرا|مشكور|يعطيك العافيه|الله يعطيك|تسلم|يسلمو)", re.I)
_ABOUT = re.compile(r"(who are you|what (are you|can you do|do you do)|how do you work|introduce yourself|"
                    r"من انت|مين انت|وش تسوي|ايش تسوي|ماذا تفعل|ماذا تستطيع|وش تقدر|وش تعرف|عرفني بنفسك|كيف تشتغل|كيف تعمل)")
_OFFENSIVE = re.compile(
    r"((write|create|build|make|generate|code|develop|give me)\b.{0,40}\b(ransomware|keylogger|malware|virus|botnet|"
    r"rat|trojan|exploit|payload|phishing (page|kit|site|email)|ddos tool)|"
    r"(hack|break) into\b.{0,30}\b(account|phone|wifi|wi-fi|computer|server|website|someone|her|his|ex)|"
    r"(steal|crack)\b.{0,20}\b(password|credentials|account)s? (from|of)|"
    r"(اكتب|سوي|سو|اعمل|برمج|انشئ|ابي|ابغى|اريد)\b.{0,40}(فديه|فيروس|برنامج تجسس|كيلوجر|كي لوجر|اكسبلويت|استغلال|"
    r"تروجان|صفحه تصيد|رساله تصيد)|"
    r"(اخترق|اهكر|هكر|كيف اخترق|طريقه اختراق)\b.{0,30}(حساب|جهاز|واي فاي|هاتف|جوال|موقع|سيرفر|شخص))", re.I)


_QUESTION_WORDS = {"how", "what", "why", "can", "does", "is", "are", "كيف", "ما", "ماذا", "لماذا", "هل", "وش", "ايش"}


def classify(text: str) -> str:
    t = normalize_arabic(text.lower()).strip()
    if _OFFENSIVE.search(t):
        return "offensive"
    short = len(t.split()) <= 6
    g = _GREET.search(t)
    if g and short and not re.search(r"cve-|\bt\d{4}|[?؟]", t):
        rest = t[g.end():].split()
        if len(rest) <= 3 and not (set(rest) & _QUESTION_WORDS):     # "hello there" yes; "مرحبا كيف احمي..." no
            return "greeting"
    if short and _THANKS.search(t):
        return "thanks"
    if len(t.split()) <= 8 and _ABOUT.search(t):
        return "about"
    return "security"


CANNED = {
    "greeting": {
        "en": "Hi! I'm AMAN, your cybersecurity assistant. Ask me about a vulnerability (CVE), an attack technique "
              "(MITRE ATT&CK), or how to defend against a threat. I answer in Arabic or English and show my sources.",
        "ar": "أهلًا بك! أنا «أمان»، مساعدك في الأمن السيبراني. اسألني عن ثغرة (CVE) أو تقنية هجوم (MITRE ATT&CK) "
              "أو كيف تحمي نفسك من تهديد معيّن، بالعربي أو الإنجليزي، وأعرض لك مصادر كل إجابة."},
    "thanks": {
        "en": "You're welcome! Ask me anything else about security whenever you like.",
        "ar": "العفو! إذا عندك أي سؤال ثاني في الأمن السيبراني أنا حاضر."},
    "about": {
        "en": "I'm AMAN, an AI assistant for cybersecurity. I search trusted security sources (MITRE ATT&CK, NVD, "
              "CISA, OWASP), then write an answer based only on what I found and cite it. If my sources don't cover "
              "a question, I'll say so instead of guessing. I focus on understanding and defending, not attacking.",
        "ar": "أنا «أمان»، مساعد ذكاء اصطناعي متخصص في الأمن السيبراني. أبحث في مصادر أمنية موثوقة (MITRE ATT&CK وNVD "
              "وCISA وOWASP)، ثم أكتب إجابة مبنية على ما وجدته فقط مع ذكر المصادر. وإذا ما غطّت مصادري سؤالك أخبرك بدل "
              "أن أخمّن. تركيزي على الفهم والدفاع، وليس على تنفيذ الهجمات."},
    "clarify": {
        "en": "I'm not sure what you're referring to. Could you name the vulnerability, attack or technique you mean? "
              "For example:",
        "ar": "ما فهمت وش تقصد بالضبط. ممكن تذكر اسم الثغرة أو الهجوم أو التقنية اللي تسأل عنها؟ مثلًا:"},
    "offensive": {
        "en": "I can't help with building attack tools or breaking into systems or accounts. I'm happy to explain how "
              "the threat works, how to detect it, or how to defend against it. What would you like to protect?",
        "ar": "ما أقدر أساعد في بناء أدوات هجومية أو اختراق أنظمة أو حسابات. لكن يسعدني أشرح لك كيف يعمل هذا التهديد "
              "وكيف تكتشفه وتحمي نفسك منه. وش اللي تبغى تحميه؟"},
}

SUGGESTIONS = {
    "en": ["What is CVE-2024-3094?", "How does SQL Injection work?", "What is Credential Access?",
           "How can I defend against phishing?"],
    "ar": ["ما هي ثغرة CVE-2021-44228؟", "كيف يعمل هجوم حقن SQL؟", "ما هو Credential Access؟", "كيف أحمي نفسي من التصيد؟"],
}
