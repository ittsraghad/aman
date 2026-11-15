"""List which Gemini models your API key can use (names change over time).
Usage: GEMINI_API_KEY=... python scripts/check_providers.py"""
import json
import os
import urllib.request

key = os.environ["GEMINI_API_KEY"]
req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                             headers={"x-goog-api-key": key})
models = json.load(urllib.request.urlopen(req, timeout=30)).get("models", [])
for label, method in (("TEXT GENERATION (AMAN_GEMINI_MODELS)", "generateContent"),
                      ("EMBEDDINGS (AMAN_GEMINI_EMBED_MODEL)", "embedContent")):
    print(f"\n{label}")
    for m in models:
        if method in m.get("supportedGenerationMethods", []):
            print("  ", m["name"].replace("models/", ""))
