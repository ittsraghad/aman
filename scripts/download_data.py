"""Download raw source data into data/raw (run on YOUR machine; needs internet; standard library only).

  python scripts/download_data.py all                     # mitre + kev + owasp + nvd (last 365 days, CRITICAL)
  python scripts/download_data.py nvd --days 730 --severity HIGH --max 3000
  python scripts/download_data.py owasp | mitre | kev

Optional: export NVD_API_KEY=... (much faster NVD rate limit).
Source licences: check each provider's terms and keep the attribution shown in the app's About page.
If a URL ever changes, the script prints the HTTP error; fix the constant below.
"""
import argparse
import datetime as dt
import io
import json
import os
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

RAW = Path("data/raw")
MITRE_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
OWASP_ZIP = "https://github.com/OWASP/CheatSheetSeries/archive/refs/heads/master.zip"
NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def get(url: str, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "aman-project", **(headers or {})})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def mitre():
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "enterprise-attack.json").write_bytes(get(MITRE_URL))
    print("MITRE ATT&CK downloaded")


def kev():
    RAW.mkdir(parents=True, exist_ok=True)
    data = get(KEV_URL)
    n = len(json.loads(data).get("vulnerabilities", []))
    (RAW / "kev.json").write_bytes(data)
    print(f"CISA KEV downloaded ({n} vulnerabilities)")


def owasp():
    out = RAW / "owasp"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(io.BytesIO(get(OWASP_ZIP))) as z:
        for name in z.namelist():
            if "/cheatsheets/" in name and name.endswith(".md"):
                (out / Path(name).name).write_bytes(z.read(name))
                n += 1
    print(f"OWASP Cheat Sheets downloaded ({n} files)")


def nvd(days: int, severity: str, max_items: int):
    out = RAW / "nvd"
    out.mkdir(parents=True, exist_ok=True)
    headers = {"apiKey": os.environ["NVD_API_KEY"]} if os.getenv("NVD_API_KEY") else {}
    pause = 0.8 if headers else 6.5
    end, total = dt.datetime.utcnow(), 0
    while days > 0 and total < max_items:
        span = min(days, 119)                              # NVD allows max 120-day windows
        start = end - dt.timedelta(days=span)
        idx = 0
        while True:
            q = {"pubStartDate": start.strftime("%Y-%m-%dT%H:%M:%S.000"), "pubEndDate": end.strftime("%Y-%m-%dT%H:%M:%S.000"),
                 "cvssV3Severity": severity, "resultsPerPage": 2000, "startIndex": idx}
            raw = get(f"{NVD_URL}?{urllib.parse.urlencode(q)}", headers)
            page = json.loads(raw)
            (out / f"{start:%Y%m%d}_{end:%Y%m%d}_{idx}.json").write_bytes(raw)
            got = len(page.get("vulnerabilities", []))
            total += got
            idx += got
            print(f"  NVD {start:%Y-%m-%d}..{end:%Y-%m-%d}: {idx}/{page.get('totalResults', 0)}")
            time.sleep(pause)
            if got == 0 or idx >= page.get("totalResults", 0) or total >= max_items:
                break
        end, days = start, days - span
    print(f"NVD downloaded ({total} CVEs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["all", "mitre", "kev", "owasp", "nvd"])
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--severity", default="CRITICAL", choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"])
    ap.add_argument("--max", type=int, default=3000)
    a = ap.parse_args()
    if a.what in ("all", "mitre"): mitre()
    if a.what in ("all", "kev"): kev()
    if a.what in ("all", "owasp"): owasp()
    if a.what in ("all", "nvd"): nvd(a.days, a.severity, a.max)
