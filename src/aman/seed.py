"""Small built-in sample knowledge base so AMAN works out of the box (and in tests).

CVE and ATT&CK items go through the SAME parsers as real downloaded data.
Texts are short summaries written for AMAN, shaped like the official sources.
For real coverage, run scripts/download_data.py and build the full corpus.
"""
from __future__ import annotations

from aman.ingest.mitre import parse_mitre
from aman.ingest.nvd import parse_nvd
from aman.schema import Document


def _ref(eid, name, url=None):
    return [{"source_name": "mitre-attack", "external_id": eid,
             "url": url or f"https://attack.mitre.org/techniques/{eid.replace('.', '/')}"}]


def _tech(eid, name, desc, tactics, detection="", platforms=("Windows", "Linux", "macOS"), sub=False):
    return {"type": "attack-pattern", "name": name, "description": desc, "external_references": _ref(eid, name),
            "kill_chain_phases": [{"kill_chain_name": "mitre-attack", "phase_name": t} for t in tactics],
            "x_mitre_detection": detection, "x_mitre_platforms": list(platforms), "x_mitre_is_subtechnique": sub}


MITRE_BUNDLE = {"objects": [
    {"type": "x-mitre-tactic", "name": "Credential Access", "x_mitre_shortname": "credential-access",
     "description": "The adversary is trying to steal account names and passwords. Credential Access consists of "
                    "techniques such as keylogging, credential dumping, brute force and stealing credentials from "
                    "password stores. Using legitimate credentials lets adversaries blend in, access systems and "
                    "make detection harder. Defenders mitigate it with multi-factor authentication, account "
                    "lockout policies, credential hygiene and monitoring of authentication logs.",
     "external_references": [{"source_name": "mitre-attack", "external_id": "TA0006",
                              "url": "https://attack.mitre.org/tactics/TA0006"}]},
    _tech("T1003", "OS Credential Dumping",
          "Adversaries may attempt to dump credentials to obtain account login and credential material, normally "
          "in the form of a hash or clear text password, from the operating system and software. Credentials can "
          "then be used to perform lateral movement and access restricted information. Several sub-techniques "
          "target specific stores such as LSASS memory, the SAM database or cached domain credentials.",
          ["credential-access"],
          "Monitor for unexpected processes accessing the memory of lsass.exe, and for command-line use of "
          "credential dumping tools. Enable credential protection features and alert on suspicious access to the SAM registry hive."),
    _tech("T1003.001", "LSASS Memory",
          "Adversaries may attempt to access credential material stored in the process memory of the Local "
          "Security Authority Subsystem Service (LSASS). After a user logs on, the system generates and stores "
          "credential material in LSASS memory, which an adversary with administrative rights can extract.",
          ["credential-access"], "Monitor for processes opening handles to lsass.exe with read-memory rights. "
          "Mitigation: enable Credential Guard and Windows Defender attack surface reduction rules, and restrict debug privileges.",
          ["Windows"], True),
    _tech("T1110", "Brute Force",
          "Adversaries may use brute force techniques to gain access to accounts when passwords are unknown or "
          "when password hashes are obtained. Without knowledge of the password, an adversary may systematically "
          "guess passwords. Sub-techniques include password guessing, password cracking, password spraying and credential stuffing.",
          ["credential-access"], "Monitor authentication logs for many failed attempts against one account or a few attempts across many accounts. "
          "Mitigation: multi-factor authentication, account lockout, and strong password policies.",
          ("Windows", "Linux", "macOS", "SaaS", "IaaS")),
    _tech("T1110.003", "Password Spraying",
          "Adversaries may use a single or small list of commonly used passwords against many different accounts "
          "to attempt to acquire valid account credentials. Spraying avoids account lockouts because each account "
          "sees only a few attempts.", ["credential-access"],
          "Alert on one source attempting logons across many accounts, and on bursts of failed logons with the same password.",
          ("Windows", "Linux", "SaaS", "IaaS"), True),
    _tech("T1566", "Phishing",
          "Adversaries may send phishing messages to gain access to victim systems. Phishing uses electronically "
          "delivered social engineering to target individuals. Common techniques are spearphishing attachments, "
          "spearphishing links and spearphishing through third-party services, where messages impersonate trusted "
          "senders to trick users into opening malicious files, following links or giving up credentials.",
          ["initial-access"], "Inspect email attachments and links, detect newly registered look-alike domains and "
          "monitor for users opening files that then spawn unusual child processes. Mitigation: email filtering, "
          "user awareness training, attachment sandboxing and multi-factor authentication.",
          ("Windows", "Linux", "macOS", "SaaS")),
    _tech("T1566.001", "Spearphishing Attachment",
          "Adversaries may send spearphishing emails with a malicious attachment to gain access to victim systems. "
          "The attachment, such as an Office document, archive or script, relies on the user opening it to execute.",
          ["initial-access"], "Detect Office applications spawning script interpreters and analyse attachments in a sandbox.",
          ("Windows", "Linux", "macOS"), True),
    _tech("T1566.002", "Spearphishing Link",
          "Adversaries may send spearphishing emails with a malicious link to gain access to victim systems. "
          "The link leads to a credential harvesting page or to malware the user is tricked into downloading.",
          ["initial-access"], "Inspect URLs in email, flag newly registered or look-alike domains and monitor for unusual browser downloads.",
          ("Windows", "Linux", "macOS", "SaaS"), True),
    _tech("T1078", "Valid Accounts",
          "Adversaries may obtain and abuse credentials of existing accounts to gain initial access, persistence, "
          "privilege escalation or defense evasion. Compromised credentials may be used to bypass access controls "
          "and to appear as legitimate activity.", ["initial-access", "persistence", "privilege-escalation", "defense-evasion"],
          "Monitor for logons from unusual locations or times and for accounts used outside their normal role.",
          ("Windows", "Linux", "macOS", "SaaS", "IaaS")),
]}


def _cve(cid, published, desc, score, vec, products, ref):
    return {"cve": {"id": cid, "published": published, "descriptions": [{"lang": "en", "value": desc}],
            "metrics": {"cvssMetricV31": [{"cvssData": {"version": "3.1", "baseScore": score,
                        "baseSeverity": "CRITICAL" if score >= 9 else "HIGH", "vectorString": vec}}]},
            "configurations": [{"nodes": [{"cpeMatch": [{"vulnerable": True, "criteria": p} for p in products]}]}],
            "references": [{"url": ref}]}}


NVD_PAYLOAD = {"vulnerabilities": [
    _cve("CVE-2024-3094", "2024-03-29T17:15:21.150",
         "Malicious code was discovered in the upstream tarballs of xz, starting with version 5.6.0. Through a "
         "series of complex obfuscations, the liblzma build process extracts a prebuilt object file from a disguised "
         "test file, which modifies specific functions in the liblzma code. This results in a modified liblzma library "
         "that can be used by any software linked against it, enabling interception and modification of data used with the library. "
         "It is a supply chain compromise (backdoor) affecting sshd on some systems.",
         10.0, "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
         ["cpe:2.3:a:tukaani:xz:5.6.0:*:*:*:*:*:*:*", "cpe:2.3:a:tukaani:xz:5.6.1:*:*:*:*:*:*:*"],
         "https://www.openwall.com/lists/oss-security/2024/03/29/4"),
    _cve("CVE-2021-44228", "2021-12-10T10:15:09.143",
         "Apache Log4j2 2.0-beta9 through 2.15.0 (excluding security releases 2.12.2, 2.12.3 and 2.3.1) JNDI features "
         "used in configuration, log messages and parameters do not protect against attacker-controlled LDAP and other "
         "JNDI related endpoints. An attacker who can control log messages or log message parameters can execute "
         "arbitrary code loaded from LDAP servers when message lookup substitution is enabled. Known as Log4Shell.",
         10.0, "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
         ["cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*", "cpe:2.3:a:apache:log4j:2.15.0:*:*:*:*:*:*:*"],
         "https://logging.apache.org/log4j/2.x/security.html"),
    _cve("CVE-2014-0160", "2014-04-07T18:55:03.893",
         "The (1) TLS and (2) DTLS implementations in OpenSSL 1.0.1 before 1.0.1g do not properly handle Heartbeat "
         "Extension packets, which allows remote attackers to obtain sensitive information from process memory via "
         "crafted packets that trigger a buffer over-read. Known as the Heartbleed bug.",
         7.5, "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N",
         ["cpe:2.3:a:openssl:openssl:1.0.1:*:*:*:*:*:*:*", "cpe:2.3:a:openssl:openssl:1.0.1f:*:*:*:*:*:*:*"],
         "https://www.openssl.org/news/secadv/20140407.txt"),
]}

G = "Security Guidance"
_DOCS = [
    ("OWASP-SQLI", "OWASP", "SQL Injection", "https://owasp.org/www-community/attacks/SQL_Injection",
     "SQL Injection is an attack in which untrusted input is inserted into a database query so that the input is "
     "interpreted as SQL code instead of data. The vulnerability arises when an application builds queries by "
     "concatenating strings. A successful attack can read or modify data, bypass authentication, delete records, and "
     "sometimes execute commands on the database server.\n\n"
     "Prevention: use parameterized queries (prepared statements) for all database access, use an ORM safely, validate "
     "input against an allow-list, apply least privilege to database accounts, and avoid showing detailed database "
     "errors to users. Web application firewalls are a useful extra layer but not a replacement for secure code."),
    ("OWASP-XSS", "OWASP", "Cross-Site Scripting (XSS)", "https://owasp.org/www-community/attacks/xss/",
     "Cross-Site Scripting (XSS) is an injection attack where malicious scripts are delivered to other users' browsers "
     "through a trusted website. The browser runs the script because it appears to come from the site. Types are "
     "stored XSS, reflected XSS and DOM-based XSS. Impact includes session theft, account takeover, defacement and "
     "redirecting users to malicious pages.\n\n"
     "Prevention: encode output according to context (HTML, attribute, JavaScript, URL), validate input, use frameworks "
     "that escape by default, set a strict Content Security Policy, and mark session cookies HttpOnly and Secure."),
    ("OWASP-CSRF", "OWASP", "Cross-Site Request Forgery (CSRF)", "https://owasp.org/www-community/attacks/csrf",
     "Cross-Site Request Forgery (CSRF) forces an authenticated user's browser to send an unwanted request to a web "
     "application where they are logged in. The attacker abuses the fact that the browser automatically attaches "
     "cookies. Unlike XSS, CSRF does not run attacker script in the victim's page; it makes the site trust a forged "
     "request, such as changing an email address or transferring funds.\n\n"
     "Prevention: use anti-CSRF tokens on state-changing requests, set SameSite cookie attributes, verify the Origin "
     "or Referer header and require re-authentication for sensitive actions."),
    ("OWASP-A07", "OWASP", "Identification and Authentication Failures (weak authentication)",
     "https://owasp.org/Top10/A07_2021-Identification_and_Authentication_Failures/",
     "Weak authentication lets attackers compromise passwords, keys or session tokens, or assume other users' identities. "
     "Risks include credential stuffing, brute force, weak or default passwords, missing multi-factor authentication, "
     "exposed session identifiers and poor password recovery flows. The result is account takeover and unauthorized "
     "access to data.\n\n"
     "Mitigation: enforce multi-factor authentication, never ship default credentials, check passwords against breached "
     "lists, limit and delay failed login attempts, and use secure server-side session management with new session "
     "identifiers after login."),
    ("OWASP-A06", "OWASP", "Vulnerable and Outdated Components (outdated software)",
     "https://owasp.org/Top10/A06_2021-Vulnerable_and_Outdated_Components/",
     "Using outdated or unsupported software exposes a system to publicly known vulnerabilities. When a system runs "
     "libraries, frameworks or operating systems without security patches, attackers can use public exploits, often "
     "automatically. Risks include remote code execution, data theft and full compromise, as seen with Log4Shell.\n\n"
     "Mitigation: keep an inventory of components and versions (a software bill of materials), monitor CVE and NVD "
     "feeds, apply patches promptly, remove unused dependencies and only obtain components from trusted sources."),
    ("GUIDE-PHISH", G, "Phishing attacks: common techniques", "",
     "Phishing is a social engineering attack that tricks people into revealing credentials, installing malware or "
     "sending money. Common techniques include deceptive emails that impersonate trusted brands, spearphishing aimed at "
     "specific people, malicious links to fake login pages, malicious attachments, smishing by SMS, vishing by phone, "
     "business email compromise and look-alike domains. Attackers create urgency or fear to make people act without thinking.\n\n"
     "Defenses: email filtering and sandboxing, multi-factor authentication (preferably phishing-resistant), user "
     "awareness training, DMARC, SPF and DKIM on your domains, and an easy way for users to report suspicious messages."),
    ("GUIDE-NETDET", G, "Detecting suspicious network activity", "",
     "Signs of suspicious network activity include unusual outbound connections, traffic to rare or newly registered "
     "domains, large data transfers at odd hours, repeated failed connections or port scans, beaconing at regular "
     "intervals and unexpected use of protocols such as DNS tunneling. Detection combines a baseline of normal traffic "
     "with flow data, firewall and DNS logs, intrusion detection systems and threat intelligence indicators.\n\n"
     "Recommendations: centralize logs in a SIEM, alert on deviations from baseline, segment the network, restrict "
     "egress traffic and investigate alerts using indicators of compromise."),
    ("GUIDE-XZ", G, "CVE-2024-3094 (xz) mitigation guidance", "",
     "For CVE-2024-3094, affected systems run xz or liblzma versions 5.6.0 or 5.6.1. Mitigation: downgrade to an "
     "uncompromised release such as 5.4.6 or update to the fixed packages provided by your distribution, then check "
     "that sshd is not linked against a compromised liblzma. Treat exposed systems as potentially compromised and "
     "review access logs and rotate credentials where appropriate."),
    ("GUIDE-LOG4J", G, "CVE-2021-44228 (Log4Shell) mitigation guidance", "",
     "For CVE-2021-44228 (Log4Shell), upgrade Apache Log4j2 to the latest fixed release in your supported line; "
     "version 2.15.0 was an incomplete first fix, and later releases (2.16.0 and 2.17.x) closed related issues. If upgrading is "
     "not immediately possible, remove the JndiLookup class from the classpath. Block outbound LDAP and RMI from "
     "application servers and search logs for jndi lookup strings as indicators of exploitation."),
    ("GUIDE-HEARTBLEED", G, "CVE-2014-0160 (Heartbleed) mitigation guidance", "",
     "For CVE-2014-0160 (Heartbleed), upgrade OpenSSL to 1.0.1g or later. Because the bug can leak private keys and "
     "session data from memory, revoke and reissue affected TLS certificates and rotate passwords and session "
     "secrets of affected services."),
]


def seed_documents() -> list[Document]:
    docs = parse_mitre(MITRE_BUNDLE) + parse_nvd(NVD_PAYLOAD)
    for did, src, title, url, text in _DOCS:
        docs.append(Document(did, src, title, text, url, {"type": "guide"}))
    for d in docs:
        d.metadata["seed"] = True
    return docs
