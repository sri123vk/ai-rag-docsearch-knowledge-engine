from __future__ import annotations

import re


WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")

DOMAIN_KEYWORDS = {
    "privacy_compliance": ["privacy", "consent", "pii", "data subject", "retention", "deletion"],
    "financial_controls": ["sox", "journal entry", "revenue", "financial", "auditor", "quarter"],
    "security_risk": ["security", "vulnerability", "encryption", "incident", "access", "endpoint"],
    "vendor_risk": ["vendor", "supplier", "third-party", "contract", "service level", "termination"],
    "ai_governance": ["ai system", "model", "training data", "bias", "fairness", "human oversight"],
}

RISK_PATTERNS = {
    "critical": ["critical exception", "material weakness", "breach notification", "executive acknowledgement"],
    "high": ["pii", "cross-border", "vulnerability", "unauthorized", "termination risk"],
    "medium": ["late approval", "missing approval", "exception", "remediation"],
    "low": ["monitoring", "review", "documentation"],
}

ENTITY_PATTERNS = {
    "ROLE": [
        "control owner",
        "auditor",
        "reviewer",
        "compliance analyst",
        "executive",
        "issue owner",
    ],
    "CONTROL": [
        "access review",
        "approval workflow",
        "audit evidence",
        "control testing",
        "change management",
        "encryption",
    ],
    "RISK": [
        "critical exception",
        "material weakness",
        "pii",
        "breach notification",
        "vulnerability",
        "termination risk",
    ],
    "ARTIFACT": [
        "remediation ticket",
        "system record",
        "evidence package",
        "policy",
        "contract",
        "data processing addendum",
    ],
    "AI": [
        "ai system",
        "model",
        "training data",
        "bias",
        "fairness",
        "human oversight",
    ],
}


def tokenize_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_PATTERN.finditer(text)]

