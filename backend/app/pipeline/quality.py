"""Validation and conservative duration rewriting for translated utterances."""

import json
import re
import urllib.request
import unicodedata
from ..config import settings


def digits(text):
    normalized = "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)
    return sorted(re.findall(r"\d+(?:[.,]\d+)*", normalized))


def issues(source, translated, glossary=None):
    found = []
    if not translated.strip():
        found.append("empty translation")
    if digits(source) != digits(translated):
        found.append("numbers differ; review translation")
    for original, expected in (glossary or {}).items():
        if (
            original.casefold() in source.casefold()
            and expected.casefold() not in translated.casefold()
        ):
            found.append(f"required term missing: {expected}")
    return found


def rewrite(text, source, language, seconds, glossary=None):
    prompt = (
        f"Rewrite this translated utterance in {language} for at most {seconds:.2f} seconds of natural speech. "
        "Preserve every factual claim, negation, name and number. Do not summarize away information. "
        "If it cannot be shortened safely return the original translation. Return only the rewritten text.\n"
        f"Source: {source}\nTranslation: {text}\nRequired terminology: {json.dumps(glossary or {}, ensure_ascii=False)}"
    )
    request = urllib.request.Request(
        f"{settings.ollama_host.rstrip('/')}/api/generate",
        data=json.dumps(
            {
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        candidate = json.load(response).get("response", "").strip()
    problems = issues(source, candidate, glossary)
    if problems:
        raise ValueError("unsafe duration rewrite: " + "; ".join(problems))
    if len(candidate) >= len(text):
        raise ValueError("no shorter faithful translation produced")
    return candidate
