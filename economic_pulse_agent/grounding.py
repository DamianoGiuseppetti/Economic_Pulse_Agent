"""
Deterministic check for explanations the data doesn't support.

Haiku reliably quotes numbers from tool results but sometimes adds causes
from general knowledge ("AI optimism", "earnings growth") that no tool
returned. A prompt rule alone fixed this about 1 time in 3, so the agent
runs this check on every final answer and asks for one revision when it
fires.

The check is a word list, not a judgement: a sentence is flagged when it
names a cause from CAUSE_PATTERNS that appears nowhere in the evidence (tool
results this turn plus earlier answers) and doesn't carry the label (exact
wording or a variant containing "not verified by my data").
"""

import re

LABEL = "(not verified by my data)"
# The model sometimes elaborates inside the label ("(not verified by my data: the
# earnings mechanism)"); any sentence containing this phrase counts as labelled.
_LABEL_PHRASE = re.compile(r"not verified by my data", re.I)

# Causes the model tends to reach for that none of the tools report on.
# Words the tools *can* return (e.g. "geopolitical" in FOMC statements) are
# still listed: they pass whenever they appear in the evidence.
CAUSE_PATTERNS = [
    r"\bAI\b",
    r"artificial intelligence",
    r"earnings",
    r"tariff",
    r"trade war",
    r"geopolit",
    r"stimulus",
    r"\btech(nology)?\b",
    r"oil price",
    r"election",
    r"supply chain",
    r"\bbank(ing)? (crisis|failure)",
    r"\bwar\b",
    # Market expectations about Fed moves. No tool reports these; plain facts
    # ("the Fed cut rates by 70 bp", from FEDFUNDS) are not matched.
    r"expectations? (of|for) (further |more |additional )?(the )?(fed(eral reserve)?'?s? )?rate (cuts?|hikes?|increases?)",
    r"rate[- ](cut|hike)s? (expectations?|hopes?|bets?|pricing)",
    r"pric(ed|ing) in (further |more |additional )?(fed )?rate (cuts?|hikes?)",
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def find_unsupported(answer: str, evidence: str) -> list[dict]:
    """
    Sentences in `answer` that name a cause absent from `evidence` without LABEL.

    Returns [{"term": "AI", "sentence": "..."}], one entry per flagged sentence.
    """
    flagged = []
    for sentence in _SENTENCE_SPLIT.split(answer):
        if not sentence.strip() or _LABEL_PHRASE.search(sentence):
            continue
        for pattern in CAUSE_PATTERNS:
            match = re.search(pattern, sentence, re.I)
            if match and not re.search(pattern, evidence, re.I):
                flagged.append({"term": match.group(0), "sentence": sentence.strip()})
                break
    return flagged


def revision_request(flagged: list[dict]) -> str:
    """The one-shot instruction sent back to the model when the check fires."""
    lines = "\n".join(f'- "{f["sentence"]}" (mentions {f["term"]})' for f in flagged)
    return (
        "Before this answer goes out: these sentences name causes that none of the tool results "
        f"in this conversation support:\n{lines}\n\n"
        f'Rewrite the answer. For each of these sentences, either remove it or keep it and add "{LABEL}" '
        "inside the same sentence. Change nothing else. Reply with the full revised answer only, "
        "with no preamble."
    )
