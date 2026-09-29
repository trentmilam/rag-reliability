"""Deterministic text fixtures + genuine fault-injection transforms.

The corrupted chunks are DERIVED from the clean ones by real, mechanical
transforms (interleave two columns; flatten table delimiters). Nothing about
the fault is hand-labeled: the gate must catch it from the text alone.
"""

# Two topically-distinct, locally-cohesive single-column passages.
COLUMN_A = [
    "Plants capture sunlight using chlorophyll in their leaves.",
    "The chlorophyll molecules absorb light energy from the sun.",
    "This light energy drives the splitting of water molecules.",
    "Splitting water releases oxygen into the surrounding air.",
    "The released oxygen is a byproduct plants do not need.",
    "Meanwhile the plant stores energy inside sugar molecules.",
    "These sugar molecules feed the plant and its new tissues.",
    "New tissues let the plant grow more leaves and deep roots.",
]

COLUMN_B = [
    "Quarterly revenue rose sharply across the retail division.",
    "The retail division reported record sales this fiscal quarter.",
    "Fiscal quarter earnings beat the analyst consensus estimate.",
    "Analyst estimates had projected much slower earnings growth.",
    "Slower growth was expected due to steadily rising supply costs.",
    "Supply costs increased because of higher inbound shipping tariffs.",
    "Higher tariffs pressured the overall profit margin outlook badly.",
    "The profit margin outlook now remains cautious for next year.",
]

# A well-formed table: aligned columns separated by multi-space gaps.
CLEAN_TABLE = [
    "Region      Units    Revenue",
    "North        1200     45000",
    "South         980     37000",
    "East         1450     52000",
    "West          760     28000",
]


def clean_prose():
    """Correct single-column reading order."""
    return "\n".join(COLUMN_A)


def interleaved_prose():
    """FAULT: two columns interleaved line-by-line (A1,B1,A2,B2,...)."""
    rows = []
    for a, b in zip(COLUMN_A, COLUMN_B):
        rows.append(a)
        rows.append(b)
    return "\n".join(rows)


def clean_table():
    """Well-formed table with intact cell delimiters."""
    return "\n".join(CLEAN_TABLE)


def collapsed_table():
    """FAULT: cell delimiters flattened. Every 2+ space run becomes one space.

    Row/column structure is destroyed; each row parses as a single blob.
    """
    import re
    return "\n".join(re.sub(r" {2,}", " ", ln).strip() for ln in CLEAN_TABLE)
