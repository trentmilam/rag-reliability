"""FAIR incumbent baseline: a commodity text-quality / gibberish filter.

This is the real comparator for the A/B in eval.py. It is NOT a strawman:
it is a reasonable, competent implementation of exactly what the cited prior art
(CCNet/datatrove perplexity filtering, Gopher/C4 quality heuristics, langid-style
gibberish detection) actually does to decide whether an extracted chunk is
low-quality and should be dropped before indexing:

  1. Character-trigram perplexity vs a clean-English reference model
     (the CCNet/datatrove approach). The reference model is trained on the CLEAN
     half of the corpus, the most FAVORABLE setup for the incumbent, so clean
     text scores lowest and anomalous text has the best chance to stand out.

  2. Gopher/C4-style quality heuristics: symbol-to-word ratio, fraction of
     non-alphabetic characters, mean word length, and stopword ratio.

The threshold for each check is CALIBRATED on the clean chunks so the incumbent
keeps all clean content (no cheap false-positive rejection), then it flags
anything that trips a check. A chunk is quarantined if perplexity OR any
heuristic trips. This is intentionally generous to the incumbent.

The point of the A/B is not that this filter is bad: it is good at its job
(mojibake, gibberish, non-language, junk characters). It simply CANNOT SEE
structural corruption: interleaved columns and delimiter-collapsed tables are,
character-for-character and word-for-word, fluent English, so a quality/
gibberish filter passes them. That is the underserved wedge Legigate owns.
"""
import math
import re
from collections import defaultdict

_WORD = re.compile(r"[A-Za-z]+")
_ALNUM = re.compile(r"[A-Za-z0-9]")

# A small, common English stopword set (Gopher requires enough of these).
_STOPWORDS = {
    "the", "and", "of", "to", "a", "in", "is", "it", "for", "on", "with",
    "as", "are", "was", "at", "by", "an", "be", "this", "that", "from",
    "or", "they", "their", "its", "into", "each", "both", "have", "has",
}


# --- 1. character-trigram perplexity model ----------------------------------
class CharLM:
    """Add-one smoothed character-trigram model; scores mean neg-log-prob."""

    def __init__(self):
        self._ctx = defaultdict(lambda: defaultdict(int))
        self._vocab = set()

    @staticmethod
    def _prep(text):
        # normalize whitespace so the model sees the same char stream regardless
        # of how many spaces separated tokens (a real char-LM is not fooled by
        # single- vs multi-space, which is exactly why it misses collapsed
        # tables). Lowercased letters/digits/space only.
        t = re.sub(r"\s+", " ", text.lower()).strip()
        return "  " + t + " "  # pad start context

    def train(self, texts):
        for text in texts:
            s = self._prep(text)
            for i in range(2, len(s)):
                ctx, nxt = s[i - 2:i], s[i]
                self._ctx[ctx][nxt] += 1
                self._vocab.add(nxt)
        return self

    def mean_neglogprob(self, text):
        s = self._prep(text)
        v = max(1, len(self._vocab))
        total, n = 0.0, 0
        for i in range(2, len(s)):
            ctx, nxt = s[i - 2:i], s[i]
            row = self._ctx.get(ctx, {})
            count = row.get(nxt, 0)
            denom = sum(row.values())
            prob = (count + 1) / (denom + v)  # add-one smoothing
            total += -math.log(prob)
            n += 1
        return (total / n) if n else 0.0


# --- 2. Gopher/C4-style quality heuristics ----------------------------------
def _symbol_word_ratio(text):
    words = _WORD.findall(text)
    symbols = len(re.findall(r"[#.]{1}", text))
    return (symbols / len(words)) if words else 0.0


def _frac_non_alpha(text):
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 1.0
    non_alnum = sum(1 for c in chars if not _ALNUM.match(c))
    return non_alnum / len(chars)


def _mean_word_len(text):
    words = _WORD.findall(text)
    return (sum(len(w) for w in words) / len(words)) if words else 0.0


def _stopword_ratio(text):
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
    if not words:
        return 0.0
    return sum(1 for w in words if w in _STOPWORDS) / len(words)


class IncumbentQualityFilter:
    """Perplexity + Gopher heuristics, calibrated on the clean chunks."""

    def __init__(self, ppl_margin=0.15):
        self.lm = CharLM()
        self.ppl_max = None
        self.mean_word_lo = None
        self.mean_word_hi = None
        self.symbol_max = None
        self.nonalpha_max = None
        self.stopword_min = None
        self._ppl_margin = ppl_margin

    def fit(self, clean_texts):
        """Train the char-LM and calibrate thresholds so ALL clean text passes.

        Bounds are set to the observed clean range (with a small margin), which
        is the most generous stance toward the incumbent: it never false-flags
        clean content, so any miss on corrupted content is a genuine blind spot,
        not a mis-tuned threshold.
        """
        self.lm.train(clean_texts)
        ppls = [self.lm.mean_neglogprob(t) for t in clean_texts]
        # perplexity ceiling = worst clean chunk + a margin
        self.ppl_max = max(ppls) * (1 + self._ppl_margin)
        mwl = [_mean_word_len(t) for t in clean_texts]
        self.mean_word_lo = min(mwl) - 1.0
        self.mean_word_hi = max(mwl) + 1.0
        self.symbol_max = max(_symbol_word_ratio(t) for t in clean_texts) + 0.1
        self.nonalpha_max = max(_frac_non_alpha(t) for t in clean_texts) + 0.05
        self.stopword_min = max(0.0, min(_stopword_ratio(t) for t in clean_texts) - 0.05)
        return self

    def reasons(self, text):
        out = []
        ppl = self.lm.mean_neglogprob(text)
        if ppl > self.ppl_max:
            out.append(f"perplexity {ppl:.3f} > {self.ppl_max:.3f}")
        if _symbol_word_ratio(text) > self.symbol_max:
            out.append("symbol/word ratio too high")
        if _frac_non_alpha(text) > self.nonalpha_max:
            out.append("too many non-alphanumeric chars")
        mwl = _mean_word_len(text)
        if not (self.mean_word_lo <= mwl <= self.mean_word_hi):
            out.append(f"mean word length {mwl:.2f} out of range")
        if _stopword_ratio(text) < self.stopword_min:
            out.append("stopword ratio too low (non-prose / gibberish)")
        return out

    def quarantine(self, text):
        """True => the incumbent would DROP this chunk as low-quality."""
        return len(self.reasons(text)) > 0
