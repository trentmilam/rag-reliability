"""Incumbent baselines Headroom is measured against.

The novelty claim ("no existing hop-governor gates on measured hardware
physics") is only credible if it is proved against a REAL incumbent, not
against "no governor at all." This module ships that incumbent:

    CostGovernor: a token-cost + answer-sufficiency governor of the
    Adaptive-RAG / CA-RAG family. It escalates another retrieval/reasoning hop
    while (a) the answer is judged insufficient AND (b) the token budget still
    covers the next hop, exactly the software-economics gate those systems use.

It is blind to VRAM / thermal telemetry. That is the incumbent's actual design
(these methods predate hardware-aware gating), not a handicap imposed here, so
the comparison is fair, and the breach it suffers on a headroom-constrained
card is exactly the gap Headroom fills.

Fair-baseline discipline:
  * ONE fixed configuration is used for every scenario in eval.py.
  * TOKEN_BUDGET is a common context-window size, not tuned to lose.
  * TOKENS_PER_HOP is DERIVED from the real corpus (retrieved context tokens +
    a realistic generation cost), not a magic number.
  * The governor demonstrably STOPS when its own signals bind (eval.py's
    _fairness_gate proves sufficiency- and budget-triggered stops), so it is a
    working comparator, not a strawman no-op.
"""

from __future__ import annotations

from headroom import Decision, GpuState
from loop import _CORPUS

# --- derived per-hop token cost ----------------------------------------------
# A hop retrieves a chunk of context and generates a paragraph of reasoning.
# Context cost is DERIVED from the real corpus; generation is a realistic fixed
# budget. (~1.3 tokens/word is the usual English rule of thumb.)
_GEN_TOKENS = 400  # a hop's reasoning/answer draft


def _chunk_tokens(text: str) -> int:
    return int(round(len(text.split()) * 1.3))


TOKENS_PER_HOP = round(sum(_chunk_tokens(c) for c in _CORPUS) / len(_CORPUS)) + _GEN_TOKENS
# A common context-window budget (e.g. an 8k-context model). Chosen from answer
# economics, NOT from hardware: the whole point is that it does not know about
# the card.
TOKEN_BUDGET = 8192


class CostGovernor:
    """Token-cost + sufficiency hop-governor (Adaptive-RAG / CA-RAG family).

    Interface-compatible with `Headroom` so `loop.run_loop` can drive it: it
    consumes the same `GpuState` telemetry stream via `observe()` and answers
    `gate()` with ALLOW / DENY. It reads ONLY the hop index from the state (to
    accrue token cost) and its own sufficiency signal; it never looks at
    `vram_used_mb` or `temp_c`, because a cost governor structurally cannot.
    """

    def __init__(self, *, token_budget: float, tokens_per_hop: float,
                 sufficient_after: int | None = None):
        self.token_budget = float(token_budget)
        self.tokens_per_hop = float(tokens_per_hop)
        # None => the query is never satisfied within the run (a hard multi-hop
        # question); an int => the answer is good enough after that many hops.
        self.sufficient_after = sufficient_after
        self._hops_done = 0
        self._tokens = 0.0
        self.log: list[tuple[Decision, str]] = []

    def observe(self, state: GpuState) -> None:
        # A completed escalation hop spends tokens. Hop 0 is the baseline load.
        if state.hop >= 1:
            self._hops_done = state.hop
            self._tokens = state.hop * self.tokens_per_hop

    def gate(self) -> Decision:
        """Permit one more hop on SOFTWARE ECONOMICS alone."""
        # (a) sufficiency: don't pay for a hop we don't need.
        if self.sufficient_after is not None and self._hops_done >= self.sufficient_after:
            return self._log(Decision.DENY,
                             f"answer sufficient after {self._hops_done} hops")
        # (b) token budget: would the next hop blow the context/cost budget?
        projected = self._tokens + self.tokens_per_hop
        if projected > self.token_budget:
            return self._log(Decision.DENY,
                             f"token budget: {projected:.0f} > {self.token_budget:.0f}")
        # otherwise the economics say "keep going": it has no hardware signal
        # that would say otherwise.
        return self._log(Decision.ALLOW,
                         f"within budget ({projected:.0f}/{self.token_budget:.0f} tok), "
                         f"answer insufficient")

    def _log(self, dec: Decision, reason: str) -> Decision:
        self.log.append((dec, reason))
        return dec
