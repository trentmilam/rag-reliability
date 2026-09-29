"""GraphRx: a GraphRAG structural linter with retrieval-poisoning scoring.

Pure numpy + stdlib. Deterministic. Offline. No Neo4j.
"""
from .lint import report, SEP_THRESH  # noqa: F401
from .probe import probe_local, probe_community, poisoning_delta, apply_proposal  # noqa: F401
from . import fixtures  # noqa: F401
