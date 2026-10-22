"""Group log messages into event templates with the Drain algorithm.

Drain (He et al., ICWS 2017) is the standard online log parser. This is a
compact re-implementation of its core so the project doesn't depend on the
drain3 package (whose strict dependency pins clash with MLflow).

How it works, for each message:
1. Split into tokens. Messages with different token counts never share a template.
2. Walk a fixed-depth tree keyed by the first few tokens (tokens with digits
   go to a <*> branch so variable prefixes don't explode the tree).
3. In that leaf, find the existing cluster whose template has the highest
   fraction of positions equal to the message (<*> matches anything).
4. If the best similarity >= sim_threshold, merge: positions that differ become
   <*>. Otherwise start a new cluster.

Example: "invalid user tianf [preauth]" and "invalid user xbmc [preauth]"
become one template "invalid user <*> [preauth]".
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

WILDCARD = "<*>"


@dataclass
class _Cluster:
    tokens: list[str]
    keys: list[str] = field(default_factory=list)

    @property
    def template(self) -> str:
        return " ".join(self.tokens)


class Drain:
    def __init__(self, depth: int = 4, sim_threshold: float = 0.5, max_children: int = 100):
        self.prefix_len = max(1, depth - 2)  # depth counts the root and length layers
        self.sim_threshold = sim_threshold
        self.max_children = max_children
        self._tree: dict[int, dict] = {}

    def _leaf(self, tokens: list[str]) -> list[_Cluster]:
        node = self._tree.setdefault(len(tokens), {})
        for tok in tokens[: self.prefix_len]:
            key = WILDCARD if any(c.isdigit() for c in tok) or tok.startswith("<") else tok
            if key not in node and len(node) >= self.max_children:
                key = WILDCARD  # cap fan-out, as in the paper
            node = node.setdefault(key, {})
        return node.setdefault("__clusters__", [])

    @staticmethod
    def _similarity(template: list[str], tokens: list[str]) -> float:
        same = sum(1 for a, b in zip(template, tokens, strict=True) if a in (b, WILDCARD))
        return same / len(tokens) if tokens else 1.0

    def add(self, message: str) -> _Cluster:
        tokens = message.split()
        clusters = self._leaf(tokens)
        best, best_sim = None, -1.0
        for c in clusters:
            sim = self._similarity(c.tokens, tokens)
            if sim > best_sim:
                best, best_sim = c, sim
        if best is not None and best_sim >= self.sim_threshold:
            best.tokens = [
                a if a == b else WILDCARD for a, b in zip(best.tokens, tokens, strict=True)
            ]
        else:
            best = _Cluster(list(tokens))
            clusters.append(best)
        best.keys.append(message)
        return best


def mine_templates(
    keys_by_source: dict[str, Iterable[str]], sim_threshold: float = 0.5
) -> dict[tuple[str, str], str]:
    """Map (source, masked key) -> "source:template". Templates are mined per source."""
    out: dict[tuple[str, str], str] = {}
    for source, keys in sorted(keys_by_source.items()):
        drain = Drain(sim_threshold=sim_threshold)
        clusters = {}
        for key in dict.fromkeys(keys):  # unique, first-seen order
            clusters[key] = drain.add(key)
        # Read templates after all merges so early keys get the final template.
        for key, cluster in clusters.items():
            out[(source, key)] = f"{source}:{cluster.template}"
    return out
