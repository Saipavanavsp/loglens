"""How sure is the local model about its answer?

Two methods, chosen in configs/gateway.yaml:

1. "logprobs" (default, one model call). The decision tokens are the first
   tokens of the severity and category values in the JSON. Their probability
   is how strongly the model preferred that class over the others at the
   moment it committed. Confidence = min(p(severity), p(category)), because one
   uncertain field is enough to make the answer risky.
   Cost: free. Caveat: token probabilities of fine-tuned models are often
   over-confident, so the threshold has to be tuned on the validation set,
   not guessed.

2. "self_consistency" (N extra samples at temperature > 0). Confidence = the
   fraction of samples that agree with the greedy answer on both fields.
   Better calibrated in practice, but N times the compute on a CPU.

If neither signal is available, confidence is None and the router treats that
as "unknown" (configurable: trust it or fall back).
"""

import math
import re
from collections.abc import Sequence

from loglens.gateway.providers.base import TokenLogprob
from loglens.schema import TriageLabel

_FIELD_VALUE = re.compile(r'"(severity|category)"\s*:\s*"')


def logprob_confidence(text: str, logprobs: Sequence[TokenLogprob] | None) -> float | None:
    """min over {severity, category} of the probability of the value's first token."""
    if not logprobs:
        return None
    # Rebuild character offsets of each token in the generated text.
    joined = "".join(t.token for t in logprobs)
    if joined != text:
        # Tokens don't reconstruct the text (e.g. stripped whitespace); search in the join.
        text = joined
    starts, pos = [], 0
    for t in logprobs:
        starts.append(pos)
        pos += len(t.token)
    probs: dict[str, float] = {}
    for m in _FIELD_VALUE.finditer(text):
        value_start = m.end()
        # Token whose span contains the first character of the value.
        idx = next(
            (
                i
                for i, s in enumerate(starts)
                if s <= value_start < s + max(1, len(logprobs[i].token))
            ),
            None,
        )
        if idx is not None and m.group(1) not in probs:
            # If the token started before the value (e.g. '"error'), it still carries the decision.
            probs[m.group(1)] = math.exp(logprobs[idx].logprob)
    if len(probs) < 2:
        return None
    return min(probs.values())


def agreement_confidence(greedy: TriageLabel, samples: Sequence[TriageLabel | None]) -> float:
    """Fraction of samples matching the greedy answer on severity and category."""
    if not samples:
        return 1.0
    same = sum(
        1
        for s in samples
        if s is not None and s.severity == greedy.severity and s.category == greedy.category
    )
    return same / len(samples)
