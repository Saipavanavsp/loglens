"""Group log messages into event templates with Drain (drain3).

Drain builds a parse tree over token positions and merges messages whose
tokens mostly agree, replacing the differing tokens with <*>. For example
"invalid user tianf [preauth]" and "invalid user xbmc [preauth]" become
"invalid user <*> [preauth]".

Templates are mined per source, on the already-masked message keys, and fed in
a fixed order so results are reproducible.
"""

import logging
from collections.abc import Iterable

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

logging.getLogger("drain3").setLevel(logging.WARNING)


def _miner() -> TemplateMiner:
    config = TemplateMinerConfig()
    config.drain_sim_th = 0.5
    config.drain_depth = 4
    config.profiling_enabled = False
    return TemplateMiner(config=config)


def mine_templates(keys_by_source: dict[str, Iterable[str]]) -> dict[tuple[str, str], str]:
    """Map (source, masked key) -> "source:template" string."""
    out: dict[tuple[str, str], str] = {}
    for source, keys in sorted(keys_by_source.items()):
        unique = list(dict.fromkeys(keys))
        miner = _miner()
        for key in unique:
            miner.add_log_message(key)
        for key in unique:
            cluster = miner.match(key, full_search_strategy="always")
            template = cluster.get_template() if cluster else key
            out[(source, key)] = f"{source}:{template}"
    return out
