"""Spend guard used by every script that calls a paid API.

Flow: the script works out how many requests it would make and what that will
roughly cost, then calls `confirm_spend`. Nothing is sent unless the run is not
a dry run, the request count is within the cap, and the user agrees (or passed
--yes).
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class SpendPlan:
    model: str
    n_requests: int
    est_input_tokens: int
    est_output_tokens: int
    est_cost_usd: float


class SpendCapError(RuntimeError):
    pass


def confirm_spend(
    plan: SpendPlan,
    *,
    max_requests: int,
    dry_run: bool,
    assume_yes: bool = False,
    ask: Callable[[str], str] = input,
    out=sys.stdout,
) -> bool:
    """Print the plan and decide whether to proceed. Returns True to go ahead."""
    print(
        f"Plan: {plan.n_requests} request(s) to {plan.model}\n"
        f"  estimated tokens: ~{plan.est_input_tokens:,} in / ~{plan.est_output_tokens:,} out\n"
        f"  estimated cost:   ~${plan.est_cost_usd:.4f} (rough; actual usage is reported after)",
        file=out,
    )
    if plan.n_requests > max_requests:
        raise SpendCapError(
            f"{plan.n_requests} requests exceeds the cap of {max_requests}. "
            "Raise --max-requests (or LOGLENS_MAX_REQUESTS) if this is intended."
        )
    if dry_run:
        print("Dry run: no requests sent. Re-run with --no-dry-run to execute.", file=out)
        return False
    if plan.n_requests == 0:
        return False
    if assume_yes:
        return True
    answer = ask("Proceed? [y/N] ").strip().lower()
    return answer in {"y", "yes"}
