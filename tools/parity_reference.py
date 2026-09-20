#!/usr/bin/env python3
"""
DEV-ONLY: generate parity fixtures by running the REAL laya runtime (torch, CPU fp32)
on a fixed question set. Output: tests/fixtures/parity-<name>.json

The TS engine test replays the same inputs through the ONNX stack and compares.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reference"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))  # for _fix_tokenizer_config reuse

from laya.agent import Agent, _fix_tokenizer_config  # noqa: E402

ROUTING_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds, subscription charges",
            "technical": "bugs, outages, system errors, performance problems",
            "sales": "pricing, new contracts, upgrades",
            "other": "everything else",
        },
    },
    "urgent": {"type": "noul", "instructions": "Does this require immediate intervention?"},
    "severity": {
        "type": "score",
        "instructions": "How severe is this issue?",
        "criteria": ["minor annoyance", "degraded experience", "blocking work", "production down"],
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated does the user sound?",
        "criteria": ["calm", "annoyed", "angry", "threatening to cancel"],
    },
}

TICKETS = [
    {
        "from": "user@acme.com",
        "subject": "Duplicate charge on invoice #4411",
        "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
    },
    "Payment gateway reports timeout on charge authorizations. Customers cannot checkout since 09:00 UTC. This is URGENT.",
    {"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"},
    "Der Kunde wurde zweimal belastet und wir werden die Zahlung jetzt prüfen.",
    "Minor: the export button label is misspelled on the settings page. No rush at all, thanks!",
]

TYPED_DECISION_QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": "What action should be taken?",
        "criteria": ["escalate", "auto_resolve", "monitor", "request_info"],
    },
    "needs_review": {"type": "noul", "instructions": "Does this need human review?"},
    "urgency": {
        "type": "score",
        "instructions": "Rate the urgency.",
        "criteria": ["low", "medium", "high", "critical"],
    },
}


def main():
    out_dir = os.path.join(HERE, "..", "tests", "fixtures")
    os.makedirs(out_dir, exist_ok=True)

    plans = [
        ("english", "convaiinnovations/laya", None, TICKETS[:2] + [TICKETS[4]], ROUTING_QUESTIONS),
        ("multilingual", "convaiinnovations/laya", "multilingual", TICKETS, ROUTING_QUESTIONS),
        ("typed-decisions", "convaiinnovations/laya", "typed-decisions", TICKETS[:2], TYPED_DECISION_QUESTIONS),
    ]

    for name, repo, subfolder, states, questions in plans:
        agent = Agent(repo, device="cpu", subfolder=subfolder)
        cases = []
        for state in states:
            res = agent.system_one(state, questions)
            cases.append({"state": state, "questions": questions, "expected": res})
        out_path = os.path.join(out_dir, f"parity-{name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": name, "cases": cases}, f, ensure_ascii=False, indent=1)
        print(f"[ok] {out_path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
