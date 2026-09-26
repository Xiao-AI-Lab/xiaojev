"""Unit tests for rag_eval/iterative_rag.py — all model-touching steps are
fakes; no GPU, no network, no 8020."""
import pytest

from rag_eval.iterative_rag import (ARMS, ArmConfig, IterativeRAG,
                                    build_subquery_prompt, calibrate_tau_iter,
                                    decide)


def passages(prefix, n, start=0):
    return [{"docid": f"{prefix}{i}", "title": f"t{i}", "text": f"text {i}"}
            for i in range(start, start + n)]


class Fakes:
    """Recording fakes for the five injected callables."""

    def __init__(self, gate_probs=(), pool_per_query=None):
        self.gate_probs = list(gate_probs)
        self.pool_per_query = pool_per_query or {}
        self.calls = {"retrieve": [], "rerank": [], "gate": [], "read": [],
                      "propose": []}

    def retrieve(self, query, k):
        self.calls["retrieve"].append(query)
        return self.pool_per_query.get(query, passages("d", 10))

    def rerank(self, question, cands):
        self.calls["rerank"].append([p["docid"] for p in cands])
        return cands  # identity: dense order is already the fused order

    def gate(self, question, evidence):
        self.calls["gate"].append([p["docid"] for p in evidence])
        return self.gate_probs[len(self.calls["gate"]) - 1]

    def read(self, question, evidence):
        self.calls["read"].append([p["docid"] for p in evidence])
        return "the answer", {"prompt_tokens": 100, "completion_tokens": 5}

    def propose(self, question, evidence, round_):
        self.calls["propose"].append((round_, [p["docid"] for p in evidence]))
        return f"subquery after round {round_}", {"prompt_tokens": 50,
                                                  "completion_tokens": 10}


def make(fakes, arm, tau=None):
    return IterativeRAG(fakes.retrieve, fakes.rerank, gate=fakes.gate,
                        read=fakes.read, propose=fakes.propose,
                        config=arm, tau=tau)


def test_single_arm_one_round_no_gate_no_propose():
    f = Fakes()
    out = make(f, ARMS["single"]).run("q?")
    assert out["stop"] == "ungated" and out["n_rounds"] == 1
    assert out["evidence_docids"] == [f"d{i}" for i in range(5)]
    assert f.calls["retrieve"] == ["q?"]
    assert f.calls["read"] == [out["evidence_docids"]]
    assert f.calls["gate"] == [] and f.calls["propose"] == []
    assert out["gate_probs"] == [None]


def test_fixed2_two_rounds_subquery_drives_second_retrieval():
    f = Fakes(pool_per_query={"subquery after round 1": passages("e", 10)})
    out = make(f, ARMS["fixed2"]).run("q?")
    assert out["stop"] == "ungated" and out["n_rounds"] == 2
    assert f.calls["retrieve"] == ["q?", "subquery after round 1"]
    assert f.calls["propose"] == [(1, [f"d{i}" for i in range(5)])]
    assert out["evidence_docids"] == [f"d{i}" for i in range(5)] + [f"e{i}" for i in range(5)]
    assert len(f.calls["read"]) == 1 and len(f.calls["read"][0]) == 10


def test_gated_accept_round2_stops_before_round3():
    f = Fakes(gate_probs=[0.30, 0.90, 0.99],
              pool_per_query={"subquery after round 1": passages("e", 10)})
    out = make(f, ARMS["gated"], tau=0.65).run("q?")
    assert out["stop"] == "gate_accept" and out["n_rounds"] == 2
    assert out["gate_probs"] == [0.30, 0.90]
    assert len(f.calls["retrieve"]) == 2  # no round-3 retrieval
    assert f.calls["propose"] == [(1, [f"d{i}" for i in range(5)])]
    assert f.calls["gate"] == [[f"d{i}" for i in range(5)],
                               [f"d{i}" for i in range(5)] + [f"e{i}" for i in range(5)]]
    assert len(f.calls["read"][0]) == 10  # reader gets accumulated evidence


def test_gated_accept_round1_reads_round1_evidence_only():
    f = Fakes(gate_probs=[0.80])
    out = make(f, ARMS["gated"], tau=0.65).run("q?")
    assert out["stop"] == "gate_accept" and out["n_rounds"] == 1
    assert f.calls["propose"] == []
    assert f.calls["read"] == [[f"d{i}" for i in range(5)]]


def test_gated_exhaust_answer_fallback():
    f = Fakes(gate_probs=[0.1, 0.2, 0.3],
              pool_per_query={"subquery after round 1": passages("e", 10),
                              "subquery after round 2": passages("f", 10)})
    out = make(f, ARMS["gated"], tau=0.65).run("q?")
    assert out["stop"] == "exhaust_answer" and out["n_rounds"] == 3
    assert len(f.calls["read"]) == 1 and len(f.calls["read"][0]) == 15
    assert out["answer"] == "the answer"
    assert [r for r, _ in f.calls["propose"]] == [1, 2]


def test_gated_refuse_exhaust_never_calls_reader():
    f = Fakes(gate_probs=[0.1, 0.2, 0.3])
    out = make(f, ARMS["gated_refuse"], tau=0.65).run("q?")
    assert out["stop"] == "exhaust_refuse" and out["n_rounds"] == 3
    assert out["answer"] is None and f.calls["read"] == []
    assert len(f.calls["gate"]) == 3


def test_gated_refuse_can_still_accept_early():
    f = Fakes(gate_probs=[0.9])
    out = make(f, ARMS["gated_refuse"], tau=0.65).run("q?")
    assert out["stop"] == "gate_accept" and out["answer"] == "the answer"


def test_evidence_dedup_across_rounds():
    # every query returns the same pool: round 2 must skip round-1 docids
    f = Fakes(gate_probs=[0.1, 0.9])
    out = make(f, ARMS["gated"], tau=0.65).run("q?")
    assert out["evidence_docids"] == [f"d{i}" for i in range(10)]
    assert f.calls["rerank"][1] == [f"d{i}" for i in range(5, 10)]


def test_exclude_docids_banned_from_every_round():
    f = Fakes(gate_probs=[0.9])
    out = make(f, ARMS["gated"], tau=0.65).run("q?", exclude_docids=("d0", "d1"))
    assert out["evidence_docids"] == [f"d{i}" for i in range(2, 7)]
    assert "d0" not in f.calls["rerank"][0] and "d1" not in f.calls["rerank"][0]


def test_rerank_ordering_respected():
    f = Fakes()
    rev = lambda q, cands: list(reversed(cands))
    rag = IterativeRAG(f.retrieve, rev, gate=f.gate, read=f.read,
                       propose=f.propose, config=ARMS["gated"], tau=0.5)
    f.gate_probs = [0.9]
    out = rag.run("q?")
    assert out["evidence_docids"] == [f"d{i}" for i in range(9, 4, -1)]


def test_usage_and_latency_recorded():
    f = Fakes(gate_probs=[0.1, 0.9])
    out = make(f, ARMS["gated"], tau=0.65).run("q?")
    assert out["reader_usage"] == {"prompt_tokens": 100, "completion_tokens": 5}
    assert out["proposer_usage"] == [{"round": 1, "subquery": "subquery after round 1",
                                      "prompt_tokens": 50, "completion_tokens": 10}]
    assert out["latency_s"] >= 0


def test_interface_validation():
    f = Fakes()
    with pytest.raises(ValueError, match="gate"):
        IterativeRAG(f.retrieve, f.rerank, gate=None, read=f.read,
                     propose=f.propose, config=ARMS["gated"], tau=0.5)
    with pytest.raises(ValueError, match="tau"):
        IterativeRAG(f.retrieve, f.rerank, gate=f.gate, read=f.read,
                     propose=f.propose, config=ARMS["gated"], tau=None)
    with pytest.raises(ValueError, match="propose"):
        IterativeRAG(f.retrieve, f.rerank, read=f.read, propose=None,
                     config=ARMS["fixed2"])
    with pytest.raises(ValueError, match="read"):
        IterativeRAG(f.retrieve, f.rerank, read=None, config=ARMS["single"])
    # gated_refuse without read is legal: it may only refuse
    IterativeRAG(f.retrieve, f.rerank, gate=f.gate, read=None,
                 propose=f.propose, config=ARMS["gated_refuse"], tau=0.5)


def test_decide_matches_pipeline_trajectories():
    for probs, arm, tau, want in [
            ([0.9, 0.1, 0.1], ARMS["gated"], 0.65, (1, "gate_accept")),
            ([0.1, 0.7, 0.1], ARMS["gated"], 0.65, (2, "gate_accept")),
            ([0.1, 0.2, 0.3], ARMS["gated"], 0.65, (3, "exhaust_answer")),
            ([0.1, 0.2, 0.3], ARMS["gated_refuse"], 0.65, (3, "exhaust_refuse")),
            ([0.9], ARMS["single"], 0.65, (1, "ungated")),
            ([0.9, 0.9], ARMS["fixed2"], 0.65, (2, "ungated")),
    ]:
        assert decide(probs, arm, tau) == want
    # end-to-end agreement: decide() on recorded probs == pipeline outcome
    for probs, arm in [([0.3, 0.9, 0.1], ARMS["gated"]),
                       ([0.1, 0.2, 0.3], ARMS["gated_refuse"]),
                       ([], ARMS["fixed2"])]:
        f = Fakes(gate_probs=probs)
        out = make(f, arm, tau=0.65 if arm.gated else None).run("q?")
        n, stop = decide(probs, arm, 0.65)
        assert (out["n_rounds"], out["stop"]) == (n, stop)


def test_calibrate_tau_iter_precision_target_then_coverage():
    records = [
        {"qid": "a1", "variant": "ans", "probs": [0.90]},
        {"qid": "a2", "variant": "ans", "probs": [0.20]},
        {"qid": "u1", "variant": "unans", "probs": [0.10]},
        {"qid": "u2", "variant": "unans", "probs": [0.80]},
    ]
    tau, curve = calibrate_tau_iter(records, target_precision=0.90)
    # tau<=0.80 accepts u2 -> precision <= 2/3 < 0.90; tau in (0.80, 0.90]
    # accepts only a1 -> precision 1.0, coverage 0.5
    assert tau == 0.81
    chosen = next(c for c in curve if c["tau"] == tau)
    assert chosen["precision"] == 1.0 and chosen["coverage"] == 0.5
    assert len(curve) == 99


def test_calibrate_tau_iter_multiround_probs():
    records = [
        {"qid": "a1", "variant": "ans", "probs": [0.1, 0.1, 0.9]},
        {"qid": "u1", "variant": "unans", "probs": [0.1, 0.2, 0.3]},
    ]
    tau, _ = calibrate_tau_iter(records, target_precision=0.90)
    # any tau in (0.3, 0.9] rejects u1 (max 0.3) and accepts a1; the rule
    # picks max coverage among feasible taus, i.e. the lowest: 0.31
    assert tau == 0.31


def test_calibrate_tau_iter_target_unreachable_fallback():
    records = [
        {"qid": "a1", "variant": "ans", "probs": [0.50]},
        {"qid": "u1", "variant": "unans", "probs": [0.90]},
    ]
    tau, curve = calibrate_tau_iter(records, target_precision=0.99)
    # no tau reaches precision 0.99 with any acceptance; fallback = max
    # precision then max coverage -> tau <= 0.50 accepts both (prec 0.5)
    feas = [c for c in curve if c["precision"] is not None]
    best = max(feas, key=lambda c: (c["precision"], c["coverage"]))
    assert tau == best["tau"]


def test_build_subquery_prompt_carries_question_evidence_and_missing_info():
    prompt = build_subquery_prompt("Who wrote X?", passages("d", 2))
    assert "Who wrote X?" in prompt and "[t0]" in prompt and "[t1]" in prompt
    assert "still missing" in prompt and "subquery" in prompt


def test_arm_configs_match_spec():
    assert ARMS["single"].max_rounds == 1 and not ARMS["single"].gated
    assert ARMS["fixed2"].max_rounds == 2 and not ARMS["fixed2"].gated
    assert ARMS["gated"].max_rounds == 3 and ARMS["gated"].gated
    assert ARMS["gated"].on_exhaust == "answer"
    assert ARMS["gated_refuse"].on_exhaust == "refuse"
