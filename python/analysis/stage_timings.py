"""Four nonoverlapping elapsed-time scopes for a single party and MPC lane."""
import math
import re


def four_stage_timings(rows, lanes):
    if lanes != 1:
        raise ValueError("four-stage elapsed accounting requires one MPC lane; concurrent leaf timers cannot be subtracted")

    def duration(stage):
        values = [float(r["duration_seconds"]) for r in rows if r["stage"] == stage]
        if not values:
            raise ValueError(f"missing P1 metric {stage}; new instrumented run required")
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"invalid metric {stage}")
        return sum(values)

    caps = sorted({int(match[1]) for row in rows
                   if (match := re.fullmatch(r"embedding_M(\d+)_(?:coordinates|finalize)", row["stage"]))})
    if not caps:
        raise ValueError("missing per-cap indexing metrics")
    result = {stage + "_s_p1": duration(stage) for stage in
              ("alignment", "collective_setup", "null_model", "compute_weights", "packed_statistics", "finalize", "release")}
    local_inside = sum(duration(s) for s in
                       ("null_local_equations", "local_gtx", "local_gtg", "local_gty", "local_pvalues"))
    genotype_io = duration("genotype_loading")
    secure = sum(result[s + "_s_p1"] for s in
                 ("null_model", "compute_weights", "packed_statistics", "finalize", "release")) - local_inside - genotype_io
    if secure < 0:
        raise ValueError("local/I/O times exceed Stage 3 parent scopes")
    result.update(
        stage1_s_p1=duration("embedding_private_cap"),
        stage2_s_p1=sum(duration(f"embedding_M{cap}_{phase}") for cap in caps for phase in ("coordinates", "finalize")),
        stage3_local_s_p1=local_inside + duration("local_dosage_sums"),
        stage3_secure_s_p1=secure,
        stage3_input_loading_s_p1=genotype_io + duration("dosage_loading"),
        he_setup_s_p1=duration("embedding_keygen") + result["collective_setup_s_p1"],
    )
    return result


STAGE_SERIES = [
    ("stage1_s_p1", "Stage 1: Counting", "#009e73"),
    ("stage2_s_p1", "Stage 2: Indexing", "#0072b2"),
    ("stage3_local_s_p1", "Stage 3-1: Local", "#d55e00"),
    ("stage3_secure_s_p1", "Stage 3-2: Secure", "#cc79a7"),
]
