#!/usr/bin/env python3
"""Apply the frozen S06 seed-13 evaluation8 progression gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CONDITIONS = ["M", "G", "V", "Local_no_extra_risk", "Local_risk",
              "Context_no_extra_risk", "Context_risk"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--evaluation-root",type=Path,required=True)
    p.add_argument("--staging-manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    results={}; sources={}
    for condition in CONDITIONS:
        path=args.evaluation_root/condition/"summary.json"; row=json.loads(path.read_text())
        if row["execution_status"]!="PASS" or row["sequence_count"]!=8:
            raise RuntimeError(f"incomplete {condition}")
        results[condition]={"COMBINED_SEQ":row["official_trackeval_combined"],
                            "per_sequence":row["per_sequence"]}
        sources[condition]={"path":str(path),"sha256":sha(path)}
    c=results["Context_risk"]; m=results["M"]; v=results["V"]; l=results["Local_risk"]
    def delta(a,b,metric): return results[a]["COMBINED_SEQ"][metric]-results[b]["COMBINED_SEQ"][metric]
    wins=sum(c["per_sequence"][seq]["HOTA"]>m["per_sequence"][seq]["HOTA"]
             for seq in c["per_sequence"])
    clauses={
        "Context_risk_minus_M_HOTA_at_least_0.5":delta("Context_risk","M","HOTA")>=.5,
        "Context_risk_minus_M_AssA_at_least_0.5":delta("Context_risk","M","AssA")>=.5,
        "Context_risk_minus_M_IDSW_not_increased":delta("Context_risk","M","IDSW")<=0,
        "Context_risk_positive_HOTA_videos_at_least_5":wins>=5,
        "Context_risk_beats_V_HOTA":delta("Context_risk","V","HOTA")>0,
        "Context_risk_beats_V_AssA":delta("Context_risk","V","AssA")>0,
        "Context_risk_beats_Local_risk_HOTA":delta("Context_risk","Local_risk","HOTA")>0,
        "Context_risk_beats_Local_risk_AssA":delta("Context_risk","Local_risk","AssA")>0,
    }
    summary={"status":"COMPLETED","seed":13,
        "decision":"AUTHORIZE_SEEDS_37_73" if all(clauses.values()) else "STOP_S06_NO_SEEDS_37_73",
        "gate_passed":all(clauses.values()),"gate_clauses":clauses,
        "Context_risk_positive_HOTA_videos":wins,
        "deltas":{f"Context_risk_minus_{base}":{metric:delta("Context_risk",base,metric)
                  for metric in ("HOTA","AssA","IDF1","IDSW","MOTA")}
                  for base in ("M","V","Local_risk")},
        "results":results,"source_summaries":sources,
        "staging_manifest":str(args.staging_manifest),
        "staging_manifest_sha256":sha(args.staging_manifest),
        "interpretation":"single_seed_reused_development_progression_gate_not_significance_or_validation"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:summary[k] for k in ("decision","gate_passed","gate_clauses","deltas")},indent=2))


if __name__=="__main__": main()
