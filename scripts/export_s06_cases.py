#!/usr/bin/env python3
"""Export reproducible S06 changed decisions and fixed-hash unchanged controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CONDITIONS=["Local_no_extra_risk","Local_risk","Context_no_extra_risk","Context_risk"]


def rank(value:str)->str: return hashlib.sha256(value.encode()).hexdigest()


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--tracking-root",type=Path,required=True)
    p.add_argument("--summary",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); s=json.loads(a.summary.read_text()); records=[]
    for condition in CONDITIONS:
        unchanged=[]
        for trace in sorted((a.tracking_root/condition).glob("dancetrack*/traces/M/*.jsonl")):
            sequence=trace.parent.parent.parent.name
            delta=s["results"][condition]["per_sequence"][sequence]["HOTA"]-s["results"]["M"]["per_sequence"][sequence]["HOTA"]
            for line in trace.read_text().splitlines():
                row=json.loads(line)
                for index,component in enumerate(row.get("selector_components",[])):
                    case={"case_id":f"{condition}:{sequence}:{row['frame']}:{index}",
                          "condition":condition,"sequence":sequence,"frame":row["frame"],
                          "component_index":index,"chosen":component["chosen"],
                          "predictions":component["predictions"],
                          "sequence_HOTA_delta_vs_M":delta,
                          "comparison_type":"descriptive_full_sequence_not_causal_case_effect",
                          "source_trace":str(trace)}
                    if component["chosen"]!="M": records.append(case)
                    else: unchanged.append(case)
        records.extend(sorted(unchanged,key=lambda x:rank(x["case_id"]))[:3])
    if a.output.exists(): raise FileExistsError(a.output)
    with a.output.open("w") as h:
        for row in records: h.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
    print(json.dumps({"status":"COMPLETED","cases":len(records),
                      "changed":sum(r["chosen"]!="M" for r in records),
                      "unchanged_controls":sum(r["chosen"]=="M" for r in records)},sort_keys=True))


if __name__=="__main__": main()
