#!/usr/bin/env python3
"""Export small full-precision S06 seed-13 report tables from authoritative files."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
from pathlib import Path
import shutil


CONDITIONS=["M","G","V","Local_no_extra_risk","Local_risk",
            "Context_no_extra_risk","Context_risk"]
FIELDS=["run_id","condition","seed","scope_id","sequence","status","num_frames",
        "HOTA","DetA","AssA","IDF1","MOTA","IDSW","FP","FN","Frag",
        "source_artifact_id","source_locator","missing_reason"]


def frames(root:Path,seq:str)->int:
    c=configparser.ConfigParser(); c.read(root/seq/"seqinfo.ini")
    return c["Sequence"].getint("seqLength")


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--evaluation-root",type=Path,required=True)
    p.add_argument("--dataset-root",type=Path,required=True)
    p.add_argument("--training-curves",type=Path,required=True)
    p.add_argument("--selection",type=Path,required=True)
    p.add_argument("--tracking-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    s=json.loads(a.summary.read_text()); seqs=sorted(s["results"]["M"]["per_sequence"])
    count={q:frames(a.dataset_root,q) for q in seqs}; agg=[]; per=[]
    for condition in CONDITIONS:
        official=a.evaluation_root/condition/"official"/condition/"pedestrian_detailed.csv"
        with official.open(newline="") as h: detail={r["seq"]:r for r in csv.DictReader(h)}
        for sequence in ["COMBINED_SEQ",*seqs]:
            key="COMBINED" if sequence=="COMBINED_SEQ" else sequence
            metric=s["results"][condition]["COMBINED_SEQ" if sequence=="COMBINED_SEQ" else "per_sequence"]
            if sequence!="COMBINED_SEQ": metric=metric[sequence]
            d=detail[key]
            row={"run_id":f"s06_seed13_{condition}","condition":condition,"seed":13,
                 "scope_id":"evaluation8_reused","sequence":sequence,"status":"COMPLETED",
                 "num_frames":sum(count.values()) if sequence=="COMBINED_SEQ" else count[sequence],
                 "HOTA":repr(float(metric["HOTA"])),"DetA":repr(float(metric["DetA"])),
                 "AssA":repr(float(metric["AssA"])),"IDF1":repr(float(metric["IDF1"])),
                 "MOTA":repr(float(metric["MOTA"])),"IDSW":int(metric["IDSW"]),
                 "FP":int(float(d["CLR_FP"])),"FN":int(float(d["CLR_FN"])),
                 "Frag":int(metric["Frag"]),"source_artifact_id":"s06_official_trackeval",
                 "source_locator":f"{condition}/summary.json:{sequence}; official/{condition}/pedestrian_detailed.csv:{key}",
                 "missing_reason":""}
            (agg if sequence=="COMBINED_SEQ" else per).append(row)
    a.output_dir.mkdir(parents=True)
    for name,rows in (("aggregate_metrics.csv",agg),("per_sequence_metrics.csv",per)):
        with (a.output_dir/name).open("w",newline="") as h:
            w=csv.DictWriter(h,fieldnames=FIELDS,lineterminator="\n"); w.writeheader(); w.writerows(rows)
    with a.training_curves.open(newline="") as src, (a.output_dir/"learning_curves.csv").open("w",newline="") as dst:
        old=csv.DictReader(src); fields=["run_id","seed","epoch","step","split","metric","value","unit",
                                             "samples_seen","elapsed_seconds","checkpoint_id","missing_reason"]
        w=csv.DictWriter(dst,fieldnames=fields,lineterminator="\n"); w.writeheader()
        for r in old: w.writerow({"run_id":f"s06_seed13_{r['model']}","seed":r["seed"],
            "epoch":r["epoch"],"step":"","split":r["split"],"metric":r["metric"],
            "value":r["value"],"unit":"MSE","samples_seen":r["samples_seen"],
            "elapsed_seconds":r["elapsed_seconds"],"checkpoint_id":r["checkpoint"],"missing_reason":""})
    shutil.copy2(a.summary,a.output_dir/"seed13_gate.json")
    shutil.copy2(a.selection,a.output_dir/"checkpoint_selection.json")
    changed={}
    for condition in CONDITIONS[3:]:
        rows=[]
        for pth in sorted((a.tracking_root/condition).glob("dancetrack*/engineering_checks.json")):
            r=json.loads(pth.read_text()); rows.append({"sequence":pth.parent.name,
                "components":r["selector_components"],"changed_components":r["selector_changed_components"],
                "wall_seconds":r["result"]["wall_seconds"],"cpu_core_hours":r["result"]["cpu_core_hours"]})
        changed[condition]=rows
    (a.output_dir/"diagnostics.json").write_text(json.dumps({"schema_version":"mot-report-v1",
        "status":"COMPLETED","comparison_type":"natural_closed_loop_component_selector",
        "unit_of_analysis":"component_frame","per_condition_sequence":changed,
        "limitations":["reused development sequences","owner-proxy training labels","single seed"]},indent=2)+"\n")
    print(json.dumps({"status":"COMPLETED","aggregate_rows":len(agg),
                      "per_sequence_rows":len(per)},sort_keys=True))


if __name__=="__main__": main()
