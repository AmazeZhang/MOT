# G2 D1 v1 failed operationalization

Scope: train-only, 165 resolved G2 events from four selected difficult sequences. This run is retained for transparency but is **not** used as evidence about history pollution.

V1 kept the baseline first-round assignment unchanged and relabelled the track receiving the target-positive detection with the GT owner of its most recent mapped observation in the previous 30 frames. That definition inverted the intended intervention: during a detector-gap event, a target track can transiently absorb another person's detection, and v1 then treated that contaminating observation as the track's true identity.

The frozen v1 run produced 0 recoveries among 53 identity-change failures and labelled 92/112 success controls as harmed. This control failure invalidated the operationalization. The raw run remains at `/media/imc/data/mot-data/runs/g2_d1_history_correspondence_20261111/`; it was not deleted or silently replaced.

V2 was separately frozen only after this failure was observed. Therefore v2 is explicitly exploratory and not an independent preregistered confirmation.
