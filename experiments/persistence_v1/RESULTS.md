
# Frozen validation conclusion
Two predeclared generated prompts x Step20/50, six dynamic tracks, same frozen code/config.
Entities persist for all49 frames; evidence state/inference source kept separate.
Robust C2 curves reduce translation/angular acceleration jitter by median98.4%/98.9%.
This is not sufficient: observed mask IoU mean declines70.3% to58.8%.
Withheld mask-span IoU mean59.6% vs linear interpolation70.3%; only1/6 tracks wins.
Courtyard did not generate intended behind-kiosk occlusion; kiosk SAM failed both stages.
Courtyard Step20 flags22 coarse bound conflicts with parked truck.
No per-case retuning, regeneration or post-fit topology revisions.
Read full records and videos at runs/persistence_v1_report/index.html.
The generic mechanism is implemented; the current frozen policy is not validated as
generally more accurate or physically complete. Next work should integrate projection
evidence with continuous optimization, rather than add test-case-specific corrections.
