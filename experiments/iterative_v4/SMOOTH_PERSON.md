# Smooth-axis person follow-up

User requested coarse head/body or smooth-axis person. Inspected private reference head_body_proxy: two ellipsoids in common coordinates, independent head/body rotations; no skeleton required. Chose the user-permitted smooth-axis alternative, implemented independently without copying private source.

Candidate retains shared tube dimensions, removes constant/linear local bend, uses two spatial modes and at most seven temporal control states, natural cubic interpolation to 17 surface rings. Renderer and differentiable surface samples share the same smooth geometry; approximate surface losses still apply. Does not add head/limbs.

Step20/50 use their own data, same initial person and 600 optimization steps. Comparison:
- Step20 mask IoU .6269 -> .6208; capped foreground depth error .3736 -> .3737.
- Step50 mask IoU .6187 -> .6244; capped foreground depth error .3800 -> .3679.
- Bend second-difference energy drops about 19x/18x; this is a parameter smoothness metric, not validated physical motion.
- Root second-difference worsens at Step20 (5.08e-5 -> 8.85e-5), improves at Step50 (3.48e-5 -> 1.97e-5). No blanket temporal stability claim.
- Visual review shows reduced pipe corners, but foot contact and overall lean remain imperfect.
- 30 tests pass, including smooth NumPy/Torch geometry equality.

Agent implementation audit:
Current-chat Codex reviews and writes decisions. No external VLM and no automatic autonomous closed loop. Tools generate diagnostics; agent explicitly chooses revisions and runs tools.
Priority provides role/duty labels, retention, representation gates, part caps and per-entity step caps. It does not automatically synthesize parts or allocate work adaptively. Person consumed 600-step allocation; background had budgets but was initialized/fitted with existing surface tools rather than consuming those step allocations. Court equal-budget ablations precede priority plan, so no causal claim that priority improved court results.
Relations in policy do not automatically create physical constraints. Only existing scene/fitter-supported constraints take effect.
