# Step20 paired appearance/type replacement

User requested both variants rerun from Step20. No Step50 observations supervise this experiment.

Input scene: runs/v4_courtyard_r1/step20_visible_soft_containment_free/scene.json. Existing Step20 VGGT, masks and fitting are reused. The same role-aware colored geometry and lateral camera intervention are rendered for both variants. Shared video SHA-256: 7616f4b5b386be1a4622d8d283f714dc0fe44d1f74fe6310889695f5810a8234.

A: yellow forklift / blue pickup, gray metal kiosk, brown wall and pale gray ground. Appearance subjects are isolated using exclusive Step20 masks, independently centered/normalized; material samples are strict interior pixels. No RGB full frame/video or annotations submitted.

B: green vintage toy coupe / white box-bodied delivery van; deep teal metal kiosk, ivory plaster wall, reddish sandstone ground. Vehicle screenshots come from Khronos glTF-Sample-Assets ToyCar (CC0) and CesiumMilkTruck (Cesium, CC-BY-4.0); only one frame is used, the delivery side emblem is painted over, and SAM3 masks remove all background before independent normalization. Materials are deterministic procedural interior swatches. Source attribution, masks and operations are in runs/v9_appearance_assets_r1. This is not newly AI-generated image material.

Invariant: complete 4s/24fps/720square colored proxy video, geometry, object poses, camera trajectory, seed40124, model seedance-2.5 and provider params (adaptive/-1, no audio/watermark). Changed: all5 appearance/material images and matching type/material text. Instance IDs I1/I2 in prompt are intentionally neutral rather than source-category IDs. Actual requests are retained locally, Base64 image bytes verified against files.

Scene proxy encoding supports optional appearance_label and prompt_id without modifying source semantic labels or geometry. Generic camera prompt must not reintroduce a source vehicle type. Intermediate r1/r2 packages were not submitted: final r3 packages remove the fixed source-type reference before generation. An initial asset-preparation bfloat16 score conversion issue was fixed by explicit float conversion; no segmentation fallback was used.

Submitted jobs: A8163005, B8163006. Local dirs: seedance_courtyard_v9_s20_original_seed40124 and seedance_courtyard_v9_s20_swapped_seed40124 under Desktop/temp.

Review output: runs/v9_s20_fiveway_r1. Source panels use original Step20 camera; proxy and generation panels use the added lateral camera. Cyan dots are observed image tracks (32 fixed indices of82), pink crosses are fitted projections; these are not independent3D ground truth. Source49 frames are retimed to4s for presentation. Generated clips retain provider fps and hold their last frame only when needed to fill4s, with a visible label. The user input workbook is built from actual submitted request JSON, not intended settings.

36 tests passed, including appearance-label/source-label separation. Replacing vehicle types is a stress test of coarse envelopes, not proof of strict geometric or camera fidelity.
