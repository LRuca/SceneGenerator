# Why a low point reprojection error does not certify the white volume

Sparse fitting jointly adjusts local feature coordinates q and object pose (rotation R, translation C). A point projects from world position X = C + Rq. The rendered primitive box is a separate volume defined in the object coordinate system.

For any local offset d, q' = q - d and C' = C + Rd preserve X, so their point reprojections are identical. The box moves with C unless its geometry is adjusted too. In one dimension, 10 + 0.4 = 10.3 + 0.1: the feature stays at 10.4 while the proxy center shifts by 0.3.

Existing depth, anchor and motion priors partially restrict this ambiguity; this is not an invariance of every objective term. But there is no sufficiently strong explicit attachment between the recovered feature constellation and the proxy surfaces/volume. Low image-point error alone therefore cannot establish a correct occupied volume.

The audit found roughly half of shared body-local points outside the current coarse box at 3% tolerance. This alone does not prove a wrong trajectory: a simplified proxy can omit visible parts. It does show that feature accuracy and proxy-volume accuracy must be evaluated separately.

A second, experimentally isolated issue was unsigned point-to-surface fitting: a visible depth sample could fit the box's back surface. V3 constrains correspondences to camera-visible bounded faces for selected convex box/slab parts. It improves that controlled case but does not solve feature attachment, general self-occlusion, or collision avoidance.

Potential next work (not implemented here): fix the local coordinate gauge using trusted shape anchors; constrain verified body features to appropriate proxy parts; retain uncertainty for occluded points; validate visible silhouette, first visible depth and free space in addition to reprojection. Do not hard-clamp all feature points to a crude box without checking whether the proxy omitted real structure.
