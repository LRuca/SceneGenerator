"""Opaque triangle ray intersections in the same camera-Z convention as primitives."""
import numpy as np

def intersect(origin,directions,vertices,faces):
    origin=np.asarray(origin);directions=np.asarray(directions);tri=np.asarray(vertices)[np.asarray(faces)]
    count=len(directions);depth=np.full(count,np.inf);which=np.full(count,-1,int)
    for j,(a,b,c) in enumerate(tri):
        e1=b-a;e2=c-a;h=np.cross(directions,e2);det=h@e1
        good=np.abs(det)>1e-10;inv=np.zeros_like(det);inv[good]=1/det[good]
        s=origin-a;u=(h@s)*inv;q=np.cross(s,e1);v=(directions@q)*inv;t=(q@e2)*inv
        hit=good&(u>=-1e-7)&(v>=-1e-7)&(u+v<=1+1e-7)&(t>1e-6)&(t<depth)
        depth[hit]=t[hit];which[hit]=j
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.linalg.norm(normal,axis=-1,keepdims=True).clip(1e-12)
    return depth,which,normal
