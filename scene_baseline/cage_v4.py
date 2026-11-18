"""Low-dimensional coarse tube deformation, used identically by fit and rendering."""
import numpy as np

def projector(sections=5):
    y=np.linspace(-.5,.5,sections)
    X=np.column_stack([np.ones(sections),y])
    return np.eye(sections)-X@np.linalg.pinv(X)

def smooth_basis(samples=17):
    from scipy.interpolate import CubicSpline
    return CubicSpline(np.linspace(0,1,5),np.eye(5),axis=0,bc_type="natural")(np.linspace(0,1,samples))

def local_vertices(state,t,circumference=12):
    shared=state["shared_size"];size=np.asarray(shared,float)
    bend=np.asarray(state["bend_xz"],float)[t]
    bend=projector(len(bend))@bend
    radial=np.asarray(state.get("radial_log",np.zeros((len(state["bend_xz"]),len(bend),2))),float)[t]
    if state.get("axis_mode")=="smooth_v5":
        B=smooth_basis();bend=B@bend;radial=B@radial
    y=np.linspace(-.5,.5,len(bend));phi=np.arange(circumference)*2*np.pi/circumference
    rings=[]
    for k in range(len(y)):
        rings.append(np.column_stack([size[0]*.5*np.exp(radial[k,0])*np.cos(phi)+size[1]*bend[k,0],
          np.full(circumference,size[1]*y[k]),
          size[2]*.5*np.exp(radial[k,1])*np.sin(phi)+size[1]*bend[k,1]]))
    return np.concatenate(rings)

def faces(sections=5,circumference=12):
    out=[]
    for j in range(1,circumference-1):out.append([0,j+1,j])
    for k in range(sections-1):
        for j in range(circumference):
            a=k*circumference+j;b=k*circumference+(j+1)%circumference
            out.extend([[a,b,b+circumference],[a,b+circumference,a+circumference]])
    a=(sections-1)*circumference
    for j in range(1,circumference-1):out.append([a,a+j,a+j+1])
    return np.asarray(out,int)

def world_mesh(entity,t):
    vertices=local_vertices(entity["deformation"],t)
    T=np.asarray(entity["entity_to_world"])[t]
    return vertices@T[:3,:3].T+T[:3,3],faces(sections=17 if entity["deformation"].get("axis_mode")=="smooth_v5" else 5)

def validate(state,n):
    if state.get("schema")!="scene_baseline.cage.v4":raise ValueError("Cage schema")
    size=np.asarray(state["shared_size"],float);bend=np.asarray(state["bend_xz"],float)
    radial=np.asarray(state.get("radial_log",np.zeros((n,5,2))),float)
    if size.shape!=(3,) or not np.isfinite(size).all() or np.any(size<=0):raise ValueError("Shared cage dimensions")
    if bend.shape!=(n,5,2) or not np.isfinite(bend).all() or np.max(np.abs(bend))>.30:raise ValueError("Cage bending bounds")
    if radial.shape!=(n,5,2) or not np.isfinite(radial).all() or np.max(np.abs(radial))>.15:raise ValueError("Cage radial bounds")
    if not np.allclose(np.einsum("ij,tjk->tik",projector(),bend),bend,atol=1e-6):raise ValueError("Rigid translation/tilt remains in local deformation")
    if len(state.get("state_source",[]))!=n:raise ValueError("Cage evidence timeline")
    return state

def make_state(size,bend,source):
    b=np.einsum("ij,tjk->tik",projector(),np.asarray(bend,float))
    state={"schema":"scene_baseline.cage.v4","shared_size":np.asarray(size).tolist(),"bend_xz":b.tolist(),
      "radial_log":np.zeros_like(b).tolist(),"state_source":list(source),
      "parameterization":"five cross-sections; constant and linear bend components assigned to root pose; no anatomical detail"}
    validate(state,len(b));return state

def torch_vertices(size,bend,circumference=12,radial_log=None,smooth=False):
    import torch
    n,sections,_=bend.shape
    P=torch.as_tensor(projector(sections),device=bend.device,dtype=bend.dtype)
    bend=torch.einsum("ij,tjk->tik",P,bend)
    if smooth:
        B=torch.as_tensor(smooth_basis(),device=bend.device,dtype=bend.dtype)
        bend=torch.einsum("ij,tjk->tik",B,bend)
        if radial_log is not None:radial_log=torch.einsum("ij,tjk->tik",B,radial_log)
        sections=bend.shape[1]
    y=torch.linspace(-.5,.5,sections,device=bend.device,dtype=bend.dtype)
    phi=torch.arange(circumference,device=bend.device,dtype=bend.dtype)*2*torch.pi/circumference
    radial=torch.zeros_like(bend) if radial_log is None else radial_log
    x=size[0]*.5*torch.exp(radial[:,:,0,None])*torch.cos(phi)[None,None]+size[1]*bend[:,:,0,None]
    z=size[2]*.5*torch.exp(radial[:,:,1,None])*torch.sin(phi)[None,None]+size[1]*bend[:,:,1,None]
    yy=(size[1]*y)[None,:,None].expand(n,-1,circumference)
    return torch.stack([x,yy,z],-1).reshape(n,-1,3)

def so3(v):
    import torch
    theta=torch.linalg.norm(v,dim=-1)
    x,y,z=v.unbind(-1);o=torch.zeros_like(x)
    K=torch.stack([o,-z,y,z,o,-x,-y,x,o],-1).reshape(-1,3,3)
    I=torch.eye(3,dtype=v.dtype,device=v.device)[None]
    return I+torch.sinc(theta/torch.pi)[:,None,None]*K+(.5*torch.sinc(theta/(2*torch.pi)).square())[:,None,None]*(K@K)
