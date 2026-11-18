#!/usr/bin/env python3
"""Five-category replacement clouds; original shaders/meshes, explicit grouping."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from pointcloud_batch import ROOT, atomic_json, digest


def grid(shape, origin, resolution):
    x = origin[0] + (np.arange(shape[1])+.5)*resolution
    y = origin[1] + (np.arange(shape[0])+.5)*resolution
    xx, yy = np.meshgrid(x, y)
    return np.c_[xx.ravel(), yy.ravel()]


def height_at(height, origin, resolution, xy):
    x = np.arange(height.shape[1], dtype=float)
    y = np.arange(height.shape[0], dtype=float)
    indices = (np.asarray(xy) - origin) / resolution - .5
    rounded = np.rint(indices)
    indices = np.where(np.isclose(indices, rounded, rtol=0., atol=1e-9), rounded, indices)
    valid = np.isfinite(height)
    # Zero-weight NaN neighbors must not erase a measured grid center or grid line.
    value = RegularGridInterpolator((y, x), np.where(valid, height, 0.),
        bounds_error=False, fill_value=np.nan)(indices[:, ::-1])
    support = RegularGridInterpolator((y, x), valid.astype(float),
        bounds_error=False, fill_value=np.nan)(indices[:, ::-1])
    return np.where(np.isclose(support, 1., rtol=0., atol=1e-12), value, np.nan)


def raster(points, shape, origin, resolution):
    ij = np.floor((points[:, :2]-origin)/resolution).astype(int)
    inside = (ij[:, 0]>=0)&(ij[:, 0]<shape[1])&(ij[:, 1]>=0)&(ij[:, 1]<shape[0])
    ids = ij[inside, 1]*shape[1]+ij[inside, 0]
    low = np.full(np.prod(shape), np.inf, np.float32)
    np.minimum.at(low, ids, points[inside, 2])
    return np.where(np.isfinite(low), low, np.nan).reshape(shape)


def epfl_settings(seed, profile="metric_v2"):
    if profile not in ("legacy_v1", "metric_v2"):
        raise ValueError("unknown EPFL generation profile")
    rng = np.random.default_rng(seed)
    settings = {"seed": int(seed % 1000000), "type": int(seed % 4),
        "dx": float(rng.uniform(-5,5)), "dy": float(rng.uniform(-5,5)),
        "hcomp": float(rng.uniform(2,6)), "vcomp": float(rng.uniform(1,4)), "voffset": 0.0,
        "H": float(rng.uniform(1,1.6)), "lacunarity": 2.0, "warp": .15, "octaves": 5}
    if profile == "metric_v2" and settings["type"] == 3:
        # The Swiss shader passes H as an amplitude gain, not an fBm exponent.
        settings["H"] = float(np.random.default_rng(seed+9000000).uniform(.35,.65))
    return settings


class EpflHeight:
    def __init__(self, source, profile="metric_v2"):
        import moderngl
        self.context = moderngl.create_standalone_context(require=330, backend="egl")
        self.info = {k: self.context.info[k] for k in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION")}
        if "NVIDIA" not in self.info["GL_VENDOR"]:
            raise RuntimeError("requested GPU export did not select NVIDIA; no silent software rendering")
        folder = source / "project/heightmap"
        self.program = self.context.program(vertex_shader=(folder / "heightmap_vshader.glsl").read_text(),
                                             fragment_shader=(folder / "heightmap_fshader.glsl").read_text())
        vertices = np.array([[-1,-1,0,0,0],[1,-1,0,1,0],[-1,1,0,0,1],[1,1,0,1,1]], np.float32)
        buffer = self.context.buffer(vertices.tobytes())
        self.vao = self.context.vertex_array(self.program, [(buffer, "3f 2f", "vpoint", "vtexcoord")])
        texture = self.context.texture((500, 500), 1, dtype="f4")
        self.framebuffer = self.context.framebuffer(color_attachments=[texture])
        self.mode = moderngl.TRIANGLE_STRIP
        self.profile = profile

    def generate(self, seed):
        settings = epfl_settings(seed, self.profile)
        for key, value in settings.items():
            self.program[key].value = value
        self.framebuffer.use(); self.context.viewport = (0,0,500,500)
        self.vao.render(mode=self.mode)
        height = np.frombuffer(self.framebuffer.read(components=1,dtype="f4",alignment=1), np.float32).reshape(500,500).copy()
        if not np.isfinite(height).all() or np.ptp(height) < .05:
            raise ValueError("invalid shader terrain export")
        return height, settings


def entities(family, seed, height):
    rng = np.random.default_rng(seed)
    origin = np.array([-25,-25]); xy = grid(height.shape, origin, .1)
    collision = np.zeros(len(xy), bool)
    clouds = []
    if family == "hilly":
        return np.empty((0,3),np.float32), collision.reshape(height.shape), []
    specifications = []
    for i in range(int(rng.integers(50,101))):
        center = rng.uniform(-23,23,2); radius = float(rng.uniform(.07,.22))
        base = float(height_at(height, origin, .1, center[None])[0]); top = float(rng.uniform(1.8,3.8))
        angles = np.linspace(0,2*np.pi,14,endpoint=False)
        levels = np.linspace(.05,top,20)
        aa, zz = np.meshgrid(angles, levels)
        trunk = np.c_[center[0]+radius*np.cos(aa.ravel()),center[1]+radius*np.sin(aa.ravel()),base+zz.ravel()]
        clouds.append(trunk)
        # Irregular high canopy remains above the bumper-height target band.
        directions = rng.normal(size=(160,3)); directions /= np.linalg.norm(directions,axis=1)[:,None]
        clouds.append(np.r_[center,base+top][None]+directions*np.array([.6,.6,.4]))
        collision |= np.sum((xy-center)**2,axis=1) <= (radius+.05)**2
        specifications.append({"type":"tree","center":center.tolist(),"radius":radius,"height":top})
    if family == "ruin":
        for i in range(int(rng.integers(18,36))):
            center = rng.uniform(-22,22,2); dimensions = rng.uniform([.25,.25,.4],[2,1.5,1.5]); yaw = rng.uniform(-np.pi,np.pi)
            c,s = np.cos(yaw),np.sin(yaw); rotation = np.array([[c,-s],[s,c]])
            base = float(height_at(height,origin,.1,center[None])[0])
            local = rng.uniform(-.5,.5,(260,3))*dimensions
            face = np.arange(len(local))%6; axis=face//2; sign=np.where(face%2==0,-1,1)
            local[np.arange(len(local)),axis] = sign*dimensions[axis]/2
            clouds.append(np.c_[local[:,:2]@rotation.T+center,local[:,2]+base+dimensions[2]/2])
            relative=(xy-center)@rotation
            collision |= (np.abs(relative)<=dimensions[:2]/2+.05).all(axis=1)
            specifications.append({"type":"solid_box","center":center.tolist(),"dimensions":dimensions.tolist(),"yaw":float(yaw)})
    return np.concatenate(clouds).astype(np.float32), collision.reshape(height.shape), specifications


def write_case(output, row, xyz, is_ground, height, origin, resolution, collision, center, full_cloud=None):
    path = output / "cases" / (row["id"]+".npz")
    roi_origin = np.asarray(center)-10.4
    inside = (xyz[:,:2]>=roi_origin).all(axis=1)&(xyz[:,:2]<roi_origin+20.8).all(axis=1)
    fields = dict(xyz=xyz[inside].astype(np.float32), is_ground=is_ground[inside],
        truth_height=height.astype(np.float32), truth_collision=collision, truth_origin=np.asarray(origin),
        truth_resolution=resolution, query_origin=roi_origin, shape=np.array([104,104]), resolution=.2)
    if full_cloud is not None:
        fields["full_xyz"] = full_cloud
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **fields)
    temporary.replace(path)
    row.update(path=str(path.relative_to(output)), sha256=digest(path), roi_size_m=20.8,
        raw_points=int(inside.sum()), ground_points=int(is_ground[inside].sum()),
        truth_valid_fraction=float(np.isfinite(height).mean()))


def mesh_scene(path):
    import trimesh
    import open3d as o3d
    scene = trimesh.load(path, force="scene",process=False)
    floors = [v for k,v in scene.geometry.items() if "floor" in k.lower() and "lamp" not in k.lower()]
    if not floors:
        raise ValueError("no named floor geometry; do not invent a flat floor")
    floor = trimesh.util.concatenate(floors)
    vertical = int(np.argmin(np.ptp(max(floors,key=lambda m:m.area).vertices,axis=0)))
    transforms = {1: np.array([[1,0,0],[0,0,-1],[0,1,0]]),2:np.eye(3),0:np.array([[0,1,0],[0,0,1],[1,0,0]])}
    rotation = transforms[vertical]
    full = trimesh.util.concatenate(list(scene.geometry.values()))
    floor.vertices = floor.vertices@rotation.T; full.vertices = full.vertices@rotation.T
    offset = np.median(floor.vertices[:,2]); floor.vertices[:,2]-=offset; full.vertices[:,2]-=offset
    origin = floor.bounds[0,:2]-.5
    shape = tuple(np.ceil((floor.bounds[1,:2]-origin+.5)/.1).astype(int)[::-1])
    if min(shape)<10 or np.prod(shape)>3000000:
        raise ValueError("floor geometry has unsupported metric dimensions")
    xy=grid(shape,origin,.1)
    def ray_scene(mesh):
        target=o3d.t.geometry.RaycastingScene()
        target.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices,np.float32)),o3d.core.Tensor(np.asarray(mesh.faces,np.uint32)))
        return target
    floor_cast=ray_scene(floor); full_cast=ray_scene(full)
    top=float(full.bounds[1,2]+1)
    rays=np.c_[xy,np.full(len(xy),top),np.tile([0,0,-1],(len(xy),1))].astype(np.float32)
    hit=floor_cast.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    height=np.where(np.isfinite(hit),top-hit,np.nan).reshape(shape)
    distance=np.full(len(xy),np.inf,np.float32)
    valid=np.isfinite(height.ravel())
    for level in (.35,.7,1.25):
        queries=np.c_[xy[valid],height.ravel()[valid]+level].astype(np.float32)
        distance[valid]=np.minimum(distance[valid],full_cast.compute_distance(o3d.core.Tensor(queries)).numpy())
    collision=(distance<=.12).reshape(shape)&np.isfinite(height)
    # Geometry fingerprint ignores material/text labels and vertex ordering.
    canonical=np.unique(np.round(full.vertices,4),axis=0).astype(np.float32)
    geometry_hash=hashlib.sha256(canonical.tobytes()).hexdigest()
    return full,floor,height,origin,collision,geometry_hash,vertical


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--phase",choices=("epfl","indoor","road"),required=True)
    parser.add_argument("--per-family",type=int,default=500)
    parser.add_argument("--seed",type=int,default=31261013)
    parser.add_argument("--generator-profile",choices=("legacy_v1","metric_v2"),default="metric_v2")
    parser.add_argument("--road-registration",choices=("semantic_suma","kitti_gt"),default="semantic_suma")
    parser.add_argument("--road-export-ids",help="explicit comma-separated scan IDs; all registration neighbors retained")
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True); (args.output/"cases").mkdir(exist_ok=True)
    raw=ROOT/"data/public_clouds"; manifest_path=args.output/"dataset_manifest.json"
    catalog=json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema_version":2,"date":"2026-10-13","datasets":[],"selection_audit":[],
        "scope":"paper-category replacements; NOT original ground truth or full temporal paper reproduction",
        "phases":{},"source_manifest":str(raw/"paper_sources.json")}
    if args.phase in catalog["phases"]:
        raise ValueError("phase already exists; keep frozen data and choose a new output")
    rows=catalog["datasets"]
    if args.phase=="epfl":
        generator=EpflHeight(raw/"epfl_generator",args.generator_profile); seen=set()
        for fi,family in enumerate(("hilly","forest","ruin")):
            for index in range(args.per_family):
                seed=args.seed+fi*10000+index; rng=np.random.default_rng(seed)
                height, settings=generator.generate(seed)
                sha=hashlib.sha256(height.tobytes()).hexdigest()
                if sha in seen:
                    raise ValueError("duplicate generated geometry")
                seen.add(sha); origin=np.array([-25,-25]); xy=grid(height.shape,origin,.1).reshape(500,500,2)[::2,::2].reshape(-1,2)
                ground=np.c_[xy,height[::2,::2].ravel()]
                objects,collision,specs=entities(family,seed,height)
                xyz=np.r_[ground,objects].astype(np.float32); flags=np.r_[np.ones(len(ground),bool),np.zeros(len(objects),bool)]
                row={"id":f"epfl_{family}_{index:04d}","family":family,"cluster_id":f"epfl_{family}_{seed}",
                    "geometry_seed":seed,"shader_parameters":settings,"generator_profile":args.generator_profile,
                    "geometry_sha256":sha,"source_kind":"epfl_shader_geometry",
                    "reference_kind":"generated_dense_surface","source_extent_m":[50,50],"entities":specs}
                write_case(args.output,row,xyz,flags,height,origin,.1,collision,rng.uniform(-6,6,2),full_cloud=xyz)
                rows.append(row)
                if (index+1)%50==0:
                    atomic_json(manifest_path,catalog); print(f"[clouds] {family} {index+1}/{args.per_family}",flush=True)
        catalog["phases"]["epfl"]={"generator":generator.info,"profile":args.generator_profile,
            "count":3*args.per_family,"seed":args.seed,"script_sha256":digest(__file__)}
        generator.context.release()
    elif args.phase=="indoor":
        import trimesh
        paths=sorted((raw/"scenenet_original").rglob("*.obj")); accepted=[]
        # Geometry-only screening, not prediction/quality-based selection.
        for path in paths:
            mesh=None
            try:
                mesh=mesh_scene(path)
                if np.isfinite(mesh[2]).sum()<100:
                    raise ValueError("fewer than 100 floor reference cells")
                accepted.append(path)
                catalog["selection_audit"].append({"path":str(path),"status":"included","geometry_sha256":mesh[5]})
            except Exception as exc:
                catalog["selection_audit"].append({"path":str(path),"status":"excluded_geometry","reason":f"{type(exc).__name__}: {exc}"})
            del mesh
        if not accepted:
            raise ValueError("no usable annotated SceneNet floors")
        for pi,path in enumerate(accepted):
            full,floor,height,origin,collision,geo_sha,vertical=mesh_scene(path)
            count=args.per_family//len(accepted)+(pi<args.per_family%len(accepted))
            tokens=re.findall(r"\d+",path.stem)
            layout=path.parent.name+"_"+("_".join(tokens) if tokens else path.stem)
            for repeat in range(count):
                seed=args.seed+200000+pi*100+repeat
                points,_=trimesh.sample.sample_surface(full,60000,seed=seed)
                ground,_=trimesh.sample.sample_surface(floor,10000,seed=seed+1)
                xyz=np.r_[ground,points].astype(np.float32); flags=np.r_[np.ones(len(ground),bool),np.zeros(len(points),bool)]
                center=(floor.bounds[0,:2]+floor.bounds[1,:2])/2
                row={"id":f"scenenet_{pi:03d}_{repeat:02d}","family":"indoor","cluster_id":"scenenet_"+layout,
                    "source_kind":"scenenet_original_mesh","reference_kind":"mesh_distance_geometry_proxy",
                    "mesh_sha256":digest(path),"geometry_sha256":geo_sha,"source_path":str(path),"source_vertical_axis":vertical,
                    "surface_sample_seed":seed,"reference_note":"named floor mesh and 3 bumper-height distance probes; not certified solid collision truth"}
                write_case(args.output,row,xyz,flags,height,origin,.1,collision,center,full_cloud=xyz)
                rows.append(row)
            atomic_json(manifest_path,catalog); print(f"[clouds] SceneNet {pi+1}/{len(accepted)} source resources",flush=True)
        catalog["phases"]["indoor"]={"obj_resources":len(paths),"usable_resources":len(accepted),"count":args.per_family,
            "seed":args.seed,"script_sha256":digest(__file__)}
    else:
        from paper_road_clouds import build_road
        build_road(args,raw,catalog)
    atomic_json(manifest_path,catalog)
    print(f"[clouds] complete {len(rows)} point-cloud cases: {manifest_path}",flush=True)


if __name__=="__main__":
    main()
