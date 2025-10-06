import math, random
from typing import List, Tuple, Optional, Dict, Any

Vec2 = Tuple[float, float]

def _l2(a: Vec2, b: Vec2) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _fallback_circle_walk(center: Vec2, n_points: int, min_hop: float, max_hop: float, rng: random.Random) -> List[Vec2]:
    cx, cy = center
    R = (min_hop + max_hop) * 0.5
    base_ang = rng.uniform(0, 2*math.pi)
    pts = []
    for i in range(n_points):
        ang = base_ang + 2*math.pi*i/n_points + rng.uniform(-0.2, 0.2)
        r = R * rng.uniform(0.8, 1.2)
        pts.append((cx + r*math.cos(ang), cy + r*math.sin(ang)))
    pts.append(pts[0])
    return pts

def generate_random_patrol_route(
    *, amap: Optional[object], current_xy: Vec2, n_points: int = 12,
    min_hop: float = 60.0, max_hop: float = 220.0,
    fallback_places: Optional[List[Dict[str, Any]]] = None,
    rng: Optional[random.Random] = None,
) -> List[Vec2]:
    assert n_points >= 3
    rng = rng or random.Random()

    bbox = None
    if amap is not None:
        for attr in ("walkable_bbox", "bbox", "map_bbox", "map_boundary", "boundary"):
            if hasattr(amap, attr):
                bbox = getattr(amap, attr)
                break

    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        cx, cy = current_xy
        angle = rng.uniform(0, 2*math.pi)
        r = rng.uniform(min_hop, max_hop)
        start = (cx + r*math.cos(angle), cy + r*math.sin(angle))
        sx = min(max(start[0], xmin), xmax)
        sy = min(max(start[1], ymin), ymax)
        waypoints = [(sx, sy)]

        last = waypoints[-1]
        for _ in range(n_points - 1):
            tries = 0
            while True:
                tries += 1
                base_ang = rng.uniform(0, 2*math.pi)
                step_r = rng.uniform(min_hop, max_hop)
                cand = (last[0] + step_r*math.cos(base_ang), last[1] + step_r*math.sin(base_ang))
                cand = (min(max(cand[0], xmin), xmax), min(max(cand[1], ymin), ymax))
                if _l2(last, cand) >= 0.6*min_hop or tries > 16:
                    waypoints.append(cand)
                    last = cand
                    break

        if _l2(waypoints[-1], waypoints[0]) > max_hop * 1.2:
            mid = ((waypoints[-1][0] + waypoints[0][0]) * 0.5, (waypoints[-1][1] + waypoints[0][1]) * 0.5)
            waypoints.append(mid)

        return waypoints

    if fallback_places:
        centers: List[Vec2] = []
        for p in fallback_places:
            for k in ("center","centroid","pos","position"):
                if k in p:
                    pt = p[k]
                    break
            else:
                pt = None
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                centers.append((float(pt[0]), float(pt[1])))

        if not centers:
            return _fallback_circle_walk(current_xy, n_points, min_hop, max_hop, rng)

        centers.sort(key=lambda xy: _l2(xy, current_xy))
        pool = centers[: max(8, min(len(centers), 64))]
        waypoints = [pool[0]]
        last = pool[0]
        used = {0}

        import random as _r
        while len(waypoints) < n_points:
            if rng.random() < 0.7:
                cand_idx, cand_xy, best_d = None, None, 1e18
                for i, xy in enumerate(pool):
                    if i in used:
                        continue
                    d = _l2(last, xy)
                    if min_hop*0.5 <= d <= max_hop*1.5 and d < best_d:
                        cand_idx, cand_xy, best_d = i, xy, d
                if cand_xy is None:
                    i = rng.randrange(len(pool))
                    cand_idx, cand_xy = i, pool[i]
            else:
                i = rng.randrange(len(pool))
                cand_idx, cand_xy = i, pool[i]

            waypoints.append(cand_xy)
            used.add(cand_idx)
            last = cand_xy

        if _l2(waypoints[-1], waypoints[0]) > max_hop * 1.8:
            waypoints.append(waypoints[0])

        return waypoints

    return _fallback_circle_walk(current_xy, n_points, min_hop, max_hop, rng)
