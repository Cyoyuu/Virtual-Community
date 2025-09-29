from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import random

# ----------------------------
# Event Base & Implementations
# ----------------------------

ANSI_ORANGE = "\033[38;5;214m"
ANSI_RESET  = "\033[0m"

@dataclass
class BaseEvent:
    """Base class for a timed event."""
    name: str
    start_step: int
    duration_steps: int

    def apply(self, env: Any) -> None:
        """Apply the event's effect into the environment (or its modules)."""
        raise NotImplementedError

    def revert(self, env: Any) -> None:
        """Revert (cleanup) the event's effect."""
        raise NotImplementedError

    def is_expired(self, current_step: int) -> bool:
        return current_step >= self.start_step + self.duration_steps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start_step": self.start_step,
            "duration_steps": self.duration_steps,
        }


class RoadClosureEvent(BaseEvent):
    """
    Close a specific road in the Amap navigation app for a fixed duration.
    Effect: add road_id to Amap.blocked_roads; on expiry, remove it.
    """
    def __init__(self, start_step: int, duration_steps: int, road_id: str):
        super().__init__(name="road_closure", start_step=start_step, duration_steps=duration_steps)
        self.road_id = road_id

    def apply(self, env: Any) -> None:
        amap = _get_amap(env)
        if amap is None:
            return
        if not hasattr(amap, "blocked_roads"):
            amap.blocked_roads = set()
        amap.blocked_roads.add(self.road_id)

    def revert(self, env: Any) -> None:
        amap = _get_amap(env)
        if amap is None:
            return
        if hasattr(amap, "blocked_roads"):
            amap.blocked_roads.discard(self.road_id)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["road_id"] = self.road_id
        return d


# ----------------------------
# Event Manager
# ----------------------------

class EventManager:
    """
    Probabilistic timed event manager.

    Usage:
        em = EventManager(seed=0, specs={
            "road_closure": {
                "prob": 0.02,                # per-step trigger probability
                "duration_steps": (5, 12),   # fixed int or (min,max) inclusive
                # Optional: "max_concurrent": 3
            }
        })

        # In env.step(...) or a similar loop:
        em.step(current_step, env=self)  # 'self' is the env that holds nav_app=Amap
    """
    def __init__(self, seed: Optional[int] = None, specs: Optional[Dict[str, Dict[str, Any]]] = None):
        self.rng = random.Random(seed)
        self.specs = specs or {}
        self.active_events: List[BaseEvent] = []
        self.event_log: List[Dict[str, Any]] = []  # append-only history

    # ---- Public API

    def _log_active_status(self, current_step: int, env: Any = None) -> None:
        """Log remaining steps for all active road closures (orange)."""
        for ev in self.active_events:
            if isinstance(ev, RoadClosureEvent):
                remaining = (ev.start_step + ev.duration_steps) - current_step
                if remaining < 0:
                    remaining = 0
                print(
                    f"{ANSI_ORANGE}[Event] Road '{ev.road_id}' is CLOSED — "
                    f"{remaining} step(s) until reopening.{ANSI_RESET}"
                )


    def step(self, current_step: int, env: Any = None) -> None:
        """Advance one simulation step: expire old events, then maybe trigger new ones."""
        # 1) Expire and revert old events
        still_active: List[BaseEvent] = []
        for ev in self.active_events:
            if ev.is_expired(current_step):
                try:
                    if env is not None:
                        ev.revert(env)
                finally:
                    self.event_log.append({"type": "expired", "event": ev.to_dict(), "step": current_step})
            else:
                still_active.append(ev)
        self.active_events = still_active

        # 2) Try to trigger new events by spec
        for ev_type, cfg in self.specs.items():
            prob = float(cfg.get("prob", 0.0))
            if prob <= 0.0:
                continue

            # Max concurrent constraint (optional)
            max_conc = cfg.get("max_concurrent", None)
            if max_conc is not None:
                if self._count_active_by_name(ev_type) >= int(max_conc):
                    continue

            # Bernoulli trial
            if self.rng.random() < prob:
                # Construct and apply event
                ev = self._construct_event(ev_type, current_step, cfg, env)
                if ev is None:
                    # Could not construct (e.g., no available road to close)
                    continue
                if env is not None:
                    ev.apply(env)
                self.active_events.append(ev)
                self.event_log.append({"type": "triggered", "event": ev.to_dict(), "step": current_step})

        self._log_active_status(current_step, env)

    def get_active_events(self) -> List[Dict[str, Any]]:
        return [ev.to_dict() for ev in self.active_events]

    def force_event(self, ev: BaseEvent, env: Any = None) -> None:
        """Force-insert an event immediately."""
        if env is not None:
            ev.apply(env)
        self.active_events.append(ev)
        self.event_log.append({"type": "forced", "event": ev.to_dict(), "step": ev.start_step})

    def clear_all(self, env: Any = None) -> None:
        """Expire and revert all events right now."""
        for ev in self.active_events:
            try:
                if env is not None:
                    ev.revert(env)
            finally:
                self.event_log.append({"type": "cleared", "event": ev.to_dict()})
        self.active_events.clear()

    # ---- Internals

    def _construct_event(self, ev_type: str, current_step: int, cfg: Dict[str, Any], env: Any) -> Optional[BaseEvent]:
        """Create a concrete event instance from type+config."""
        duration = cfg.get("duration_steps", 1)
        if isinstance(duration, (list, tuple)) and len(duration) == 2:
            d0, d1 = int(duration[0]), int(duration[1])
            duration_steps = self.rng.randint(min(d0, d1), max(d0, d1))
        else:
            duration_steps = int(duration)

        if ev_type == "road_closure":
            road_id = _pick_random_open_road(env, rng=self.rng, avoid=self._current_closed_roads(env))
            if road_id is None:
                return None
            return RoadClosureEvent(start_step=current_step, duration_steps=duration_steps, road_id=road_id)

        # Unknown event type: ignore
        return None

    def _count_active_by_name(self, name: str) -> int:
        return sum(1 for ev in self.active_events if ev.name == name)

    def _current_closed_roads(self, env: Any) -> set:
        amap = _get_amap(env)
        if amap is None or not hasattr(amap, "blocked_roads"):
            return set()
        return set(amap.blocked_roads)


# ----------------------------
# Helpers
# ----------------------------

def _wp_xy(wp):
    loc = getattr(wp, "location", None) or getattr(wp, "pos", None)
    if loc is not None:
        return float(loc[0]), float(loc[1])
    x = getattr(wp, "x", None)
    y = getattr(wp, "y", None)
    return float(x), float(y)

def _nearest_road_id_from_xy(amap, x, y):
    if amap is None or not hasattr(amap, "waypoints"): 
        return None
    best, best_d2 = None, float("inf")
    for wp in amap.waypoints:
        rid = getattr(wp, "belong", None)
        if not rid:
            continue
        wx, wy = _wp_xy(wp)
        d2 = (wx - x)**2 + (wy - y)**2
        if d2 < best_d2:
            best, best_d2 = rid, d2
    return best

def _get_amap(env: Any):
    """
    Try to fetch the navigation app (Amap) from the environment or directly.
    Expected locations:
        - env.nav_app
        - env.get("nav_app") if env is dict-like
        - if env itself is an Amap
    """
    if env is None:
        return None
    # direct
    if hasattr(env, "waypoints") and hasattr(env, "blocked_roads"):
        return env
    # env.nav_app
    if hasattr(env, "nav_app"):
        return getattr(env, "nav_app")
    # dict-like
    try:
        return env.get("nav_app", None)
    except Exception:
        return None

def _pick_random_open_road(env, rng: random.Random, avoid: Optional[set] = None) -> Optional[str]:
    amap = _get_amap(env)
    if amap is None or not hasattr(amap, "waypoints"):
        return None

    road_ids = {wp.belong for wp in amap.waypoints if getattr(wp, "belong", None)}
    if not road_ids:
        return None

    avoid = set(avoid or set())

    current_poses = []
    try:
        current_poses = [p for p in env.config.get("agent_poses", [])]
    except Exception:
        pass
    for p in current_poses:
        try:
            x, y = float(p[0]), float(p[1])
            rid = _nearest_road_id_from_xy(amap, x, y)
            if rid:
                avoid.add(rid)
        except Exception:
            continue

    blocked = set(getattr(amap, "blocked_roads", set()))
    candidates = [rid for rid in road_ids if rid not in avoid and rid not in blocked]
    if not candidates:
        return None
    return rng.choice(candidates)
