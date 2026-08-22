"""Bathymetric mapping and scour quantification.

Every sonar ping yields one nearest-return sounding per beam. Projected
through the estimated vehicle pose these accumulate into a gridded depth map
of the riverbed. Two products are derived from it:

  coverage  the fraction of the survey box that has been sounded, which is the
            operational measure of whether a search was actually complete

  scour     the depth and volume of material removed around each bridge pier,
            obtained by comparing the mapped bed against the undisturbed bed
            level estimated from an annulus outside the scour hole

Scour is the quantity that matters for the inspection mission. IRC:SP:35
requires periodic underwater inspection of bridge foundations, and scour depth
is what determines whether a pier is at risk. Measuring it currently needs
divers; here it falls out of the survey automatically.

Because the map is built in the *estimated* frame, any navigation drift
propagates straight into it. The evaluation therefore reports mapped scour
against ground truth, which measures the whole chain, not just the sonar.
"""

from __future__ import annotations

import numpy as np


class BathymetryMap:
    """Gridded running mean of seabed soundings."""

    def __init__(self, x_min, x_max, y_min, y_max, res=0.5):
        self.x0, self.y0, self.res = float(x_min), float(y_min), float(res)
        self.nx = int(np.ceil((x_max - x_min) / res)) + 1
        self.ny = int(np.ceil((y_max - y_min) / res)) + 1
        self.sum = np.zeros((self.ny, self.nx))
        self.count = np.zeros((self.ny, self.nx), dtype=np.int32)
        self.min_z = np.full((self.ny, self.nx), np.inf)
        self.n_soundings = 0

    def add(self, points, max_incidence_deg=None, incidence=None):
        """Insert a batch of world-frame soundings.

        Returns at very shallow grazing angles are long, smeared and poorly
        localised, so they can be rejected by incidence angle.
        """
        p = np.asarray(points, float)
        if p.ndim != 2 or p.shape[0] == 0:
            return 0
        good = np.isfinite(p).all(axis=1)
        if max_incidence_deg is not None and incidence is not None:
            inc = np.asarray(incidence, float)
            good &= np.isfinite(inc) & (inc <= max_incidence_deg)
        p = p[good]
        if len(p) == 0:
            return 0
        i = np.round((p[:, 0] - self.x0) / self.res).astype(int)
        j = np.round((p[:, 1] - self.y0) / self.res).astype(int)
        inb = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        i, j, z = i[inb], j[inb], p[inb, 2]
        np.add.at(self.sum, (j, i), z)
        np.add.at(self.count, (j, i), 1)
        np.minimum.at(self.min_z, (j, i), z)
        self.n_soundings += len(z)
        return len(z)

    @property
    def mean_z(self):
        out = np.full_like(self.sum, np.nan)
        m = self.count > 0
        out[m] = self.sum[m] / self.count[m]
        return out

    @property
    def xs(self):
        return self.x0 + np.arange(self.nx) * self.res

    @property
    def ys(self):
        return self.y0 + np.arange(self.ny) * self.res

    def coverage(self, box=None, min_hits=1):
        """Fraction of cells with at least ``min_hits`` soundings."""
        c = self.count
        if box is not None:
            x0, x1, y0, y1 = box
            i0 = max(int((x0 - self.x0) / self.res), 0)
            i1 = min(int((x1 - self.x0) / self.res) + 1, self.nx)
            j0 = max(int((y0 - self.y0) / self.res), 0)
            j1 = min(int((y1 - self.y0) / self.res) + 1, self.ny)
            c = c[j0:j1, i0:i1]
        if c.size == 0:
            return 0.0
        return float(np.mean(c >= min_hits))

    def sample(self, x, y):
        """Nearest-cell mapped depth, NaN where unsounded."""
        i = np.round((np.asarray(x) - self.x0) / self.res).astype(int)
        j = np.round((np.asarray(y) - self.y0) / self.res).astype(int)
        ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        out = np.full(np.shape(i), np.nan, dtype=float)
        mz = self.mean_z
        out[ok] = mz[j[ok], i[ok]]
        return out


def estimate_scour(bmap: BathymetryMap, pier_xy, pier_radius,
                   search_radius=11.0, ref_inner=1.35, ref_outer=1.85):
    """Measure the scour hole around one pier.

    The undisturbed bed level is taken as the median of an annulus outside the
    hole, between ``ref_inner`` and ``ref_outer`` times the search radius. Depth
    is the deepest excursion below that datum, and volume is the integral of
    the depression over the cells inside the search radius.

    Returns None if the pier surroundings were not sufficiently sounded, so a
    partially surveyed pier is reported as unmeasured rather than as a
    spuriously shallow one.
    """
    X, Y = np.meshgrid(bmap.xs, bmap.ys)
    R = np.hypot(X - pier_xy[0], Y - pier_xy[1])
    z = bmap.mean_z
    have = np.isfinite(z)

    ring = have & (R > search_radius * ref_inner) & (R < search_radius * ref_outer)
    if np.count_nonzero(ring) < 25:
        return None
    datum = float(np.median(z[ring]))

    inside = have & (R <= search_radius) & (R > pier_radius * 1.05)
    if np.count_nonzero(inside) < 25:
        return None

    depth_field = datum - z
    depth_field[~inside] = 0.0
    depth_field = np.maximum(depth_field, 0.0)

    cell = bmap.res ** 2
    # Robust depth: the 98th percentile rejects isolated outlier soundings.
    vals = (datum - z)[inside]
    max_depth = float(np.percentile(vals, 98))
    volume = float(depth_field.sum() * cell)
    n_cells = int(np.count_nonzero(inside))
    filled = float(np.count_nonzero(inside) /
                   max(np.count_nonzero((R <= search_radius) & (R > pier_radius * 1.05)), 1))
    return {
        "datum_z": datum,
        "max_depth": max_depth,
        "volume": volume,
        "cells": n_cells,
        "coverage": filled,
    }


def bed_residual(bmap: BathymetryMap, window=17):
    """Height of the mapped surface above a smooth estimate of the bare bed.

    The bed is estimated by a grey-scale morphological opening: a rolling
    minimum followed by a rolling maximum. That follows broad bathymetry,
    including scour hollows, while removing compact objects standing proud of
    it. What remains in the residual is precisely the things that do not
    belong to the riverbed.
    """
    from scipy.ndimage import (minimum_filter, maximum_filter, uniform_filter,
                               distance_transform_edt)

    z = bmap.mean_z
    have = np.isfinite(z)
    if not have.any():
        return np.full_like(z, np.nan), np.zeros_like(z)

    # Unmapped cells must be filled before filtering, but filling them with a
    # global constant puts a cliff around the surveyed area. The morphological
    # opening then reads that cliff as a very large object and merges genuine
    # targets near the boundary into it. Nearest-valid extrapolation instead
    # continues the bed smoothly outward and leaves no artificial step.
    idx = distance_transform_edt(~have, return_distances=False,
                                 return_indices=True)
    filled = z[tuple(idx)]

    opened = maximum_filter(minimum_filter(filled, size=window), size=window)
    opened = uniform_filter(opened, size=max(window // 2, 3))
    res = np.where(have, filled - opened, np.nan)
    return res, opened


def detect_objects_from_residual(bmap: BathymetryMap, min_height=0.45,
                                 min_area_m2=2.0, window=55,
                                 min_hits=2, edge_erode=1, max_extent=45.0,
                                 close_cells=5):
    """Find objects standing above the riverbed, from geometry alone.

    This is a purely geometric detector: it needs no training data and no
    labelled sonar, only a bathymetric map. It provides an independent
    detection channel alongside the learned classifier, and it is what makes
    the system useful for target classes that have no public training data,
    such as submerged buses.

    Parameter choice is set by the physics rather than tuned. ``window`` must
    exceed the largest object of interest, because morphological opening only
    removes features smaller than its structuring element: a window shorter
    than an 11 m bus absorbs the bus into the bed estimate and hides it
    entirely. The default of 55 cells is 27.5 m, clear of the 15 m deck slab.
    Depressions such as scour hollows survive opening at any window size, so
    the window can be set generously. ``min_height`` is set above the measured
    bathymetric map RMSE so that map noise alone cannot raise a detection.
    """
    from scipy.ndimage import label, find_objects

    from scipy.ndimage import binary_erosion

    res, bed = bed_residual(bmap, window)
    # Only trust cells sounded more than once, and pull the analysis away from
    # the ragged edge of the surveyed area, where the morphological bed
    # estimate has no support on one side and produces spurious steps.
    solid = bmap.count >= min_hits
    if edge_erode > 0:
        solid = binary_erosion(solid, np.ones((2 * edge_erode + 1,) * 2))
    mask = np.isfinite(res) & (res > min_height) & solid
    # Sparse coverage breaks a single object into several fragments. Closing
    # rejoins them so one object is reported once, with a usable extent.
    if close_cells > 0:
        from scipy.ndimage import binary_closing
        mask = binary_closing(mask, np.ones((close_cells,) * 2))
    lab, n = label(mask)
    cell = bmap.res ** 2
    out = []
    for i, sl in enumerate(find_objects(lab), start=1):
        blob = (lab[sl] == i)
        area = float(blob.sum() * cell)
        if area < min_area_m2:
            continue
        ext_x = (sl[1].stop - sl[1].start) * bmap.res
        ext_y = (sl[0].stop - sl[0].start) * bmap.res
        # A blob spanning tens of metres is a survey-boundary artefact, not an
        # object lying on the riverbed.
        if max(ext_x, ext_y) > max_extent:
            continue
        ys, xs = np.nonzero(blob)
        gx = bmap.x0 + (sl[1].start + xs) * bmap.res
        gy = bmap.y0 + (sl[0].start + ys) * bmap.res
        h = res[sl][blob]
        out.append({
            "centre": (float(gx.mean()), float(gy.mean())),
            "area": area,
            "height": float(np.nanmax(h)),
            "mean_height": float(np.nanmean(h)),
            "extent": (float(gx.max() - gx.min()), float(gy.max() - gy.min())),
            "cells": int(blob.sum()),
        })
    out.sort(key=lambda d: -d["area"])
    return out, res, bed


def match_detections(detections, truth, gate=6.0, footprint_aware=True):
    """Greedy nearest-neighbour association of detections to known targets.

    For extended targets a fixed centroid gate is the wrong test: a
    15 m deck slab detected anywhere along its length is found, even
    though its cluster centroid may sit several metres from the object centre.
    When ``footprint_aware`` is set the gate is widened to half the target's
    own diagonal, so large objects are judged by whether the detection falls on
    them rather than by an arbitrary fixed radius.

    Returns (matches, missed, false_positives) where a match carries the
    localisation error in metres.
    """
    used = set()
    matches, missed = [], []
    for name, tg in truth.items():
        c = np.asarray(tg["centre"][:2], float)
        g = gate
        if footprint_aware and "size" in tg:
            s = np.asarray(tg["size"], float)
            g = max(gate, 0.5 * float(np.hypot(s[0], s[1])))
        best, bd = None, np.inf
        for k, det in enumerate(detections):
            if k in used:
                continue
            dist = float(np.linalg.norm(np.asarray(det["centre"]) - c))
            if dist < bd:
                best, bd = k, dist
        if best is not None and bd <= g:
            used.add(best)
            matches.append({"name": name, "truth": tuple(c),
                            "detected": detections[best]["centre"],
                            "error": bd, "det": detections[best],
                            "class": tg.get("class", "")})
        else:
            missed.append({"name": name, "truth": tuple(c),
                           "class": tg.get("class", ""), "nearest": bd})
    fps = [d for k, d in enumerate(detections) if k not in used]
    return matches, missed, fps


class TargetTracker:
    """Clusters and tracks detections so each object is reported once.

    A detection is associated with an existing track if it falls within
    ``gate`` metres of it, otherwise it opens a new track. A track is only
    promoted to a confirmed contact after it has been seen from several
    distinct pings, which suppresses isolated false alarms from speckle.
    """

    def __init__(self, gate=3.5, confirm_hits=3):
        self.gate = gate
        self.confirm_hits = confirm_hits
        self.tracks = []

    def update(self, detections, t=0.0):
        """detections: list of dicts with 'position', 'label', 'confidence'."""
        for d in detections:
            p = np.asarray(d["position"], float)
            best, bd = None, np.inf
            for tr in self.tracks:
                dist = float(np.linalg.norm(tr["position"] - p))
                if dist < bd:
                    best, bd = tr, dist
            if best is not None and bd <= self.gate:
                n = best["hits"]
                best["position"] = (best["position"] * n + p) / (n + 1)
                best["hits"] = n + 1
                best["confidence"] = max(best["confidence"], d.get("confidence", 0.0))
                best["last_t"] = t
                best["labels"].append(d.get("label", "unknown"))
            else:
                self.tracks.append({
                    "position": p, "hits": 1, "first_t": t, "last_t": t,
                    "confidence": d.get("confidence", 0.0),
                    "labels": [d.get("label", "unknown")],
                })
        return self.tracks

    @property
    def confirmed(self):
        out = []
        for tr in self.tracks:
            if tr["hits"] >= self.confirm_hits:
                lab = max(set(tr["labels"]), key=tr["labels"].count)
                out.append({**tr, "label": lab})
        return out
