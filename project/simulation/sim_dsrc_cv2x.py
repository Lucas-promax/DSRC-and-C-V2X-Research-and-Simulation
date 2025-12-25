#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#type: ignore
"""
DSRC vs C-V2X (Mode 4) single-file Monte-Carlo simulation
Outputs all files under ./simulation/
- packets.jsonl  : per-packet records
- summary.csv    : binned metrics
- summary.json   : overall metrics
- figures/*.png  : plots

Run:
  python ./simulation/sim_dsrc_cv2x.py

Optional args:
  python ./simulation/sim_dsrc_cv2x.py --vehicles 120 --seconds 20 --seed 42
"""

from __future__ import annotations
import datetime
import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Path helpers (ALL under ./simulation)
# ---------------------------

def ensure_dirs(base_dir: str) -> Dict[str, str]:
    outputs = os.path.join(base_dir, "outputs")
    figs = os.path.join(base_dir, "figures")
    os.makedirs(outputs, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    return {"base": base_dir, "outputs": outputs, "figs": figs}


# ---------------------------
# Scenario + PHY/MAC-ish parameters (simplified but trend-faithful)
# ---------------------------

@dataclass
class Scenario:
    road_length_m: float = 2000.0
    lanes: int = 3
    lane_width_m: float = 3.6
    tx_power_dbm: float = 23.0
    noise_floor_dbm: float = -95.0  # effective noise+NF
    carrier_freq_ghz: float = 5.9
    # propagation
    pl0_db: float = 47.86  # approx FSPL at 1m @ 5.9GHz (close enough)
    pathloss_exp_los: float = 2.1
    pathloss_exp_nlos: float = 2.9
    shadow_sigma_los: float = 3.0
    shadow_sigma_nlos: float = 6.0
    nlos_prob_base: float = 0.12
    nlos_prob_slope: float = 0.0025  # grows with distance
    # message properties
    packet_bytes: int = 300  # BSM/CAM-ish
    # channel coherence-ish
    slot_ms: int = 1


@dataclass
class DsrcParams:
    name: str = "DSRC_80211p"
    data_rate_mbps: float = 6.0
    csma_slot_us: int = 13
    cw_min: int = 15
    cw_max: int = 1023
    cca_sense_range_m: float = 600.0  # within this range, node likely senses busy
    hidden_terminal_factor: float = 0.12  # extra collision prob due to hidden nodes (approx)
    retry_limit: int = 2
    processing_delay_ms: float = 1.2  # stack processing / queueing baseline
    # 802.11p tends to degrade sharply with channel load
    congestion_sensitivity: float = 1.35


@dataclass
class Cv2xParams:
    name: str = "C-V2X_Mode4_SPS"
    data_rate_mbps: float = 10.0
    subchannels: int = 25        # simplified total "resources" per 100ms window
    sps_period_ms: int = 100     # semi-persistent schedule period
    reselection_prob: float = 0.18
    harq_retx_prob: float = 0.35  # portion of packets getting a HARQ retx opportunity (simplified)
    processing_delay_ms: float = 1.0
    # Mode 4 resource selection reduces collision under load vs CSMA
    congestion_sensitivity: float = 0.85
    # collision avoidance based on sensed sidelink RSSI (simplified)
    sensing_gain: float = 0.55


# ---------------------------
# Utility functions
# ---------------------------

def dbm_to_mw(dbm: float) -> float:
    return 10 ** (dbm / 10.0)

def mw_to_dbm(mw: float) -> float:
    return 10.0 * math.log10(max(mw, 1e-15))

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def pathloss_db(scn: Scenario, d_m: float, rng: np.random.Generator) -> Tuple[float, bool]:
    """Log-distance PL with LOS/NLOS mixture + shadowing."""
    d = max(d_m, 1.0)
    nlos_prob = clamp(scn.nlos_prob_base + scn.nlos_prob_slope * d, 0.0, 0.85)
    nlos = rng.random() < nlos_prob
    if nlos:
        n = scn.pathloss_exp_nlos
        sigma = scn.shadow_sigma_nlos
    else:
        n = scn.pathloss_exp_los
        sigma = scn.shadow_sigma_los

    shadow = rng.normal(0.0, sigma)
    # PL(d) = PL0 + 10*n*log10(d/1m) + shadow
    pl = scn.pl0_db + 10.0 * n * math.log10(d) + shadow
    return pl, nlos


def sinr_db(
    scn: Scenario,
    rx_power_dbm: float,
    interf_mw: float,
) -> float:
    noise_mw = dbm_to_mw(scn.noise_floor_dbm)
    sig_mw = dbm_to_mw(rx_power_dbm)
    return mw_to_dbm(sig_mw / (noise_mw + interf_mw))


def p_succ_from_sinr(sinr_db_val: float, scheme: str) -> float:
    """
    Map SINR to success probability (soft threshold).
    - DSRC: slightly harsher curve
    - C-V2X: a bit more robust due to coding/receivers (trend)
    """
    # center + steepness tuned for plausible behavior in coursework-level sims
    if scheme.startswith("DSRC"):
        center = 5.5
        steep = 1.15
    else:
        center = 4.5
        steep = 1.05

    # logistic
    p = 1.0 / (1.0 + math.exp(-(sinr_db_val - center) / steep))
    # clamp to avoid extremes
    return clamp(p, 0.001, 0.999)


def tx_time_ms(packet_bytes: int, rate_mbps: float) -> float:
    return (packet_bytes * 8.0) / (rate_mbps * 1e6) * 1000.0


# ---------------------------
# Core simulation
# ---------------------------

@dataclass
class SimConfig:
    vehicles: int = 100
    seconds: int = 15
    msg_rate_hz: float = 10.0  # each vehicle
    seed: int = 1
    # evaluation: each packet targets receivers within range
    comm_range_m: float = 700.0
    # "interference neighborhood" (beyond this, ignore)
    interf_range_m: float = 900.0
    # distance bins for summary
    dist_bin_m: int = 50


def generate_vehicle_positions(scn: Scenario, cfg: SimConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple highway snapshot per time slot:
    x uniform along road, lane uniform.
    """
    x = rng.uniform(0.0, scn.road_length_m, size=cfg.vehicles)
    lane = rng.integers(0, scn.lanes, size=cfg.vehicles)
    y = lane.astype(float) * scn.lane_width_m
    return x, y


def compute_dist(i: int, j: int, x: np.ndarray, y: np.ndarray) -> float:
    dx = x[i] - x[j]
    dy = y[i] - y[j]
    return float(math.hypot(dx, dy))


def simulate_scheme(
    scn: Scenario,
    cfg: SimConfig,
    scheme: str,
    rng: np.random.Generator,
) -> List[dict]:
    """
    Produce per-packet results as dict records.
    A "packet" is a broadcast from one vehicle to all receivers within cfg.comm_range_m.
    We'll record receiver-level events (one record per TX->RX pair per packet).
    """
    dt_ms = scn.slot_ms
    total_slots = int(cfg.seconds * 1000 / dt_ms)

    # precompute tx time
    if scheme.startswith("DSRC"):
        par = DsrcParams()
        rate = par.data_rate_mbps
        base_proc = par.processing_delay_ms
        tx_ms = tx_time_ms(scn.packet_bytes, rate)
    else:
        par = Cv2xParams()
        rate = par.data_rate_mbps
        base_proc = par.processing_delay_ms
        tx_ms = tx_time_ms(scn.packet_bytes, rate)

    # message generation probability per slot
    p_gen = cfg.msg_rate_hz * (dt_ms / 1000.0)

    # For CV2X SPS: each vehicle has a chosen resource (0..subchannels-1), held for sps_period_ms, with reselection prob
    if scheme.startswith("C-V2X"):
        subch = par.subchannels
        sps_slots = int(par.sps_period_ms / dt_ms)
        res = rng.integers(0, subch, size=cfg.vehicles)
        hold = rng.integers(0, sps_slots, size=cfg.vehicles)  # remaining hold time

    records: List[dict] = []

    # simple mobility: refresh positions every 100ms to mimic movement (cheap)
    refresh_slots = max(1, int(100 / dt_ms))
    x, y = generate_vehicle_positions(scn, cfg, rng)

    # track channel busy ratio estimate per scheme per slot (for summary)
    busy_trace = []

    for t in range(total_slots):
        if t % refresh_slots == 0:
            x, y = generate_vehicle_positions(scn, cfg, rng)

        # which vehicles generate a message this slot
        tx_mask = rng.random(cfg.vehicles) < p_gen
        tx_nodes = np.where(tx_mask)[0]
        if tx_nodes.size == 0:
            busy_trace.append(0.0)
            continue

        # update CV2X SPS reservations
        if scheme.startswith("C-V2X"):
            hold -= 1
            reselect = (hold <= 0) | (rng.random(cfg.vehicles) < par.reselection_prob)
            if np.any(reselect):
                # sensing-based: prefer less-used subchannels (simplified)
                # estimate usage by current tx nodes (rough)
                usage = np.bincount(res[tx_nodes], minlength=par.subchannels).astype(float)
                # turn into selection probs: inverse usage
                inv = 1.0 / (1.0 + usage)
                inv = inv / inv.sum()
                for v in np.where(reselect)[0]:
                    res[v] = rng.choice(par.subchannels, p=inv)
                hold[reselect] = sps_slots

        # Build "who transmits concurrently" set:
        # DSRC: CSMA/CA reduces concurrency depending on local sensed busy.
        # CV2X: concurrency is allowed but collisions depend on resource reuse.
        if scheme.startswith("DSRC"):
            # approximate channel load based on tx_nodes density
            offered = tx_nodes.size
            load = offered / max(cfg.vehicles, 1)

            # CSMA reduces simultaneous transmitters:
            # with higher load, fewer pass CCA in same slot (backoff)
            # but hidden terminals add collisions anyway.
            pass_prob = clamp(1.0 - par.congestion_sensitivity * load, 0.05, 0.95)
            actual_tx = tx_nodes[rng.random(tx_nodes.size) < pass_prob]
            # busy ratio approximation: airtime fraction
            busy = clamp((actual_tx.size * tx_ms) / (dt_ms * cfg.vehicles), 0.0, 1.0)
            busy_trace.append(busy)

            # collisions: if multiple tx in proximity, collision probability increases
            # We'll compute receiver-level interference sum from nearby concurrent tx.
            concurrent_tx = actual_tx

        else:
            offered = tx_nodes.size
            load = offered / max(cfg.vehicles, 1)
            # busy ratio: CV2X scheduled, so busy grows but more "structured"
            busy = clamp((offered * tx_ms) / (dt_ms * cfg.vehicles), 0.0, 1.0)
            busy_trace.append(busy)
            concurrent_tx = tx_nodes

        # For each transmitter, for each receiver within range, evaluate success
        for tx in concurrent_tx:
            # Determine set of receivers in comm range (broadcast)
            # (exclude self)
            # We'll do a quick vectorized distance filter for speed
            dx = x - x[tx]
            dy = y - y[tx]
            dist = np.hypot(dx, dy)
            rx_candidates = np.where((dist <= cfg.comm_range_m) & (dist > 0.0))[0]
            if rx_candidates.size == 0:
                continue

            # baseline latency = processing + tx_time + (mac delay)
            if scheme.startswith("DSRC"):
                # mac delay ~ contention/backoff, depends on load and CW
                mac_delay = rng.exponential(scale=1.2 + 8.0 * busy)  # ms
                retries = 0
            else:
                # scheduled: smaller access delay
                mac_delay = rng.exponential(scale=0.7 + 3.0 * busy)
                retries = 0

            base_latency = base_proc + tx_ms + mac_delay

            # Evaluate each receiver
            for rx in rx_candidates:
                d = float(dist[rx])

                # Tx->Rx received power
                pl, nlos = pathloss_db(scn, d, rng)
                rx_p_dbm = scn.tx_power_dbm - pl

                # Interference:
                # sum interference from other concurrent transmitters within interf_range
                interf_mw = 0.0
                if scheme.startswith("DSRC"):
                    # DSRC: everyone shares same channel; collisions more likely
                    # Interference from concurrent tx within interf_range,
                    # plus hidden terminal factor boosts effective interference
                    interferers = concurrent_tx[concurrent_tx != tx]
                    if interferers.size > 0:
                        # compute their distances to receiver
                        dx_i = x[interferers] - x[rx]
                        dy_i = y[interferers] - y[rx]
                        di = np.hypot(dx_i, dy_i)
                        close = interferers[di <= cfg.interf_range_m]
                        if close.size > 0:
                            # each interferer contributes power at rx
                            for itx in close:
                                di_m = compute_dist(int(itx), int(rx), x, y)
                                pl_i, _ = pathloss_db(scn, di_m, rng)
                                p_i_dbm = scn.tx_power_dbm - pl_i
                                interf_mw += dbm_to_mw(p_i_dbm)

                    # hidden terminal "collision-like" penalty:
                    interf_mw *= (1.0 + par.hidden_terminal_factor + 0.6 * busy)

                else:
                    # CV2X: only interferers using same subchannel collide strongly
                    interferers = concurrent_tx[concurrent_tx != tx]
                    if interferers.size > 0:
                        same_res = interferers[res[interferers] == res[tx]]
                        if same_res.size > 0:
                            dx_i = x[same_res] - x[rx]
                            dy_i = y[same_res] - y[rx]
                            di = np.hypot(dx_i, dy_i)
                            close = same_res[di <= cfg.interf_range_m]
                            if close.size > 0:
                                for itx in close:
                                    di_m = compute_dist(int(itx), int(rx), x, y)
                                    pl_i, _ = pathloss_db(scn, di_m, rng)
                                    p_i_dbm = scn.tx_power_dbm - pl_i
                                    interf_mw += dbm_to_mw(p_i_dbm)

                    # sensing-based avoidance reduces effective interference under load
                    interf_mw *= (1.0 - par.sensing_gain * (1.0 - math.exp(-3.0 * busy)))
                    interf_mw = max(interf_mw, 0.0)

                # SINR and success probability
                sinr = sinr_db(scn, rx_p_dbm, interf_mw)
                p_phy = p_succ_from_sinr(sinr, scheme)

                # Congestion impacts:
                # DSRC drops faster with busy ratio; C-V2X less so
                if scheme.startswith("DSRC"):
                    p_cong = math.exp(-par.congestion_sensitivity * 2.4 * busy)
                else:
                    p_cong = math.exp(-par.congestion_sensitivity * 1.9 * busy)

                p_succ = clamp(p_phy * p_cong, 0.0005, 0.999)

                success = (rng.random() < p_succ)
                latency = base_latency

                # Retries / HARQ approximation
                if not success:
                    if scheme.startswith("DSRC"):
                        # limited retries; each retry adds backoff + tx time
                        for _ in range(par.retry_limit):
                            retries += 1
                            extra_backoff = rng.exponential(scale=2.0 + 10.0 * busy)
                            latency += extra_backoff + tx_ms
                            # slight diversity: success prob increases modestly
                            p_try = clamp(p_succ * 1.12, 0.0005, 0.999)
                            if rng.random() < p_try:
                                success = True
                                break
                    else:
                        # HARQ-like: sometimes gets a retx opportunity soon
                        if rng.random() < par.harq_retx_prob:
                            retries += 1
                            latency += rng.exponential(scale=1.2 + 2.0 * busy) + tx_ms
                            p_try = clamp(p_succ * 1.18, 0.0005, 0.999)
                            success = (rng.random() < p_try)

                rec = {
                    "t_ms": t * dt_ms,
                    "scheme": scheme,
                    "tx": int(tx),
                    "rx": int(rx),
                    "dist_m": d,
                    "nlos": bool(nlos),
                    "rx_p_dbm": float(rx_p_dbm),
                    "sinr_db": float(sinr),
                    "busy_ratio": float(busy),
                    "p_succ_model": float(p_succ),
                    "success": bool(success),
                    "latency_ms": float(latency),
                    "retries": int(retries),
                    "pkt_bytes": int(scn.packet_bytes),
                }
                records.append(rec)

    # attach busy ratio series (in case we want it later)
    # (we’ll compute from records anyway, but keep trace here if needed)
    return records


def simulate_scheme_stream(
    scn: Scenario,
    cfg: SimConfig,
    scheme: str,
    rng: np.random.Generator,
    jsonl_fh,
) -> None:
    """
    Stream records to jsonl_fh, instead of returning a huge list in memory.
    Each line is one TX->RX event record (JSON).
    """
    dt_ms = scn.slot_ms
    total_slots = int(cfg.seconds * 1000 / dt_ms)

    if scheme.startswith("DSRC"):
        par = DsrcParams()
        rate = par.data_rate_mbps
        base_proc = par.processing_delay_ms
        tx_ms = tx_time_ms(scn.packet_bytes, rate)
    else:
        par = Cv2xParams()
        rate = par.data_rate_mbps
        base_proc = par.processing_delay_ms
        tx_ms = tx_time_ms(scn.packet_bytes, rate)

    p_gen = cfg.msg_rate_hz * (dt_ms / 1000.0)

    if scheme.startswith("C-V2X"):
        subch = par.subchannels
        sps_slots = int(par.sps_period_ms / dt_ms)
        res = rng.integers(0, subch, size=cfg.vehicles)
        hold = rng.integers(0, sps_slots, size=cfg.vehicles)

    refresh_slots = max(1, int(100 / dt_ms))
    x, y = generate_vehicle_positions(scn, cfg, rng)

    for t in range(total_slots):
        if t % refresh_slots == 0:
            x, y = generate_vehicle_positions(scn, cfg, rng)

        tx_mask = rng.random(cfg.vehicles) < p_gen
        tx_nodes = np.where(tx_mask)[0]
        if tx_nodes.size == 0:
            continue

        # update CV2X SPS
        if scheme.startswith("C-V2X"):
            hold -= 1
            reselect = (hold <= 0) | (rng.random(cfg.vehicles) < par.reselection_prob)
            if np.any(reselect):
                usage = np.bincount(res[tx_nodes], minlength=par.subchannels).astype(float)
                inv = 1.0 / (1.0 + usage)
                inv = inv / inv.sum()
                for v in np.where(reselect)[0]:
                    res[v] = rng.choice(par.subchannels, p=inv)
                hold[reselect] = sps_slots

        # concurrent tx set
        if scheme.startswith("DSRC"):
            offered = tx_nodes.size
            load = offered / max(cfg.vehicles, 1)
            pass_prob = clamp(1.0 - par.congestion_sensitivity * load, 0.05, 0.95)
            concurrent_tx = tx_nodes[rng.random(tx_nodes.size) < pass_prob]
            busy = clamp((concurrent_tx.size * tx_ms) / (dt_ms * cfg.vehicles), 0.0, 1.0)
        else:
            concurrent_tx = tx_nodes
            busy = clamp((tx_nodes.size * tx_ms) / (dt_ms * cfg.vehicles), 0.0, 1.0)

        if concurrent_tx.size == 0:
            continue

        for tx in concurrent_tx:
            dx = x - x[tx]
            dy = y - y[tx]
            dist = np.hypot(dx, dy)
            rx_candidates = np.where((dist <= cfg.comm_range_m) & (dist > 0.0))[0]
            if rx_candidates.size == 0:
                continue

            if scheme.startswith("DSRC"):
                mac_delay = rng.exponential(scale=1.2 + 8.0 * busy)
            else:
                mac_delay = rng.exponential(scale=0.7 + 3.0 * busy)
            base_latency = base_proc + tx_ms + mac_delay

            for rx in rx_candidates:
                d = float(dist[rx])

                pl, nlos = pathloss_db(scn, d, rng)
                rx_p_dbm = scn.tx_power_dbm - pl

                interf_mw = 0.0
                if scheme.startswith("DSRC"):
                    interferers = concurrent_tx[concurrent_tx != tx]
                    if interferers.size > 0:
                        dx_i = x[interferers] - x[rx]
                        dy_i = y[interferers] - y[rx]
                        di = np.hypot(dx_i, dy_i)
                        close = interferers[di <= cfg.interf_range_m]
                        for itx in close:
                            di_m = compute_dist(int(itx), int(rx), x, y)
                            pl_i, _ = pathloss_db(scn, di_m, rng)
                            p_i_dbm = scn.tx_power_dbm - pl_i
                            interf_mw += dbm_to_mw(p_i_dbm)

                    interf_mw *= (1.0 + par.hidden_terminal_factor + 0.6 * busy)

                else:
                    interferers = concurrent_tx[concurrent_tx != tx]
                    if interferers.size > 0:
                        same_res = interferers[res[interferers] == res[tx]]
                        if same_res.size > 0:
                            dx_i = x[same_res] - x[rx]
                            dy_i = y[same_res] - y[rx]
                            di = np.hypot(dx_i, dy_i)
                            close = same_res[di <= cfg.interf_range_m]
                            for itx in close:
                                di_m = compute_dist(int(itx), int(rx), x, y)
                                pl_i, _ = pathloss_db(scn, di_m, rng)
                                p_i_dbm = scn.tx_power_dbm - pl_i
                                interf_mw += dbm_to_mw(p_i_dbm)

                    interf_mw *= (1.0 - par.sensing_gain * (1.0 - math.exp(-3.0 * busy)))
                    interf_mw = max(interf_mw, 0.0)

                sinr = sinr_db(scn, rx_p_dbm, interf_mw)
                p_phy = p_succ_from_sinr(sinr, scheme)

                if scheme.startswith("DSRC"):
                    p_cong = math.exp(-par.congestion_sensitivity * 2.4 * busy)
                else:
                    p_cong = math.exp(-par.congestion_sensitivity * 1.9 * busy)

                p_succ = clamp(p_phy * p_cong, 0.0005, 0.999)
                success = (rng.random() < p_succ)
                latency = base_latency
                retries = 0

                if not success:
                    if scheme.startswith("DSRC"):
                        for _ in range(par.retry_limit):
                            retries += 1
                            latency += rng.exponential(scale=2.0 + 10.0 * busy) + tx_ms
                            p_try = clamp(p_succ * 1.12, 0.0005, 0.999)
                            if rng.random() < p_try:
                                success = True
                                break
                    else:
                        if rng.random() < par.harq_retx_prob:
                            retries += 1
                            latency += rng.exponential(scale=1.2 + 2.0 * busy) + tx_ms
                            p_try = clamp(p_succ * 1.18, 0.0005, 0.999)
                            success = (rng.random() < p_try)

                rec = {
                    "t_ms": t * dt_ms,
                    "scheme": scheme,
                    "tx": int(tx),
                    "rx": int(rx),
                    "dist_m": d,
                    "nlos": bool(nlos),
                    "rx_p_dbm": float(rx_p_dbm),
                    "sinr_db": float(sinr),
                    "busy_ratio": float(busy),
                    "p_succ_model": float(p_succ),
                    "success": bool(success),
                    "latency_ms": float(latency),
                    "retries": int(retries),
                    "pkt_bytes": int(scn.packet_bytes),
                }
                jsonl_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")



# ---------------------------
# Summaries + plots
# ---------------------------

def summarize(records: List[dict], cfg: SimConfig) -> Tuple[pd.DataFrame, dict]:
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return pd.DataFrame(), {}

    # distance bins
    bin_size = cfg.dist_bin_m
    df["dist_bin"] = (df["dist_m"] // bin_size * bin_size).astype(int)

    # metrics by scheme + dist_bin
    g = df.groupby(["scheme", "dist_bin"], as_index=False).agg(
        samples=("success", "size"),
        pdr=("success", "mean"),
        latency_ms_mean=("latency_ms", "mean"),
        latency_ms_p95=("latency_ms", lambda x: float(np.percentile(x, 95))),
        sinr_mean=("sinr_db", "mean"),
        nlos_rate=("nlos", "mean"),
        busy_mean=("busy_ratio", "mean"),
        retries_mean=("retries", "mean"),
    )

    # overall metrics
    overall = {}
    for scheme, sub in df.groupby("scheme"):
        pdr = float(sub["success"].mean())
        lat_mean = float(sub["latency_ms"].mean())
        lat_p95 = float(np.percentile(sub["latency_ms"].to_numpy(), 95))
        busy = float(sub["busy_ratio"].mean())
        sinr = float(sub["sinr_db"].mean())
        nlos = float(sub["nlos"].mean())
        # throughput at receiver-level: successful bytes per second (receiver events)
        succ_bytes = float(sub.loc[sub["success"], "pkt_bytes"].sum())
        duration_s = cfg.seconds
        thr_mbps = succ_bytes * 8.0 / duration_s / 1e6

        overall[scheme] = {
            "receiver_event_count": int(len(sub)),
            "pdr": pdr,
            "latency_ms_mean": lat_mean,
            "latency_ms_p95": lat_p95,
            "busy_ratio_mean": busy,
            "sinr_db_mean": sinr,
            "nlos_rate": nlos,
            "throughput_mbps_success_only": thr_mbps,
        }

    # also add scenario-wide
    meta = {
        "seconds": cfg.seconds,
        "vehicles": cfg.vehicles,
        "msg_rate_hz_per_vehicle": cfg.msg_rate_hz,
        "dist_bin_m": cfg.dist_bin_m,
    }
    summary_json = {"meta": meta, "overall": overall}
    return g, summary_json

def summarize_from_jsonl(jsonl_path: str, cfg: SimConfig) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Read packets.jsonl in chunks, aggregate summary_df + summary_json.
    Also return a sampled dataframe for latency CDF plotting (to avoid loading all).
    """
    bin_size = cfg.dist_bin_m

    # group accumulators: (scheme, dist_bin) -> stats
    acc: Dict[Tuple[str, int], dict] = {}
    overall: Dict[str, dict] = {}

    # sampling for latency CDF (uniform random downsample)
    sample_rows = []
    max_sample = 200000  # enough for smooth CDF, safe memory

    rng = np.random.default_rng(123)

    chunks = pd.read_json(jsonl_path, lines=True, chunksize=200000)
    for chunk in chunks:
        if chunk.empty:
            continue
        chunk["dist_bin"] = (chunk["dist_m"] // bin_size * bin_size).astype(int)

        # sample some rows for CDF plot
        if len(sample_rows) < max_sample:
            # take up to remaining
            remain = max_sample - len(sample_rows)
            take = min(remain, len(chunk))
            # random subset indices
            idx = rng.choice(chunk.index.to_numpy(), size=take, replace=False)
            sample_rows.append(chunk.loc[idx, ["scheme", "latency_ms"]])

        # per-bin aggregation
        for (scheme, dist_bin), sub in chunk.groupby(["scheme", "dist_bin"]):
            key = (scheme, int(dist_bin))
            if key not in acc:
                acc[key] = {
                    "scheme": scheme,
                    "dist_bin": int(dist_bin),
                    "samples": 0,
                    "succ": 0,
                    "lat_sum": 0.0,
                    "sinr_sum": 0.0,
                    "nlos_sum": 0.0,
                    "busy_sum": 0.0,
                    "retries_sum": 0.0,
                    "lat_list": [],  # keep limited for p95 approx
                }
            a = acc[key]
            s = len(sub)
            a["samples"] += s
            a["succ"] += int(sub["success"].sum())
            a["lat_sum"] += float(sub["latency_ms"].sum())
            a["sinr_sum"] += float(sub["sinr_db"].sum())
            a["nlos_sum"] += float(sub["nlos"].sum())
            a["busy_sum"] += float(sub["busy_ratio"].sum())
            a["retries_sum"] += float(sub["retries"].sum())

            # keep only a capped latency list per bin for p95 estimate
            lat_vals = sub["latency_ms"].to_numpy()
            if len(a["lat_list"]) < 20000:
                a["lat_list"].extend(lat_vals[: max(0, 20000 - len(a["lat_list"]))].tolist())

        # overall aggregation
        for scheme, sub in chunk.groupby("scheme"):
            if scheme not in overall:
                overall[scheme] = {
                    "receiver_event_count": 0,
                    "succ": 0,
                    "lat_list": [],
                    "busy_sum": 0.0,
                    "sinr_sum": 0.0,
                    "nlos_sum": 0.0,
                    "succ_bytes": 0.0,
                }
            o = overall[scheme]
            o["receiver_event_count"] += len(sub)
            o["succ"] += int(sub["success"].sum())
            o["busy_sum"] += float(sub["busy_ratio"].sum())
            o["sinr_sum"] += float(sub["sinr_db"].sum())
            o["nlos_sum"] += float(sub["nlos"].sum())
            o["succ_bytes"] += float(sub.loc[sub["success"], "pkt_bytes"].sum())

            # cap overall latency list for p95
            if len(o["lat_list"]) < 300000:
                vals = sub["latency_ms"].to_numpy()
                need = 300000 - len(o["lat_list"])
                o["lat_list"].extend(vals[:need].tolist())

    # build summary_df
    rows = []
    for key, a in acc.items():
        samples = a["samples"]
        if samples == 0:
            continue
        lat_arr = np.array(a["lat_list"], dtype=float) if a["lat_list"] else np.array([np.nan])
        rows.append({
            "scheme": a["scheme"],
            "dist_bin": a["dist_bin"],
            "samples": samples,
            "pdr": a["succ"] / samples,
            "latency_ms_mean": a["lat_sum"] / samples,
            "latency_ms_p95": float(np.nanpercentile(lat_arr, 95)),
            "sinr_mean": a["sinr_sum"] / samples,
            "nlos_rate": a["nlos_sum"] / samples,
            "busy_mean": a["busy_sum"] / samples,
            "retries_mean": a["retries_sum"] / samples,
        })

    summary_df = pd.DataFrame(rows).sort_values(["scheme", "dist_bin"])

    # build summary_json overall
    overall_json = {}
    for scheme, o in overall.items():
        n = o["receiver_event_count"]
        if n == 0:
            continue
        lat_arr = np.array(o["lat_list"], dtype=float) if o["lat_list"] else np.array([np.nan])
        overall_json[scheme] = {
            "receiver_event_count": int(n),
            "pdr": o["succ"] / n,
            "latency_ms_mean": float(np.nanmean(lat_arr)),
            "latency_ms_p95": float(np.nanpercentile(lat_arr, 95)),
            "busy_ratio_mean": o["busy_sum"] / n,
            "sinr_db_mean": o["sinr_sum"] / n,
            "nlos_rate": o["nlos_sum"] / n,
            "throughput_mbps_success_only": (o["succ_bytes"] * 8.0 / cfg.seconds / 1e6),
        }

    summary_json = {
        "meta": {
            "seconds": cfg.seconds,
            "vehicles": cfg.vehicles,
            "msg_rate_hz_per_vehicle": cfg.msg_rate_hz,
            "dist_bin_m": cfg.dist_bin_m,
            "source": "chunked_read_from_packets_jsonl",
        },
        "overall": overall_json,
    }

    # latency sample df
    latency_sample_df = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()
    return summary_df, summary_json, latency_sample_df



def save_jsonl(path: str, records: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def plot_pdr_vs_distance(summary_df: pd.DataFrame, fig_path: str) -> None:
    if summary_df.empty:
        return
    plt.figure()
    for scheme, sub in summary_df.groupby("scheme"):
        sub = sub.sort_values("dist_bin")
        plt.plot(sub["dist_bin"], sub["pdr"], marker="o", label=scheme)
    plt.xlabel("Distance bin start (m)")
    plt.ylabel("PDR (success rate)")
    plt.title("PDR vs Distance")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()


def plot_latency_cdf(records_df: pd.DataFrame, fig_path: str) -> None:
    if records_df.empty:
        return
    plt.figure()
    for scheme, sub in records_df.groupby("scheme"):
        lat = np.sort(sub["latency_ms"].to_numpy())
        y = np.linspace(0.0, 1.0, len(lat), endpoint=True)
        plt.plot(lat, y, label=scheme)
    plt.xlabel("Latency (ms)")
    plt.ylabel("CDF")
    plt.title("Latency CDF (TX->RX events)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()


def plot_busy_vs_density(summary_json: dict, fig_path: str) -> None:
    """
    Density isn't swept here; we approximate "density effect" by mapping busy ratio to vehicles count
    for a single run. Still useful for coursework:
    show the mean busy ratio per scheme.
    """
    if not summary_json:
        return
    schemes = list(summary_json["overall"].keys())
    busy = [summary_json["overall"][s]["busy_ratio_mean"] for s in schemes]

    plt.figure()
    plt.bar(schemes, busy)
    plt.ylabel("Mean channel busy ratio")
    plt.title("Mean Channel Busy Ratio (single scenario)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()


def plot_overall_bars(summary_json: dict, fig_path: str) -> None:
    if not summary_json:
        return
    schemes = list(summary_json["overall"].keys())
    pdr = [summary_json["overall"][s]["pdr"] for s in schemes]
    lat95 = [summary_json["overall"][s]["latency_ms_p95"] for s in schemes]
    thr = [summary_json["overall"][s]["throughput_mbps_success_only"] for s in schemes]

    # three panels in separate figures is cleaner; but user asked "plot/table" not strictly one figure.
    # We'll do one compact figure with 3 bars groups by using repeated bars with different x offsets.
    x = np.arange(len(schemes))
    width = 0.25

    plt.figure()
    plt.bar(x - width, pdr, width=width, label="PDR")
    plt.bar(x, lat95, width=width, label="Latency p95 (ms)")
    plt.bar(x + width, thr, width=width, label="Throughput (Mbps)")
    plt.xticks(x, schemes, rotation=0)
    plt.title("Overall comparison")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()


def print_table(summary_json: dict) -> None:
    if not summary_json:
        print("No results.")
        return
    rows = []
    for scheme, m in summary_json["overall"].items():
        rows.append({
            "scheme": scheme,
            "PDR": round(m["pdr"], 4),
            "latency_mean_ms": round(m["latency_ms_mean"], 3),
            "latency_p95_ms": round(m["latency_ms_p95"], 3),
            "busy_ratio_mean": round(m["busy_ratio_mean"], 4),
            "SINR_mean_dB": round(m["sinr_db_mean"], 3),
            "NLOS_rate": round(m["nlos_rate"], 4),
            "throughput_Mbps": round(m["throughput_mbps_success_only"], 4),
            "rx_events": m["receiver_event_count"],
        })
    df = pd.DataFrame(rows)
    print("\n=== Overall comparison (single scenario) ===")
    print(df.to_string(index=False))


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Simulate DSRC vs C-V2X (Mode 4). Outputs under ./simulation/output/<timestamp>/"
    )
    parser.add_argument("--vehicles", type=int, default=100)
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--msg_rate", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--comm_range", type=float, default=700.0)
    parser.add_argument("--dist_bin", type=int, default=50)
    args = parser.parse_args()

    # timestamp dir
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_dir = os.path.join(".", "simulation", "output", timestamp)
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cfg = SimConfig(
        vehicles=args.vehicles,
        seconds=args.seconds,
        msg_rate_hz=args.msg_rate,
        seed=args.seed,
        comm_range_m=args.comm_range,
        dist_bin_m=args.dist_bin,
    )
    scn = Scenario()

    # config.json
    config = {
        "timestamp": timestamp,
        "cli_args": vars(args),
        "scenario": scn.__dict__,
        "simulation": cfg.__dict__,
        "notes": "Streaming JSONL to avoid OOM/WSL crash under high density.",
    }
    with open(os.path.join(base_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"Run directory: {base_dir}")
    print(f"Running: vehicles={cfg.vehicles}, seconds={cfg.seconds}, msg_rate={cfg.msg_rate_hz}Hz, seed={cfg.seed}")

    # independent rng streams
    rng0 = np.random.default_rng(cfg.seed)
    rng_dsrc = np.random.default_rng(int(rng0.integers(0, 2**31 - 1)))
    rng_cv2x = np.random.default_rng(int(rng0.integers(0, 2**31 - 1)))

    jsonl_path = os.path.join(base_dir, "packets.jsonl")

    # STREAM write jsonl (no huge in-memory list!)
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        print("Simulating DSRC (streaming jsonl) ...")
        simulate_scheme_stream(scn, cfg, "DSRC_80211p", rng_dsrc, fh)

        print("Simulating C-V2X (streaming jsonl) ...")
        simulate_scheme_stream(scn, cfg, "C-V2X_Mode4_SPS", rng_cv2x, fh)

    # Summarize from jsonl in chunks
    summary_df, summary_json, latency_sample_df = summarize_from_jsonl(jsonl_path, cfg)

    csv_path = os.path.join(base_dir, "summary.csv")
    summary_df.to_csv(csv_path, index=False, encoding="utf-8")

    json_path = os.path.join(base_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    # plots (use sampled df for latency cdf, summary_df for pdr)
    plot_pdr_vs_distance(summary_df, os.path.join(fig_dir, "pdr_vs_distance.png"))

    if not latency_sample_df.empty:
        plt.figure()
        for scheme, sub in latency_sample_df.groupby("scheme"):
            lat = np.sort(sub["latency_ms"].to_numpy())
            y = np.linspace(0.0, 1.0, len(lat), endpoint=True)
            plt.plot(lat, y, label=scheme)
        plt.xlabel("Latency (ms)")
        plt.ylabel("CDF")
        plt.title("Latency CDF (sampled from jsonl)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "latency_cdf.png"), dpi=160)
        plt.close()

    plot_busy_vs_density(summary_json, os.path.join(fig_dir, "busy_ratio.png"))
    plot_overall_bars(summary_json, os.path.join(fig_dir, "overall_bars.png"))

    print_table(summary_json)

    print("\nFiles written under:")
    print(f"  {base_dir}")
    print(f"- packets.jsonl")
    print(f"- summary.csv")
    print(f"- summary.json")
    print(f"- config.json")
    print(f"- figures/*.png")



if __name__ == "__main__":
    main()
