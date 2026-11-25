"""Downlink budget with DVB-S2 adaptive coding and modulation (ACM).

C/N0 = EIRP - FSPL - L_atm(el) - L_misc + G/T - k
Es/N0 = C/N0 - 10 log10(Rs)

The highest-efficiency MODCOD whose required Es/N0 (plus an implementation
margin) is met sets the instantaneous information rate Rs * eta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import BOLTZMANN_DBW, C_LIGHT_KM_S

# DVB-S2 MODCODs (ETSI EN 302 307-1): name, ideal Es/N0 [dB] (AWGN, normal frame), spectral efficiency [bit/symbol]
MODCODS: tuple[tuple[str, float, float], ...] = (
    ("QPSK 1/4", -2.35, 0.490),
    ("QPSK 1/3", -1.24, 0.656),
    ("QPSK 2/5", -0.30, 0.789),
    ("QPSK 1/2", 1.00, 0.988),
    ("QPSK 3/5", 2.23, 1.188),
    ("QPSK 2/3", 3.10, 1.322),
    ("QPSK 3/4", 4.03, 1.487),
    ("QPSK 4/5", 4.68, 1.587),
    ("QPSK 5/6", 5.18, 1.655),
    ("8PSK 3/5", 5.50, 1.779),
    ("QPSK 8/9", 6.20, 1.767),
    ("QPSK 9/10", 6.42, 1.789),
    ("8PSK 2/3", 6.62, 1.980),
    ("8PSK 3/4", 7.91, 2.228),
    ("16APSK 2/3", 8.97, 2.637),
    ("8PSK 5/6", 9.35, 2.479),
    ("16APSK 3/4", 10.21, 2.966),
    ("8PSK 8/9", 10.69, 2.646),
    ("8PSK 9/10", 10.98, 2.679),
    ("16APSK 4/5", 11.03, 3.165),
    ("16APSK 5/6", 11.61, 3.300),
    ("16APSK 8/9", 12.89, 3.523),
    ("16APSK 9/10", 13.13, 3.567),
)

_THRESHOLDS = np.array([m[1] for m in MODCODS])
_EFFICIENCY = np.array([m[2] for m in MODCODS])
_ORDER = np.argsort(_THRESHOLDS)
_SORTED_THR = _THRESHOLDS[_ORDER]
# Best achievable efficiency for any Es/N0 >= threshold (upper envelope).
_ENVELOPE = np.maximum.accumulate(_EFFICIENCY[_ORDER])
_ENVELOPE_IDX = np.array([_ORDER[: k + 1][np.argmax(_EFFICIENCY[_ORDER[: k + 1]])] for k in range(len(_ORDER))])


def free_space_path_loss_db(range_km, frequency_hz: float):
    """FSPL = 20 log10(4 pi d f / c) [dB]."""
    d_m = np.asarray(range_km, dtype=float) * 1000.0
    return 20.0 * np.log10(d_m) + 20.0 * np.log10(frequency_hz) - 147.55


def doppler_shift_hz(range_rate_km_s, frequency_hz: float):
    """Received frequency offset [Hz]; positive while the satellite approaches."""
    return -frequency_hz * np.asarray(range_rate_km_s, dtype=float) / C_LIGHT_KM_S


@dataclass(frozen=True)
class LinkBudget:
    frequency_hz: float = 8.2e9  # X-band downlink
    eirp_dbw: float = 9.0  # e.g. 2 W transmitter into a ~6 dBi patch
    symbol_rate_sps: float = 25e6
    implementation_margin_db: float = 3.0
    zenith_atmospheric_loss_db: float = 0.4
    misc_losses_db: float = 1.0  # pointing, polarisation, radome

    def cn0_dbhz(self, range_km, elevation_rad, g_over_t_db: float):
        sin_el = np.maximum(np.sin(np.asarray(elevation_rad, dtype=float)), np.sin(np.deg2rad(3.0)))
        atmospheric = self.zenith_atmospheric_loss_db / sin_el
        return (
            self.eirp_dbw
            - free_space_path_loss_db(range_km, self.frequency_hz)
            - atmospheric
            - self.misc_losses_db
            + g_over_t_db
            - BOLTZMANN_DBW
        )

    def esn0_db(self, range_km, elevation_rad, g_over_t_db: float):
        return self.cn0_dbhz(range_km, elevation_rad, g_over_t_db) - 10.0 * np.log10(self.symbol_rate_sps)

    def select_modcod(self, esn0_db):
        """Index into MODCODS of the best usable MODCOD (-1 if the link is closed)."""
        usable = np.asarray(esn0_db, dtype=float) - self.implementation_margin_db
        k = np.searchsorted(_SORTED_THR, usable, side="right") - 1
        return np.where(k >= 0, _ENVELOPE_IDX[np.clip(k, 0, None)], -1)

    def data_rate_bps(self, range_km, elevation_rad, g_over_t_db: float):
        idx = self.select_modcod(self.esn0_db(range_km, elevation_rad, g_over_t_db))
        eff = np.where(idx >= 0, _EFFICIENCY[np.clip(idx, 0, None)], 0.0)
        return eff * self.symbol_rate_sps
