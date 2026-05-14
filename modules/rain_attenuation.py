"""
雨衰计算模块 — ITU-R P.838-3 + P.618-13
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Union

Number = Union[float, int, np.ndarray]


# ══════════════════════════════════════════════════════════
#  ITU-R P.838-3 系数表
# ══════════════════════════════════════════════════════════

_KH_A  = [-5.33980, -0.35351, -0.23789, -0.94158]
_KH_B  = [-0.10008,  1.26970,  0.86036,  0.64552]
_KH_C  = [ 1.13098,  0.45400,  0.15354,  0.16817]
_KH_MK = -0.18961
_KH_CK =  0.71147

_KV_A  = [-3.80595, -3.44965, -0.39902,  0.50167]
_KV_B  = [ 0.56934, -0.22911,  0.73042,  1.07319]
_KV_C  = [ 0.81061,  0.51059,  0.11899,  0.27195]
_KV_MK = -0.16398
_KV_CK =  0.63297

_AH_A  = [-0.14318,  0.29591,  0.32177, -5.37610, 16.1721]
_AH_B  = [ 1.82442,  0.77564,  0.63773, -0.96230, -3.29980]
_AH_C  = [-0.55187,  0.19822,  0.13164,  1.47828,  3.43990]
_AH_MA =  0.67849
_AH_CA = -1.95537

_AV_A  = [-0.07771,  0.56727, -0.20238, -48.2991,  48.5833]
_AV_B  = [ 2.33840,  0.95545,  1.14520,  0.791669,  0.791459]
_AV_C  = [-0.76284,  0.54039,  0.26809,  0.116226,  0.116479]
_AV_MA = -0.053739
_AV_CA =  0.83433


# ══════════════════════════════════════════════════════════
#  P.838-3 核心计算
# ══════════════════════════════════════════════════════════

def _calc_k(freq_ghz, a, b, c, mk, ck):
    lgf = math.log10(freq_ghz)
    s = sum(aj * math.exp(-((lgf - bj) / cj) ** 2)
            for aj, bj, cj in zip(a, b, c))
    return 10 ** (s + mk * lgf + ck)


def _calc_alpha(freq_ghz, a, b, c, ma, ca):
    lgf = math.log10(freq_ghz)
    s = sum(aj * math.exp(-((lgf - bj) / cj) ** 2)
            for aj, bj, cj in zip(a, b, c))
    return s + ma * lgf + ca


def calc_p838_coeffs(freq_ghz):
    """返回 (k_H, alpha_H, k_V, alpha_V)"""
    kH = _calc_k(freq_ghz,     _KH_A, _KH_B, _KH_C, _KH_MK, _KH_CK)
    aH = _calc_alpha(freq_ghz, _AH_A, _AH_B, _AH_C, _AH_MA, _AH_CA)
    kV = _calc_k(freq_ghz,     _KV_A, _KV_B, _KV_C, _KV_MK, _KV_CK)
    aV = _calc_alpha(freq_ghz, _AV_A, _AV_B, _AV_C, _AV_MA, _AV_CA)
    return kH, aH, kV, aV


def calc_specific_attenuation(freq_ghz, rain_rate, elevation_deg, polarization_deg):
    """
    雨比衰减 γ_R (dB/km)，使用 P.838-3 完整极化合并公式。

    完整公式（P.838-3 式 4 & 5，含仰角 θ 和极化倾角 τ）：
        k = (k_H + k_V + (k_H - k_V)·cos²θ·cos2τ) / 2
        α = (k_H·α_H + k_V·α_V + (k_H·α_H - k_V·α_V)·cos²θ·cos2τ) / (2k)

    参数
    ----
    elevation_deg    : 链路仰角 θ (°)
    polarization_deg : 极化倾角 τ (°)，水平=0°，垂直=90°，圆极化=45°

    返回 (gamma_R, k, alpha)
    """
    rain_rate = np.atleast_1d(np.asarray(rain_rate, dtype=float))

    kH, aH, kV, aV = calc_p838_coeffs(freq_ghz)

    theta = math.radians(elevation_deg)
    tau   = math.radians(polarization_deg)

    cos2_theta = math.cos(theta) ** 2
    cos_2tau   = math.cos(2 * tau)

    # P.838-3 完整公式
    k = (kH + kV + (kH - kV) * cos2_theta * cos_2tau) / 2.0

    if k > 0:
        alpha = (kH * aH + kV * aV +
                 (kH * aH - kV * aV) * cos2_theta * cos_2tau) / (2.0 * k)
    else:
        alpha = (aH + aV) / 2.0

    gamma_R = k * rain_rate ** alpha
    return gamma_R, k, alpha


# ══════════════════════════════════════════════════════════
#  P.618-13 有效路径长度
# ══════════════════════════════════════════════════════════

def calc_effective_path_length(rain_height_km, station_altitude_km,
                               elevation_deg, freq_ghz, rain_rate):
    
    rain_rate = np.atleast_1d(np.asarray(rain_rate, dtype=float))
    theta = math.radians(elevation_deg)
    delta_h = rain_height_km - station_altitude_km
    if delta_h <= 0:
        return np.zeros_like(rain_rate)
    # 核心：斜路径长度 = 垂直高度差 / 正弦仰角
    L_eff = delta_h * math.sin(theta)
    return np.full_like(rain_rate, L_eff)


# ══════════════════════════════════════════════════════════
#  结果 dataclass
# ══════════════════════════════════════════════════════════

@dataclass
class RainAttenuationResult:
    freq_ghz:         float
    rain_rate:        np.ndarray
    rain_height_km:   float
    elevation_deg:    float
    polarization_deg: float

    kH: float;  alphaH: float
    kV: float;  alphaV: float
    k:  float;  alpha:  float   # 完整公式合并后的等效值

    gamma_R: np.ndarray   # 等效雨比衰减 dB/km
    L_eff:   np.ndarray   # 有效路径长度 km
    A_rain:  np.ndarray   # 总雨衰 dB


# ══════════════════════════════════════════════════════════
#  主计算入口
# ══════════════════════════════════════════════════════════

def compute_rain_attenuation(
    freq_ghz:         float,
    rain_rate:        Number,
    rain_height_km:   float,
    elevation_deg:    float = 10.0,
    station_alt_km:   float = 0.0,
    polarization_deg: float = 45.0,
) -> RainAttenuationResult:
    """
    计算链路总雨衰（ITU-R P.838-3 + P.618-13）。

    雨顶高度参考值（ITU-R P.839-4 格点数据）：
      北京（40°N, 116°E）: H_R ≈ 4.0 km
      上海（31°N, 121°E）: H_R ≈ 4.5 km
      广州（23°N, 113°E）: H_R ≈ 5.0 km
    """
    rain_rate = np.atleast_1d(np.asarray(rain_rate, dtype=float))

    kH, aH, kV, aV = calc_p838_coeffs(freq_ghz)

    # 完整极化合并公式（含仰角）
    gamma_R, k, alpha = calc_specific_attenuation(
        freq_ghz, rain_rate, elevation_deg, polarization_deg)

    L_eff  = calc_effective_path_length(
        rain_height_km, station_alt_km, elevation_deg, freq_ghz, rain_rate)

    A_rain = gamma_R * L_eff

    return RainAttenuationResult(
        freq_ghz         = freq_ghz,
        rain_rate        = rain_rate,
        rain_height_km   = rain_height_km,
        elevation_deg    = elevation_deg,
        polarization_deg = polarization_deg,
        kH=kH, alphaH=aH,
        kV=kV, alphaV=aV,
        k=k,   alpha=alpha,
        gamma_R = gamma_R,
        L_eff   = L_eff,
        A_rain  = A_rain,
    )


def rain_attenuation_db(
    freq_ghz:         float,
    rain_rate_mmh:    float,
    rain_height_km:   float,
    elevation_deg:    float = 10.0,
    station_alt_km:   float = 0.0,
    polarization_deg: float = 45.0,
) -> float:
    """单点雨衰查询，返回标量 dB。供链路预算表格直接调用。"""
    res = compute_rain_attenuation(
        freq_ghz, rain_rate_mmh, rain_height_km,
        elevation_deg, station_alt_km, polarization_deg)
    return float(res.A_rain[0])


# ══════════════════════════════════════════════════════════
#  自测
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("P.838-3 系数（40/50 GHz）")
    for f in [40, 50]:
        kH, aH, kV, aV = calc_p838_coeffs(f)
        print(f"  {f} GHz: kH={kH:.5f} αH={aH:.5f} kV={kV:.5f} αV={aV:.5f}")

    print("\n总路径雨衰（北京典型参数：仰角10°，雨高4.0km，圆极化45°）")
    for f in [40, 50]:
        res = compute_rain_attenuation(f, [5, 15, 30, 50], 4.0, 10.0, 0.0, 45.0)
        print(f"\n  {f} GHz:")
        for i, R in enumerate([5, 15, 30, 50]):
            print(f"    R={R:2d}mm/h  γ={res.gamma_R[i]:.4f}dB/km"
                  f"  L_eff={res.L_eff[i]:.3f}km"
                  f"  A={res.A_rain[i]:.3f}dB")