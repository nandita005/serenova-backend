"""
SERENOVA - Maternal Feature Representation
"""

from __future__ import annotations


def build_maternal_features(
    ppg: dict,
    ecg: dict,
    imu: dict,
    temperature: dict,
) -> dict:

    return {
        "heart_rate_bpm": ppg.get(
            "heart_rate_bpm"
        ),

        "spo2_percent": ppg.get(
            "spo2_percent"
        ),

        "ecg_signal_available": ecg.get(
            "signal_available",
            False,
        ),

        "ecg_heart_rate_bpm": ecg.get(
            "heart_rate_bpm"
        ),

        "sdnn_ms": ecg.get(
            "sdnn_ms"
        ),

        "rmssd_ms": ecg.get(
            "rmssd_ms"
        ),

        "activity_score": imu.get(
            "movement_score"
        ),

        "activity_level": imu.get(
            "activity_level"
        ),

        "temperature_c": temperature.get(
            "temperature_c"
        ),

        "temperature_mean_c": temperature.get(
            "temperature_mean_c"
        ),

        "temperature_std_c": temperature.get(
            "temperature_std_c"
        ),

        "temperature_trend": temperature.get(
            "temperature_trend_c_per_sample"
        ),
    }