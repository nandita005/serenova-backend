"""
SERENOVA - PPG Processing

Converts RAW MAX30102 IR/RED signals into physiological features.

Pipeline:

RAW PPG
  ↓
DC removal
  ↓
smoothing
  ↓
AC component
  ↓
peak detection
  ↓
Heart Rate
  ↓
SpO2 estimation
  ↓
PPG features
"""

from __future__ import annotations

import numpy as np


class PPGProcessor:

    def __init__(self, sampling_rate: int = 100):
        self.sampling_rate = sampling_rate

    # ---------------------------------------------------------
    # 1. Extract raw channels
    # ---------------------------------------------------------

    def extract_channels(
        self,
        samples: list[dict],
    ) -> tuple[np.ndarray, np.ndarray]:

        ir = np.array(
            [sample["ir"] for sample in samples],
            dtype=float,
        )

        red = np.array(
            [sample["red"] for sample in samples],
            dtype=float,
        )

        return ir, red

    # ---------------------------------------------------------
    # 2. Remove DC component
    # ---------------------------------------------------------

    def remove_dc(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:

        return signal - np.mean(signal)

    # ---------------------------------------------------------
    # 3. Moving-average smoothing
    # ---------------------------------------------------------

    def smooth(
        self,
        signal: np.ndarray,
        window_size: int = 5,
    ) -> np.ndarray:

        if len(signal) < window_size:
            return signal

        kernel = np.ones(window_size) / window_size

        return np.convolve(
            signal,
            kernel,
            mode="same",
        )

    # ---------------------------------------------------------
    # 4. Detect cardiac peaks
    # ---------------------------------------------------------

    def detect_peaks(
        self,
        signal: np.ndarray,
    ) -> list[int]:

        if len(signal) < self.sampling_rate:
            return []

        # Minimum distance between peaks.
        #
        # Physiological assumption:
        # HR should not produce beats closer than
        # approximately 0.3 seconds apart.
        min_distance = int(
            self.sampling_rate * 0.30
        )

        threshold = (
            np.mean(signal)
            + 0.5 * np.std(signal)
        )

        peaks = []

        last_peak = -min_distance

        for i in range(1, len(signal) - 1):

            if signal[i] <= threshold:
                continue

            if signal[i] < signal[i - 1]:
                continue

            if signal[i] < signal[i + 1]:
                continue

            if i - last_peak < min_distance:
                continue

            peaks.append(i)
            last_peak = i

        return peaks

    # ---------------------------------------------------------
    # 5. Calculate heart rate
    # ---------------------------------------------------------

    def calculate_heart_rate(
        self,
        peaks: list[int],
    ) -> float | None:

        if len(peaks) < 2:
            return None

        intervals = np.diff(peaks)

        mean_interval = np.mean(intervals)

        if mean_interval <= 0:
            return None

        heart_rate = (
            60
            * self.sampling_rate
            / mean_interval
        )

        return round(float(heart_rate), 2)

    # ---------------------------------------------------------
    # 6. Estimate SpO2
    # ---------------------------------------------------------

    def estimate_spo2(
        self,
        ir: np.ndarray,
        red: np.ndarray,
    ) -> float | None:

        if len(ir) == 0 or len(red) == 0:
            return None

        ir_dc = np.mean(ir)
        red_dc = np.mean(red)

        ir_ac = np.std(ir)
        red_ac = np.std(red)

        if ir_dc <= 0 or red_dc <= 0:
            return None

        if ir_ac <= 0:
            return None

        ratio = (
            (red_ac / red_dc)
            / (ir_ac / ir_dc)
        )

        # Simplified empirical approximation.
        #
        # In a real device this calibration would need
        # validation against reference SpO2 measurements.
        spo2 = 110 - 25 * ratio

        spo2 = max(
            70.0,
            min(100.0, spo2),
        )

        return round(float(spo2), 2)

    # ---------------------------------------------------------
    # 7. Process complete PPG window
    # ---------------------------------------------------------

    def process(
        self,
        samples: list[dict],
    ) -> dict:

        if not samples:
            return {
                "signal_available": False,
                "heart_rate_bpm": None,
                "spo2_percent": None,
                "peak_count": 0,
            }

        ir, red = self.extract_channels(samples)

        ir_ac = self.remove_dc(ir)

        ir_filtered = self.smooth(
            ir_ac,
            window_size=5,
        )

        peaks = self.detect_peaks(
            ir_filtered
        )

        heart_rate = self.calculate_heart_rate(
            peaks
        )

        spo2 = self.estimate_spo2(
            ir,
            red,
        )

        return {
            "signal_available": True,
            "heart_rate_bpm": heart_rate,
            "spo2_percent": spo2,
            "peak_count": len(peaks),
            "ir_mean": round(float(np.mean(ir)), 2),
            "ir_std": round(float(np.std(ir)), 2),
            "red_mean": round(float(np.mean(red)), 2),
            "red_std": round(float(np.std(red)), 2),
        }


# -------------------------------------------------------------
# Standalone test
# -------------------------------------------------------------

if __name__ == "__main__":

    from simulator.sensors.max30102 import (
        MAX30102Simulator,
    )

    simulator = MAX30102Simulator()

    samples = simulator.generate_samples(
        duration_seconds=10
    )

    processor = PPGProcessor(
        sampling_rate=100
    )

    result = processor.process(
        samples
    )

    print("\nPPG PROCESSING RESULT")
    print("=====================")

    for key, value in result.items():
        print(f"{key}: {value}")