"""
SERENOVA - ECG Signal Processor

RAW AD8232 ECG
    ↓
Baseline removal
    ↓
Smoothing
    ↓
R-peak detection
    ↓
RR intervals
    ↓
Heart-rate / HRV features
"""

from __future__ import annotations

import numpy as np


class ECGProcessor:

    def __init__(self, sampling_rate: int = 250):
        self.sampling_rate = sampling_rate

    def preprocess(
        self,
        samples: list[int],
    ) -> np.ndarray:

        signal = np.asarray(
            samples,
            dtype=float,
        )

        if len(signal) == 0:
            return signal

        # Remove DC / ADC offset
        signal = signal - np.mean(signal)

        # Simple moving-average smoothing
        window = 5

        if len(signal) >= window:
            kernel = np.ones(window) / window

            signal = np.convolve(
                signal,
                kernel,
                mode="same",
            )

        return signal

    def detect_r_peaks(
        self,
        signal: np.ndarray,
    ) -> list[int]:

        if len(signal) < self.sampling_rate:
            return []

        # Approximate physiological refractory period.
        # Prevents multiple detections around one QRS complex.
        min_distance = int(
            self.sampling_rate * 0.30
        )

        threshold = (
            np.mean(signal)
            + 2.0 * np.std(signal)
        )

        peaks = []

        last_peak = -min_distance

        for i in range(
            1,
            len(signal) - 1,
        ):

            if signal[i] < threshold:
                continue

            if signal[i] < signal[i - 1]:
                continue

            if signal[i] < signal[i + 1]:
                continue

            if (
                i - last_peak
                < min_distance
            ):
                continue

            peaks.append(i)

            last_peak = i

        return peaks

    def calculate_rr_intervals(
        self,
        peaks: list[int],
    ) -> np.ndarray:

        if len(peaks) < 2:
            return np.array([])

        peak_intervals = np.diff(peaks)

        rr_seconds = (
            peak_intervals
            / self.sampling_rate
        )

        return rr_seconds

    def calculate_heart_rate(
        self,
        rr_intervals: np.ndarray,
    ) -> float | None:

        if len(rr_intervals) == 0:
            return None

        mean_rr = np.mean(rr_intervals)

        if mean_rr <= 0:
            return None

        heart_rate = 60.0 / mean_rr

        return round(
            float(heart_rate),
            2,
        )

    def calculate_hrv(
        self,
        rr_intervals: np.ndarray,
    ) -> dict:

        if len(rr_intervals) < 2:

            return {
                "sdnn_ms": None,
                "rmssd_ms": None,
            }

        rr_ms = rr_intervals * 1000.0

        # SDNN
        sdnn = np.std(
            rr_ms,
            ddof=1,
        )

        # RMSSD
        differences = np.diff(
            rr_ms
        )

        rmssd = np.sqrt(
            np.mean(
                differences ** 2
            )
        )

        return {
            "sdnn_ms": round(
                float(sdnn),
                2,
            ),
            "rmssd_ms": round(
                float(rmssd),
                2,
            ),
        }

    def process(
        self,
        samples: list[int],
    ) -> dict:

        if not samples:

            return {
                "signal_available": False,
                "heart_rate_bpm": None,
                "r_peak_count": 0,
                "rr_intervals_ms": [],
                "sdnn_ms": None,
                "rmssd_ms": None,
            }

        signal = self.preprocess(
            samples
        )

        peaks = self.detect_r_peaks(
            signal
        )

        rr_intervals = (
            self.calculate_rr_intervals(
                peaks
            )
        )

        heart_rate = (
            self.calculate_heart_rate(
                rr_intervals
            )
        )

        hrv = self.calculate_hrv(
            rr_intervals
        )

        return {
            "signal_available": True,

            "heart_rate_bpm": heart_rate,

            "r_peak_count": len(peaks),

            "rr_intervals_ms": [
                round(
                    float(rr * 1000),
                    2,
                )
                for rr in rr_intervals
            ],

            "sdnn_ms": hrv["sdnn_ms"],

            "rmssd_ms": hrv["rmssd_ms"],
        }


if __name__ == "__main__":

    from simulator.sensors.ad8232 import (
        AD8232Simulator,
    )

    simulator = AD8232Simulator()

    samples = simulator.generate_samples(
        duration_seconds=10
    )

    processor = ECGProcessor(
        sampling_rate=250
    )

    result = processor.process(
        samples
    )

    print("\nECG PROCESSING RESULT")
    print("=====================")

    for key, value in result.items():
        print(f"{key}: {value}")