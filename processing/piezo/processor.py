"""
SERENOVA - Piezo Fetal Movement Processor

RAW PIEZO ADC
      ↓
baseline estimation
      ↓
signal deviation
      ↓
threshold detection
      ↓
movement events
      ↓
movement count + intensity
"""

from __future__ import annotations

import numpy as np


class PiezoProcessor:

    def __init__(
        self,
        sampling_rate: int = 100,
    ):
        self.sampling_rate = sampling_rate

    def remove_baseline(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:

        baseline = np.median(signal)

        return signal - baseline

    def detect_events(
        self,
        signal: np.ndarray,
    ) -> list[int]:

        if len(signal) == 0:
            return []

        # Robust threshold based on signal variation
        baseline = np.median(signal)

        deviation = np.abs(
            signal - baseline
        )

        noise_level = np.median(
            deviation
        )

        threshold = max(
            100.0,
            noise_level * 4.0,
        )

        events = []

        # Prevent several samples from being
        # counted as multiple fetal movements.
        minimum_event_distance = int(
            self.sampling_rate * 0.5
        )

        last_event = (
            -minimum_event_distance
        )

        for i in range(len(signal)):

            if deviation[i] < threshold:
                continue

            if (
                i - last_event
                < minimum_event_distance
            ):
                continue

            events.append(i)

            last_event = i

        return events

    def calculate_intensity(
        self,
        signal: np.ndarray,
        events: list[int],
    ) -> float:

        if not events:
            return 0.0

        baseline = np.median(signal)

        amplitudes = [
            abs(
                float(signal[i])
                - baseline
            )
            for i in events
        ]

        mean_amplitude = np.mean(
            amplitudes
        )

        # Normalize to approximately 0–1
        intensity = min(
            1.0,
            mean_amplitude / 1000.0,
        )

        return round(
            float(intensity),
            3,
        )

    def process(
        self,
        samples: list[int],
    ) -> dict:

        if not samples:

            return {
                "signal_available": False,
                "fetal_movement_count": 0,
                "movement_intensity": 0.0,
            }

        signal = np.asarray(
            samples,
            dtype=float,
        )

        filtered = self.remove_baseline(
            signal
        )

        events = self.detect_events(
            signal
        )

        intensity = (
            self.calculate_intensity(
                signal,
                events,
            )
        )

        return {
            "signal_available": True,

            "fetal_movement_count": len(
                events
            ),

            "movement_intensity": intensity,

            "signal_mean": round(
                float(np.mean(signal)),
                2,
            ),

            "signal_std": round(
                float(np.std(signal)),
                2,
            ),

            "detected_event_indices": events,
        }


if __name__ == "__main__":

    from simulator.sensors.piezo import (
        PiezoSimulator,
    )

    simulator = PiezoSimulator()

    samples = simulator.generate_samples(
        duration_seconds=30
    )

    processor = PiezoProcessor(
        sampling_rate=100
    )

    result = processor.process(
        samples
    )

    print("\nPIEZO PROCESSING RESULT")
    print("=======================")

    for key, value in result.items():
        print(f"{key}: {value}")