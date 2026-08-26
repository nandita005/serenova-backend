"""
SERENOVA - Piezoelectric Fetal Movement Simulator

RAW PIEZO SIGNAL
        ↓
        noise + baseline
        ↓
        fetal movement events
        ↓
    signal processing
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class PiezoConfig:
    sampling_rate_hz: int = 100

    baseline: float = 1800.0

    noise_level: float = 35.0

    movement_probability: float = 0.035

    movement_amplitude: float = 800.0


class PiezoSimulator:

    def __init__(
        self,
        config: PiezoConfig | None = None,
    ):
        self.config = config or PiezoConfig()

        self.sample_index = 0

    def generate_sample(self) -> int:

        fs = self.config.sampling_rate_hz

        # Baseline mechanical/electrical variation
        baseline_variation = (
            25
            * math.sin(
                2 * math.pi * 0.4
                * self.sample_index / fs
            )
        )

        noise = random.gauss(
            0,
            self.config.noise_level,
        )

        value = (
            self.config.baseline
            + baseline_variation
            + noise
        )

        # Occasionally generate a fetal-movement event
        if random.random() < self.config.movement_probability:

            event_amplitude = (
                random.uniform(
                    0.5,
                    1.0,
                )
                * self.config.movement_amplitude
            )

            value += event_amplitude

        self.sample_index += 1

        return max(
            0,
            int(value),
        )

    def generate_samples(
        self,
        duration_seconds: float,
    ) -> list[int]:

        total_samples = int(
            duration_seconds
            * self.config.sampling_rate_hz
        )

        return [
            self.generate_sample()
            for _ in range(total_samples)
        ]


if __name__ == "__main__":

    simulator = PiezoSimulator()

    samples = simulator.generate_samples(
        duration_seconds=10
    )

    print("PIEZO FETAL MOVEMENT SIMULATION")
    print("===============================")

    print(
        f"Sampling rate: "
        f"{simulator.config.sampling_rate_hz} Hz"
    )

    print(
        f"Generated samples: "
        f"{len(samples)}"
    )

    print("\nFirst 30 RAW samples:")

    for sample in samples[:30]:
        print(sample)