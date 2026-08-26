"""
SERENOVA - DS18B20 Temperature Sensor Simulator

Simulates the digital temperature reading produced by
the DS18B20.

RAW SENSOR
    ↓
Temperature reading (°C)
    ↓
Temporal processing
    ↓
Temperature trend / deviation
    ↓
AI feature layer
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class DS18B20Config:
    sampling_interval_seconds: float = 5.0

    baseline_temperature_c: float = 36.7

    noise_level: float = 0.03

    slow_variation_amplitude: float = 0.08


class DS18B20Simulator:

    def __init__(
        self,
        config: DS18B20Config | None = None,
    ):

        self.config = config or DS18B20Config()

        self.sample_index = 0

    def generate_sample(self) -> float:

        t = (
            self.sample_index
            * self.config.sampling_interval_seconds
        )

        # Slow physiological/environmental variation
        variation = (
            self.config.slow_variation_amplitude
            * math.sin(
                2
                * math.pi
                * t
                / 1800
            )
        )

        noise = random.gauss(
            0,
            self.config.noise_level,
        )

        temperature = (
            self.config.baseline_temperature_c
            + variation
            + noise
        )

        self.sample_index += 1

        return round(
            temperature,
            3,
        )

    def generate_samples(
        self,
        duration_seconds: float,
    ) -> list[float]:

        total_samples = int(
            duration_seconds
            / self.config.sampling_interval_seconds
        )

        return [
            self.generate_sample()
            for _ in range(total_samples)
        ]


if __name__ == "__main__":

    simulator = DS18B20Simulator()

    samples = simulator.generate_samples(
        duration_seconds=60
    )

    print("DS18B20 SIMULATION")
    print("==================")

    print(
        f"Sampling interval: "
        f"{simulator.config.sampling_interval_seconds} seconds"
    )

    print(
        f"Generated samples: "
        f"{len(samples)}"
    )

    print("\nTemperature samples:")

    for value in samples:
        print(f"{value} °C")