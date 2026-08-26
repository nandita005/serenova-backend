"""
SERENOVA - SW-420 Vibration Sensor Simulator

SW-420 is treated as a digital event sensor.

0 = no vibration
1 = vibration detected
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class SW420Config:
    polling_rate_hz: int = 20
    vibration_probability: float = 0.05


class SW420Simulator:

    def __init__(
        self,
        config: SW420Config | None = None,
    ):
        self.config = config or SW420Config()

    def generate_sample(self) -> int:

        if random.random() < self.config.vibration_probability:
            return 1

        return 0

    def generate_samples(
        self,
        duration_seconds: float,
    ) -> list[int]:

        total_samples = int(
            duration_seconds
            * self.config.polling_rate_hz
        )

        return [
            self.generate_sample()
            for _ in range(total_samples)
        ]


if __name__ == "__main__":

    simulator = SW420Simulator()

    samples = simulator.generate_samples(
        duration_seconds=10
    )

    print("SW-420 VIBRATION SIMULATION")
    print("===========================")

    print(
        f"Polling rate: "
        f"{simulator.config.polling_rate_hz} Hz"
    )

    print(
        f"Generated samples: "
        f"{len(samples)}"
    )

    print("\nFirst 50 RAW samples:")

    print(samples[:50])