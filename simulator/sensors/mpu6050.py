"""
SERENOVA - MPU6050 Sensor Simulator

Simulates RAW MPU6050 accelerometer and gyroscope output.

Pipeline:

RAW acceleration + gyroscope
        ↓
motion preprocessing
        ↓
magnitude / variability / activity features
        ↓
maternal activity representation
        ↓
AI feature layer
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class MPU6050Config:
    sampling_rate_hz: int = 50

    # Resting orientation
    gravity: float = 1.0

    noise_level: float = 0.02

    # Small natural body movement
    movement_amplitude: float = 0.08

    gyro_amplitude: float = 2.0


class MPU6050Simulator:

    def __init__(
        self,
        config: MPU6050Config | None = None,
    ):

        self.config = config or MPU6050Config()

        self.sample_index = 0

    def generate_sample(self) -> dict:

        fs = self.config.sampling_rate_hz

        t = self.sample_index / fs

        # --------------------------------------------------
        # Accelerometer
        # --------------------------------------------------

        movement_x = (
            self.config.movement_amplitude
            * math.sin(2 * math.pi * 0.5 * t)
        )

        movement_y = (
            self.config.movement_amplitude
            * 0.7
            * math.sin(2 * math.pi * 0.35 * t)
        )

        accel_x = (
            movement_x
            + random.gauss(
                0,
                self.config.noise_level,
            )
        )

        accel_y = (
            movement_y
            + random.gauss(
                0,
                self.config.noise_level,
            )
        )

        accel_z = (
            self.config.gravity
            + random.gauss(
                0,
                self.config.noise_level,
            )
        )

        # --------------------------------------------------
        # Gyroscope
        # --------------------------------------------------

        gyro_x = (
            self.config.gyro_amplitude
            * math.sin(2 * math.pi * 0.30 * t)
            + random.gauss(0, 0.15)
        )

        gyro_y = (
            self.config.gyro_amplitude
            * 0.7
            * math.sin(2 * math.pi * 0.20 * t)
            + random.gauss(0, 0.15)
        )

        gyro_z = (
            self.config.gyro_amplitude
            * 0.5
            * math.sin(2 * math.pi * 0.15 * t)
            + random.gauss(0, 0.15)
        )

        self.sample_index += 1

        return {
            "accel": {
                "x": round(accel_x, 4),
                "y": round(accel_y, 4),
                "z": round(accel_z, 4),
            },
            "gyro": {
                "x": round(gyro_x, 4),
                "y": round(gyro_y, 4),
                "z": round(gyro_z, 4),
            },
        }

    def generate_samples(
        self,
        duration_seconds: float,
    ) -> list[dict]:

        total_samples = int(
            duration_seconds
            * self.config.sampling_rate_hz
        )

        return [
            self.generate_sample()
            for _ in range(total_samples)
        ]


if __name__ == "__main__":

    simulator = MPU6050Simulator()

    samples = simulator.generate_samples(
        duration_seconds=5
    )

    print("MPU6050 SIMULATION")
    print("==================")

    print(
        f"Sampling rate: "
        f"{simulator.config.sampling_rate_hz} Hz"
    )

    print(
        f"Generated samples: "
        f"{len(samples)}"
    )

    print("\nFirst 10 RAW samples:")

    for sample in samples[:10]:
        print(sample)