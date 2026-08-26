"""
SERENOVA - MPU6050 Processing

RAW accelerometer + gyroscope
        ↓
magnitude calculation
        ↓
movement variability
        ↓
activity representation
"""

from __future__ import annotations

import math
import numpy as np


class IMUProcessor:

    def __init__(self, sampling_rate: int = 50):
        self.sampling_rate = sampling_rate

    def extract_arrays(
        self,
        samples: list[dict],
    ):

        accel = np.array([
            [
                s["accel"]["x"],
                s["accel"]["y"],
                s["accel"]["z"],
            ]
            for s in samples
        ])

        gyro = np.array([
            [
                s["gyro"]["x"],
                s["gyro"]["y"],
                s["gyro"]["z"],
            ]
            for s in samples
        ])

        return accel, gyro

    def process(
        self,
        samples: list[dict],
    ) -> dict:

        if not samples:

            return {
                "signal_available": False,
                "movement_score": None,
                "activity_level": None,
            }

        accel, gyro = self.extract_arrays(
            samples
        )

        # Acceleration magnitude
        accel_magnitude = np.sqrt(
            np.sum(accel ** 2, axis=1)
        )

        # Remove approximately 1g gravity component
        dynamic_acceleration = (
            np.abs(accel_magnitude - 1.0)
        )

        movement_mean = np.mean(
            dynamic_acceleration
        )

        movement_std = np.std(
            dynamic_acceleration
        )

        gyro_magnitude = np.sqrt(
            np.sum(gyro ** 2, axis=1)
        )

        gyro_mean = np.mean(
            np.abs(gyro_magnitude)
        )

        # Convert movement into a bounded feature.
        movement_score = min(
            100.0,
            movement_mean * 500
            + gyro_mean * 2,
        )

        if movement_score < 20:
            activity_level = "Low"

        elif movement_score < 50:
            activity_level = "Moderate"

        else:
            activity_level = "High"

        return {
            "signal_available": True,
            "movement_score": round(
                float(movement_score),
                2,
            ),
            "activity_level": activity_level,
            "acceleration_mean_g": round(
                float(np.mean(accel_magnitude)),
                4,
            ),
            "acceleration_std_g": round(
                float(np.std(accel_magnitude)),
                4,
            ),
            "gyro_mean_dps": round(
                float(gyro_mean),
                4,
            ),
        }


if __name__ == "__main__":

    from simulator.sensors.mpu6050 import (
        MPU6050Simulator,
    )

    simulator = MPU6050Simulator()

    samples = simulator.generate_samples(
        duration_seconds=10
    )

    processor = IMUProcessor(
        sampling_rate=50
    )

    result = processor.process(
        samples
    )

    print("\nMPU6050 PROCESSING RESULT")
    print("=========================")

    for key, value in result.items():
        print(f"{key}: {value}")