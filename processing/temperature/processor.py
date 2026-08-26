"""
SERENOVA - Temperature Processing
"""

from __future__ import annotations

import numpy as np


class TemperatureProcessor:

    def process(
        self,
        samples: list[float],
    ) -> dict:

        if not samples:

            return {
                "signal_available": False,
                "temperature_c": None,
                "temperature_mean_c": None,
                "temperature_trend_c_per_sample": None,
            }

        values = np.asarray(
            samples,
            dtype=float,
        )

        current_temperature = values[-1]

        mean_temperature = np.mean(values)

        # Linear trend across the window
        if len(values) >= 2:

            x = np.arange(len(values))

            slope = np.polyfit(
                x,
                values,
                1,
            )[0]

        else:
            slope = 0.0

        return {
            "signal_available": True,

            "temperature_c": round(
                float(current_temperature),
                3,
            ),

            "temperature_mean_c": round(
                float(mean_temperature),
                3,
            ),

            "temperature_std_c": round(
                float(np.std(values)),
                4,
            ),

            "temperature_trend_c_per_sample": round(
                float(slope),
                6,
            ),
        }


if __name__ == "__main__":

    from simulator.sensors.ds18b20 import (
        DS18B20Simulator,
    )

    simulator = DS18B20Simulator()

    samples = simulator.generate_samples(
        duration_seconds=300
    )

    processor = TemperatureProcessor()

    result = processor.process(
        samples
    )

    print("\nTEMPERATURE PROCESSING RESULT")
    print("=============================")

    for key, value in result.items():
        print(f"{key}: {value}")