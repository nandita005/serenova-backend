"""
SERENOVA - MAX30102 PPG Sensor Simulator

Simulates the RAW output of the MAX30102:
    - IR PPG signal
    - RED PPG signal

The simulator does NOT directly generate clinical risk.

Pipeline:

    Synthetic physiological state
            ↓
        PPG waveform
            ↓
       IR / RED ADC values
            ↓
    PPG processing (later)
            ↓
       HR / SpO2
            ↓
      Feature engineering
            ↓
         AI models
"""

import math
import random
from dataclasses import dataclass


@dataclass
class MAX30102Config:
    """
    Configuration representing the sampling behaviour
    of the simulated MAX30102.
    """

    sampling_rate_hz: int = 100

    baseline_ir: int = 12000
    baseline_red: int = 10000

    heart_rate_bpm: float = 82.0

    noise_level: float = 35.0


class MAX30102Simulator:
    """
    Synthetic MAX30102 PPG signal generator.

    Produces RAW IR and RED ADC-like values.

    No clinical interpretation happens here.
    """

    def __init__(self, config: MAX30102Config | None = None):

        self.config = config or MAX30102Config()

        self.sample_index = 0

        self.phase = 0.0

        self.samples_per_beat = (
            self.config.sampling_rate_hz
            * 60.0
            / self.config.heart_rate_bpm
        )

    def _cardiac_wave(self, phase: float) -> float:
        """
        Generate a simplified pulsatile PPG morphology.

        This is intentionally not a pure sine wave.
        It contains:
            - systolic peak
            - secondary wave
            - baseline variation
        """

        phase = phase % 1.0

        # Main systolic pulse
        systolic = math.exp(
            -((phase - 0.18) ** 2) / 0.0025
        )

        # Secondary / dicrotic component
        secondary = 0.25 * math.exp(
            -((phase - 0.48) ** 2) / 0.008
        )

        return systolic + secondary

    def _respiratory_baseline(self, time_seconds: float) -> float:
        """
        Low-frequency respiratory modulation.
        """

        respiration_rate_hz = 0.25  # ~15 breaths/min

        return 80.0 * math.sin(
            2.0
            * math.pi
            * respiration_rate_hz
            * time_seconds
        )

    def generate_sample(self) -> dict:
        """
        Generate ONE raw MAX30102 sample.
        """

        fs = self.config.sampling_rate_hz

        time_seconds = self.sample_index / fs

        phase = (
            self.sample_index
            / self.samples_per_beat
        )

        cardiac = self._cardiac_wave(phase)

        respiratory = self._respiratory_baseline(
            time_seconds
        )

        noise_ir = random.gauss(
            0,
            self.config.noise_level
        )

        noise_red = random.gauss(
            0,
            self.config.noise_level
        )

        # IR generally has a stronger pulsatile component
        ir = (
            self.config.baseline_ir
            + respiratory
            + cardiac * 850
            + noise_ir
        )

        # RED has a different amplitude
        red = (
            self.config.baseline_red
            + respiratory * 0.6
            + cardiac * 450
            + noise_red
        )

        self.sample_index += 1

        return {
            "ir": int(ir),
            "red": int(red),
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

    simulator = MAX30102Simulator()

    samples = simulator.generate_samples(
        duration_seconds=5
    )

    print("MAX30102 SIMULATION")
    print("===================")

    print(
        f"Sampling rate: "
        f"{simulator.config.sampling_rate_hz} Hz"
    )

    print(
        f"Configured HR: "
        f"{simulator.config.heart_rate_bpm} BPM"
    )

    print(
        f"Generated samples: "
        f"{len(samples)}"
    )

    print("\nFirst 10 RAW samples:")

    for sample in samples[:10]:
        print(sample)