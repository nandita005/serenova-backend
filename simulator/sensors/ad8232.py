"""
SERENOVA - AD8232 ECG Sensor Simulator

Generates a synthetic ECG waveform representing the
RAW ADC output of the AD8232.

Pipeline:

Synthetic cardiac activity
        ↓
ECG waveform
        ↓
ADC-like samples
        ↓
ECG processor
        ↓
R-peaks / RR intervals / HR / HRV
        ↓
AI feature layer
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class AD8232Config:
    sampling_rate_hz: int = 250
    heart_rate_bpm: float = 82.0

    # ADC characteristics
    adc_center: int = 2048
    signal_amplitude: float = 450.0

    noise_level: float = 12.0


class AD8232Simulator:

    def __init__(
        self,
        config: AD8232Config | None = None,
    ):

        self.config = config or AD8232Config()

        self.sample_index = 0

        self.samples_per_beat = (
            self.config.sampling_rate_hz
            * 60.0
            / self.config.heart_rate_bpm
        )

    def _ecg_waveform(
        self,
        phase: float,
    ) -> float:
        """
        Simplified ECG morphology.

        Contains approximate:
            P wave
            Q wave
            R wave
            S wave
            T wave
        """

        phase = phase % 1.0

        # P wave
        p = 0.12 * math.exp(
            -((phase - 0.12) ** 2) / 0.0015
        )

        # Q wave
        q = -0.15 * math.exp(
            -((phase - 0.20) ** 2) / 0.0004
        )

        # R wave
        r = 1.00 * math.exp(
            -((phase - 0.22) ** 2) / 0.00025
        )

        # S wave
        s = -0.25 * math.exp(
            -((phase - 0.25) ** 2) / 0.0005
        )

        # T wave
        t = 0.30 * math.exp(
            -((phase - 0.42) ** 2) / 0.004
        )

        return p + q + r + s + t

    def generate_sample(self) -> int:

        fs = self.config.sampling_rate_hz

        time_seconds = self.sample_index / fs

        phase = (
            self.sample_index
            / self.samples_per_beat
        )

        waveform = self._ecg_waveform(phase)

        # Very low-frequency baseline movement
        baseline_wander = (
            20.0
            * math.sin(
                2
                * math.pi
                * 0.20
                * time_seconds
            )
        )

        noise = random.gauss(
            0,
            self.config.noise_level,
        )

        adc_value = (
            self.config.adc_center
            + waveform * self.config.signal_amplitude
            + baseline_wander
            + noise
        )

        self.sample_index += 1

        return int(adc_value)

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

    simulator = AD8232Simulator()

    samples = simulator.generate_samples(
        duration_seconds=5
    )

    print("AD8232 ECG SIMULATION")
    print("=====================")

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

    print("\nFirst 20 RAW ECG samples:")

    for sample in samples[:20]:
        print(sample)