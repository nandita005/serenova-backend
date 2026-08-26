"""
SERENOVA - SW-420 Processor

Digital vibration events
        ↓
debouncing
        ↓
event counting
        ↓
vibration activity features
"""

from __future__ import annotations


class SW420Processor:

    def __init__(
        self,
        sampling_rate: int = 20,
        debounce_seconds: float = 0.25,
    ):

        self.sampling_rate = sampling_rate

        self.debounce_samples = max(
            1,
            int(
                sampling_rate
                * debounce_seconds
            ),
        )

    def detect_events(
        self,
        samples: list[int],
    ) -> list[int]:

        events = []

        last_event = (
            -self.debounce_samples
        )

        for index, value in enumerate(samples):

            if value != 1:
                continue

            if (
                index - last_event
                < self.debounce_samples
            ):
                continue

            events.append(index)

            last_event = index

        return events

    def process(
        self,
        samples: list[int],
    ) -> dict:

        if not samples:

            return {
                "signal_available": False,
                "vibration_event_count": 0,
                "vibration_event_rate": 0.0,
            }

        events = self.detect_events(
            samples
        )

        duration_seconds = (
            len(samples)
            / self.sampling_rate
        )

        event_rate = (
            len(events)
            / duration_seconds
            if duration_seconds > 0
            else 0.0
        )

        return {
            "signal_available": True,

            "vibration_event_count": len(
                events
            ),

            "vibration_event_rate": round(
                event_rate,
                3,
            ),

            "raw_vibration_count": sum(
                samples
            ),

            "detected_event_indices": events,
        }


if __name__ == "__main__":

    from simulator.sensors.sw420 import (
        SW420Simulator,
    )

    simulator = SW420Simulator()

    samples = simulator.generate_samples(
        duration_seconds=30
    )

    processor = SW420Processor(
        sampling_rate=20
    )

    result = processor.process(
        samples
    )

    print("\nSW-420 PROCESSING RESULT")
    print("========================")

    for key, value in result.items():
        print(f"{key}: {value}")