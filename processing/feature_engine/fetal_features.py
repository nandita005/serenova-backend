"""
SERENOVA - Fetal Feature Representation
"""


def build_fetal_features(
    piezo: dict,
    sw420: dict,
) -> dict:

    return {
        "fetal_movement_count": piezo.get(
            "fetal_movement_count",
            0,
        ),

        "fetal_movement_intensity": piezo.get(
            "movement_intensity",
            0.0,
        ),

        "piezo_signal_available": piezo.get(
            "signal_available",
            False,
        ),

        "vibration_event_count": sw420.get(
            "vibration_event_count",
            0,
        ),

        "vibration_event_rate": sw420.get(
            "vibration_event_rate",
            0.0,
        ),

        "sw420_signal_available": sw420.get(
            "signal_available",
            False,
        ),
    }