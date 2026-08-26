"""
Serenova Sensor Payload Contract

This module defines the canonical payload produced by the
sensor simulator and expected from the future ESP32 hardware.

Important:
    Raw sensor values are kept separate from processed features.
    Clinical risk, health scores, alerts, and recommendations
    are NOT part of this hardware payload.
"""

from datetime import datetime, timezone
from typing import Optional


def create_sensor_payload(
    device_id: str,
    patient_id: str,

    # MAX30102
    ir: int,
    red: int,
    heart_rate: Optional[float],
    spo2: Optional[float],

    # AD8232
    ecg: int,

    # MPU6050
    accel_x: float,
    accel_y: float,
    accel_z: float,
    gyro_x: float,
    gyro_y: float,
    gyro_z: float,
    activity_level: Optional[str] = None,
    movement_score: Optional[float] = None,

    # DS18B20
    temperature_c: float = 36.7,

    # Piezo
    piezo_value: int = 0,
    fetal_movement_count: Optional[int] = None,
    movement_intensity: Optional[float] = None,

    # SW-420
    vibration: int = 0,
) -> dict:

    if vibration not in (0, 1):
        raise ValueError("SW-420 vibration must be 0 or 1")

    return {
        "device_id": device_id,
        "patient_id": patient_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),

        "max30102": {
            "raw": {
                "ir": ir,
                "red": red,
            },
            "processed": {
                "heart_rate": heart_rate,
                "spo2": spo2,
            },
        },

        "ad8232": {
            "ecg": ecg,
        },

        "mpu6050": {
            "accel": {
                "x": accel_x,
                "y": accel_y,
                "z": accel_z,
            },
            "gyro": {
                "x": gyro_x,
                "y": gyro_y,
                "z": gyro_z,
            },
            "activity_level": activity_level,
            "movement_score": movement_score,
        },

        "ds18b20": {
            "temperature_c": temperature_c,
        },

        "piezo": {
            "value": piezo_value,
            "fetal_movement_count": fetal_movement_count,
            "movement_intensity": movement_intensity,
        },

        "sw420": {
            "vibration": vibration,
        },
    }


if __name__ == "__main__":

    payload = create_sensor_payload(
        device_id="SERENOVA_WB_001",
        patient_id="PAT001",

        # MAX30102
        ir=12345,
        red=11800,
        heart_rate=82,
        spo2=98,

        # AD8232
        ecg=1876,

        # MPU6050
        accel_x=0.12,
        accel_y=-0.04,
        accel_z=0.98,
        gyro_x=1.24,
        gyro_y=-0.52,
        gyro_z=0.31,
        activity_level="Moderate",
        movement_score=72,

        # DS18B20
        temperature_c=36.7,

        # Piezo
        piezo_value=1842,
        fetal_movement_count=18,
        movement_intensity=0.81,

        # SW-420
        vibration=0,
    )

    import json

    print(json.dumps(payload, indent=2))