from __future__ import annotations

import math
import struct
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MEDIA_DIR = ROOT / "data" / "synthetic" / "media"


def main() -> None:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    write_audio_demo(MEDIA_DIR / "consulta_demo.wav")
    write_video_demo(MEDIA_DIR / "fisioterapia_demo.mp4")
    print("demo-media-ok")
    print(str(MEDIA_DIR / "consulta_demo.wav"))
    print(str(MEDIA_DIR / "fisioterapia_demo.mp4"))


def write_audio_demo(path: Path) -> None:
    sample_rate = 16000
    duration_seconds = 4
    amplitude = 16000
    frequency = 220.0

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames: list[bytes] = []
        total_samples = sample_rate * duration_seconds
        for index in range(total_samples):
            t = index / sample_rate
            envelope = 0.7 if (index // (sample_rate // 2)) % 2 == 0 else 0.25
            value = int(amplitude * envelope * math.sin(2.0 * math.pi * frequency * t))
            frames.append(struct.pack("<h", value))
        wav_file.writeframes(b"".join(frames))


def write_video_demo(path: Path) -> None:
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        raise RuntimeError("OpenCV nao disponivel para gerar o video demo.") from exc

    width = 640
    height = 360
    fps = 12
    total_frames = 72
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Falha ao criar o arquivo de video demo.")

    for frame_idx in range(total_frames):
        frame = np.full((height, width, 3), 245, dtype=np.uint8)

        center_x = 130 + frame_idx * 3
        head_y = 90
        body_y = 200
        lean = 0
        if 28 <= frame_idx <= 52:
            lean = 28
        if frame_idx >= 56:
            cv2.rectangle(frame, (470, 110), (560, 220), (70, 70, 220), -1)

        cv2.circle(frame, (center_x, head_y), 22, (40, 40, 40), 3)
        cv2.line(frame, (center_x, head_y + 22), (center_x + lean, body_y), (40, 40, 40), 4)
        cv2.line(frame, (center_x + lean, 130), (center_x - 35, 165), (40, 40, 40), 4)
        cv2.line(frame, (center_x + lean, 130), (center_x + 40 + lean, 160), (40, 40, 40), 4)
        cv2.line(frame, (center_x + lean, body_y), (center_x - 25, 280), (40, 40, 40), 4)
        cv2.line(frame, (center_x + lean, body_y), (center_x + 35 + lean, 285), (40, 40, 40), 4)

        cv2.putText(
            frame,
            "Fisioterapia Demo",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
        if 28 <= frame_idx <= 52:
            cv2.putText(
                frame,
                "Desvio postural simulado",
                (20, 330),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 80, 180),
                2,
                cv2.LINE_AA,
            )
        if frame_idx >= 56:
            cv2.putText(
                frame,
                "Objeto inesperado na area",
                (310, 320),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 180),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)

    writer.release()


if __name__ == "__main__":
    main()
