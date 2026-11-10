"""Local speech-to-text with faster-whisper (no audio leaves the machine)."""

from __future__ import annotations

import logging
import threading
import wave
from pathlib import Path

from config.settings import WhisperSettings
from intelligence.schemas import Transcript

log = logging.getLogger(__name__)

# Segments that whisper produces over music/silence tend to have a high
# no-speech probability and low confidence; dropping them removes the classic
# "Thanks for watching!" hallucinations on music-only clips.
NO_SPEECH_THRESHOLD = 0.6
LOW_LOGPROB_THRESHOLD = -1.0


class Transcriber:
    """Lazily loads one WhisperModel and serializes access to it."""

    def __init__(self, settings: WhisperSettings) -> None:
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return f"faster-whisper/{self.settings.model_size}"

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            from faster_whisper import WhisperModel  # heavy import, deferred

            kwargs: dict = {
                "device": self.settings.device,
                "compute_type": self.settings.compute_type,
            }
            if self.settings.cpu_threads:
                kwargs["cpu_threads"] = self.settings.cpu_threads
            if self.settings.download_root:
                kwargs["download_root"] = str(self.settings.download_root)
            log.info("Loading %s (%s)", self.model_name, kwargs)
            self._model = WhisperModel(self.settings.model_size, **kwargs)
        return self._model

    @staticmethod
    def _load_pcm(audio_path: Path):  # type: ignore[no-untyped-def]
        """Read a 16 kHz mono 16-bit WAV into float32 samples.

        Passing samples directly bypasses faster-whisper's PyAV decoder, which is
        sensitive to PyAV version changes; ffmpeg has already produced the WAV.
        """

        import numpy as np

        with wave.open(str(audio_path), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000:
                return None
            frames = wav.readframes(wav.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    def transcribe(self, audio_path: str | Path) -> Transcript:
        """Transcribe a 16 kHz mono WAV (or any ffmpeg-readable file)."""

        path = Path(audio_path)
        audio = self._load_pcm(path) if path.suffix.lower() == ".wav" else None
        with self._lock:
            model = self._load()
            segments_iter, info = model.transcribe(
                audio if audio is not None else str(path),
                beam_size=self.settings.beam_size,
                vad_filter=self.settings.vad_filter,
                language=self.settings.language,
                condition_on_previous_text=False,
            )
            segments: list[tuple[float, float, str]] = []
            for seg in segments_iter:  # generator: decoding happens while iterating
                text = (seg.text or "").strip()
                if not text:
                    continue
                if seg.no_speech_prob > NO_SPEECH_THRESHOLD and seg.avg_logprob < LOW_LOGPROB_THRESHOLD:
                    continue
                segments.append((round(seg.start, 2), round(seg.end, 2), text))
        return Transcript(
            text=" ".join(s[2] for s in segments).strip(),
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            duration_s=getattr(info, "duration", None),
            segments=segments,
        )
