try:
    from faster_whisper import WhisperModel
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False

ROLLING_WINDOW_SEC = 4.0     # how much recent audio Whisper sees each pass
ASR_INTERVAL_SEC = 1.0       # how often we re-run transcription
SAMPLE_RATE = 16000          # Whisper expects 16kHz mono float32 PCM


class LiveTranscriber:
    """
    Feed raw float32 mono PCM chunks in with push_audio(), and call
    get_latest_transcript() from your 300ms risk-engine loop to attach
    the freshest available transcript to each RiskSignals window.

    A background asyncio task re-runs Whisper on the rolling buffer every
    ASR_INTERVAL_SEC, so push_audio() itself never blocks the hot path.
    """

    def __init__(self, model_size: str = "tiny.en", device: str = "cpu",
                 compute_type: str = "int8"):
        if not _WHISPER_AVAILABLE:
            raise RuntimeError(
                "faster-whisper is not installed. Run `pip install faster-whisper` "
                "in your project environment. (This check runs when you construct "
                "LiveTranscriber, not at import time, so the rest of your app can "
                "still import this module even before the dependency is installed.)"
            )
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._buffer: deque = deque(maxlen=int(ROLLING_WINDOW_SEC * SAMPLE_RATE))
        self._latest_transcript: str = ""
        self._last_run: float = 0.0
        self._running = False

    def push_audio(self, chunk: np.ndarray):
        """Call every ~300ms with a 1-D float32 array of mono samples at
        SAMPLE_RATE (resample upstream if your capture rate differs)."""
        self._buffer.extend(chunk.tolist())

    def get_latest_transcript(self) -> str:
        """Non-blocking: returns whatever the background task last produced."""
        return self._latest_transcript

    async def start(self):
        """Launch the background transcription loop. Call once per call session."""
        self._running = True
        asyncio.create_task(self._loop())

    def stop(self):
        self._running = False

    async def _loop(self):
        while self._running:
            now = time.time()
            if now - self._last_run >= ASR_INTERVAL_SEC and len(self._buffer) > SAMPLE_RATE * 0.5:
                audio = np.array(self._buffer, dtype=np.float32)
                # Run the blocking whisper call in a thread so it never
                # stalls the 300ms risk-engine cadence in the same event loop.
                text = await asyncio.to_thread(self._transcribe, audio)
                if text:
                    self._latest_transcript = text
                self._last_run = now
            await asyncio.sleep(0.1)

    def _transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=1,
            vad_filter=True, condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


# ---------------------------------------------------------------------------
# Fallback: replays a scripted transcript on a timer instead of running real
# ASR. Lets you build/demo the rest of the pipeline on a machine without
# faster-whisper installed, or without a CPU fast enough for real-time —
# swap in LiveTranscriber later with zero changes to the calling code.
# ---------------------------------------------------------------------------

class ScriptedTranscriber:
    """Drop-in stand-in for LiveTranscriber. Same interface, no real audio
    processing — just replays `script` lines one per ASR_INTERVAL_SEC."""

    def __init__(self, script: List[str]):
        self._script = script
        self._latest_transcript = ""
        self._running = False

    async def start(self):
        self._running = True
        asyncio.create_task(self._loop())

    def stop(self):
        self._running = False

    def push_audio(self, chunk):
        pass  # no-op; scripted transcriber ignores real audio input

    def get_latest_transcript(self) -> str:
        return self._latest_transcript

    async def _loop(self):
        for line in self._script:
            if not self._running:
                return
            await asyncio.sleep(ASR_INTERVAL_SEC)
            self._latest_transcript = line


def make_transcriber(script: Optional[List[str]] = None):
    """Convenience factory: returns a real LiveTranscriber if faster-whisper
    is installed, otherwise falls back to ScriptedTranscriber so the rest of
    your pipeline keeps working during development/demos."""
    if _WHISPER_AVAILABLE:
        return LiveTranscriber()
    fallback_script = script or [
        "hello, this is your bank calling",
        "please share the OTP sent to your phone",
        "it's urgent, your bank account will be blocked",
    ]
    print("[whisper_asr] faster-whisper not installed — using ScriptedTranscriber fallback.")
    return ScriptedTranscriber(fallback_script)


# ---------------------------------------------------------------------------
# Integration sketch for server.py:
#
#   from whisper_asr import make_transcriber
#   transcribers: Dict[str, object] = {}
#
#   @app.post("/calls/{call_id}/start")
#   async def start_call(call_id: str):
#       t = make_transcriber()
#       await t.start()
#       transcribers[call_id] = t
#       return {"status": "started"}
#
#   @app.post("/calls/{call_id}/audio_chunk")
#   async def push_audio_chunk(call_id: str, chunk: AudioChunkIn):
#       transcribers[call_id].push_audio(np.array(chunk.samples, dtype=np.float32))
#       return {"status": "buffered"}
#
#   # inside push_signal / your 300ms loop, before engine.update(sig):
#   sig.transcript = transcribers.get(call_id).get_latest_transcript() if call_id in transcribers else None
# ---------------------------------------------------------------------------


# --- self-contained demo (works even without faster-whisper installed) -----

async def _demo():
    from risk_engine import RiskEngine, RiskSignals

    transcriber = make_transcriber()
    await transcriber.start()
    engine = RiskEngine()
    engine.on_tier_change(lambda o, n, s: print(f"  tier {o.value} -> {n.value} (score={s:.2f})"))

    print("Simulating a call with acoustic score fixed borderline (0.55) "
          "while ASR transcript arrives over time:\n")
    for i in range(8):
        transcript = transcriber.get_latest_transcript()
        sig = RiskSignals(acoustic_deepfake_prob=0.55, vocoder_artifact_score=0.4,
                           liveness_score=0.6, transcript=transcript)
        state = engine.update(sig)
        print(f"t={i*0.3:>4.1f}s | transcript={transcript!r:<55} | "
              f"score={state['score']:.3f} | tier={state['tier']}")
        await asyncio.sleep(0.3)
    transcriber.stop()


if __name__ == "__main__":
    asyncio.run(_demo())
