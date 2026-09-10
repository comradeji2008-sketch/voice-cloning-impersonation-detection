pp.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active state tracking per call
engines: Dict[str, RiskEngine] = {}
transcribers: Dict[str, object] = {}


class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, List[WebSocket]] = {}

    async def connect(self, call_id: str, ws: WebSocket):
        await ws.accept()
        self.active.setdefault(call_id, []).append(ws)

    def disconnect(self, call_id: str, ws: WebSocket):
        if call_id in self.active and ws in self.active[call_id]:
            self.active[call_id].remove(ws)

    async def broadcast(self, call_id: str, message: dict):
        for ws in list(self.active.get(call_id, [])):
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(call_id, ws)


manager = ConnectionManager()


def get_engine(call_id: str) -> RiskEngine:
    if call_id not in engines:
        engines[call_id] = RiskEngine()
    return engines[call_id]


def get_transcriber(call_id: str):
    """Real integration point: this is where a live ASR transcript is
    picked up for calls that don't pass one explicitly (see push_signal).
    NOTE: this transcriber's background loop runs independently and keeps
    whatever it last produced. If you end/reset a call, call reset_call()
    below rather than leaving a stale transcriber attached to the call_id --
    otherwise old transcript text can leak into a later, unrelated call."""
    if call_id not in transcribers:
        t = make_transcriber()
        asyncio.create_task(t.start())
        transcribers[call_id] = t
    return transcribers[call_id]


def reset_call(call_id: str):
    """Stops and discards any engine/transcriber state for a call_id, so a
    fresh demo run or a genuinely new call never inherits stale risk score
    or leftover transcript text from a previous session on the same id."""
    if call_id in transcribers:
        try:
            transcribers[call_id].stop()
        except Exception:
            pass
        del transcribers[call_id]
    if call_id in engines:
        del engines[call_id]


class SignalIn(BaseModel):
    acoustic_deepfake_prob: float = 0.0
    vocoder_artifact_score: float = 0.0
    liveness_score: float = 1.0
    watermark_detected: bool = False
    challenge_response_latency_ms: Optional[float] = None
    transcript: Optional[str] = None


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    with open("dashboard.html", "r") as f:
        return f.read()


@app.post("/calls/{call_id}/signal")
async def push_signal(call_id: str, sig_in: SignalIn):
    engine = get_engine(call_id)
    transcriber = get_transcriber(call_id)

    # Explicit transcript wins; otherwise fall back to whatever live ASR
    # (or its scripted-fallback stand-in) has most recently produced.
    transcript = sig_in.transcript or transcriber.get_latest_transcript()

    sig = RiskSignals(
        timestamp=time.time(),
        acoustic_deepfake_prob=sig_in.acoustic_deepfake_prob,
        vocoder_artifact_score=sig_in.vocoder_artifact_score,
        liveness_score=sig_in.liveness_score,
        watermark_detected=sig_in.watermark_detected,
        challenge_response_latency_ms=sig_in.challenge_response_latency_ms,
        transcript=transcript,
    )

    state = engine.update(sig)
    await manager.broadcast(call_id, state)
    return state


@app.get("/calls/{call_id}/history")
async def get_history(call_id: str):
    engine = engines.get(call_id)
    return {"history": engine.history if engine else []}


@app.post("/calls/{call_id}/end")
async def end_call(call_id: str):
    """Call this when a call session is genuinely over, so its risk score
    and any leftover transcript text can't leak into a future call reusing
    the same call_id."""
    reset_call(call_id)
    return {"status": "ended", "call_id": call_id}


@app.post("/calls/{call_id}/demo")
async def run_demo(call_id: str):
    # Always start the scripted demo from a clean slate -- otherwise a
    # second demo run on the same call_id continues from wherever the
    # previous run's score/tier left off, which is confusing to watch.
    reset_call(call_id)
    asyncio.create_task(_demo_scenario(call_id))
    return {"status": "demo started", "call_id": call_id}


async def _demo_scenario(call_id: str):
    engine = get_engine(call_id)

    scenario = [
        (0.05, 0.03, 0.97, False, None),
        (0.10, 0.08, 0.95, False, None),
        (0.35, 0.25, 0.80, False, "hello this is customer support from your bank"),
        (0.55, 0.45, 0.60, False, "please share the OTP sent to your phone right now"),
        (0.60, 0.50, 0.50, False, "it is urgent or your bank account will be blocked"),
        (0.70, 0.65, 0.40, True, "enter the code immediately"),
    ]

    for acoustic, vocoder, liveness, watermark, text in scenario:
        # Feed the scripted text straight into this window's signal instead
        # of routing it through the shared transcriber object. The
        # transcriber's own background loop advances independently on its
        # own timer, so writing into its _latest_transcript from here raced
        # against that loop and could leave stale text behind after the
        # demo finished, leaking into later unrelated calls.
        sig = RiskSignals(
            timestamp=time.time(),
            acoustic_deepfake_prob=acoustic,
            vocoder_artifact_score=vocoder,
            liveness_score=liveness,
            watermark_detected=watermark,
            transcript=text,
        )
        state = engine.update(sig)
        await manager.broadcast(call_id, state)
        await asyncio.sleep(0.5)


@app.get("/codec-profiles")
async def list_codec_profiles():
    """Lists the telephony/cellular/VoIP degradation profiles available
    for the codec-resilience diagnostic below."""
    return {key: profile.name for key, profile in CODEC_PROFILES.items()}


@app.get("/tools/codec-degradation/{profile_key}")
async def test_codec_degradation(profile_key: str):
    """Diagnostic endpoint that actually exercises codec_resilience.py:
    runs a synthetic speech-like tone through the chosen real-world
    transmission profile and reports how much high-frequency detail (where
    vocoder artifacts live) survives. Useful to show judges the codec
    resilience module doing real signal processing, independent of the
    live-call pipeline (which doesn't carry raw audio through this API)."""
    if profile_key not in CODEC_PROFILES:
        return {"error": f"unknown profile '{profile_key}'", "available": list(CODEC_PROFILES.keys())}
    sr = 16000
    clean = synthesize_test_signal(sr=sr)
    degraded = apply_codec_pipeline(clean, sr, CODEC_PROFILES[profile_key], seed=42)
    report = signal_quality_report(clean, degraded, sr)
    return {"profile": CODEC_PROFILES[profile_key].name, **report}


@app.websocket("/ws/{call_id}")
async def ws_endpoint(websocket: WebSocket, call_id: str):
    await manager.connect(call_id, websocket)
    try:
        engine = get_engine(call_id)
        await websocket.send_json({
            "timestamp": time.time(),
            "score": engine.score,
            "tier": engine.tier.value,
            "watermark_detected": False,
            "keyword_escalation": False,
        })
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(call_id, websocket)
