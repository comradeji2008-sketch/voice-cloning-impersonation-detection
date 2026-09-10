"""
Codec Resilience Preprocessing Layer
====================================

The Gap (from your notes): AI voice detectors are trained on crisp 16kHz/48kHz
studio audio. Real scam calls travel over lossy cellular/VoIP codecs (GSM,
AMR-NB, G.711 PSTN, WhatsApp Opus) that band-limit audio to ~300-3400 Hz and
introduce quantization + packet-loss artifacts. This destroys exactly the
high-frequency phase/vocoder artifacts your acoustic classifier relies on,
so a model trained only on clean audio silently fails on real phone calls.

This module simulates that degradation so you can:
  1. Augment your training set (feed clean + cloned audio through these
     profiles so the classifier learns to detect artifacts that survive
     compression, not ones that get filtered out).
  2. Preprocess live call audio at inference time to match what the model
     was trained on, and/or run a quick "signal quality" check to widen
     your decision thresholds on heavily compressed calls.

No external audio libraries required — numpy + scipy only, so it drops
straight into the existing pipeline (risk_engine.py / server.py) without
new heavy dependencies.

Usage:
    from codec_resilience import CODEC_PROFILES, apply_codec_pipeline

    degraded = apply_codec_pipeline(audio, sr, CODEC_PROFILES["cellular_amr_nb"])
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly


@dataclass
class CodecProfile:
    """Describes one real-world transmission path's degradation characteristics."""
    name: str
    target_sr: int
    band_low: float
    band_high: float
    mu_law: bool
    mu: float
    packet_loss_rate: float
    extra_noise_db: float


CODEC_PROFILES: Dict[str, CodecProfile] = {
    "clean_studio": CodecProfile(
        name="Clean studio (baseline, no degradation)",
        target_sr=48000,
        band_low=20,
        band_high=20000,
        mu_law=False,
        mu=255,
        packet_loss_rate=0.0,
        extra_noise_db=-60,
    ),

    "pstn_g711": CodecProfile(
        name="PSTN G.711 (8kHz mu-law)",
        target_sr=8000,
        band_low=300,
        band_high=3400,
        mu_law=True,
        mu=255,
        packet_loss_rate=0.0,
        extra_noise_db=-35,
    ),

    "cellular_amr_nb": CodecProfile(
        name="Cellular AMR-NB (2G/3G)",
        target_sr=8000,
        band_low=200,
        band_high=3400,
        mu_law=True,
        mu=120,
        packet_loss_rate=0.03,
        extra_noise_db=-25,
    ),

    "voip_opus_whatsapp": CodecProfile(
        name="VoIP Opus (WhatsApp/Meet, wideband)",
        target_sr=16000,
        band_low=50,
        band_high=8000,
        mu_law=False,
        mu=255,
        packet_loss_rate=0.01,
        extra_noise_db=-45,
    ),
}


# --- individual degradation stages ------------------------------------------

def bandpass_filter(
    audio: np.ndarray,
    sr: int,
    low: float,
    high: float,
    order: int = 4
) -> np.ndarray:
    """
    Simulates the fixed telephony passband — everything outside this
    range is physically removed before transmission.
    """
    nyq = sr / 2
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.999)

    b, a = butter(
        order,
        [low_n, high_n],
        btype="band"
    )

    return filtfilt(b, a, audio)


def resample_roundtrip(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> Tuple[np.ndarray, int]:
    """
    Downsamples to the codec's actual transmission rate, then back up to
    orig_sr, so the output is the same length as the input but carries
    the resolution loss of having actually been transmitted at target_sr.
    """
    if target_sr == orig_sr:
        return audio.copy(), orig_sr

    down = resample_poly(
        audio,
        target_sr,
        orig_sr
    )

    back_up = resample_poly(
        down,
        orig_sr,
        target_sr
    )

    # resample_poly can change length by a few samples; trim/pad to match
    if len(back_up) > len(audio):
        back_up = back_up[:len(audio)]

    elif len(back_up) < len(audio):
        back_up = np.pad(
            back_up,
            (0, len(audio) - len(back_up))
        )

    return back_up, orig_sr


def mu_law_roundtrip(
    audio: np.ndarray,
    mu: float = 255.0
) -> np.ndarray:
    """
    G.711-style mu-law companding + 8-bit quantization + expansion.
    Implemented directly in numpy so this stays portable.
    """
    x = np.clip(audio, -1.0, 1.0)

    compressed = (
        np.sign(x)
        * np.log1p(mu * np.abs(x))
        / np.log1p(mu)
    )

    levels = 256

    quantized = np.round(
        (compressed + 1) / 2 * (levels - 1)
    )

    quantized = (
        quantized / (levels - 1) * 2 - 1
    )

    expanded = (
        np.sign(quantized)
        * (1.0 / mu)
        * (
            np.power(
                1 + mu,
                np.abs(quantized)
            ) - 1
        )
    )

    return expanded


def simulate_packet_loss(
    audio: np.ndarray,
    sr: int,
    loss_rate: float,
    packet_ms: float = 20.0,
    seed: int = None
) -> np.ndarray:
    """
    Zeroes out random ~20ms packets to simulate cellular/VoIP packet loss
    and jitter-buffer underruns. No concealment applied (worst case).
    """
    if loss_rate <= 0:
        return audio.copy()

    rng = np.random.default_rng(seed)

    packet_len = max(
        int(sr * packet_ms / 1000),
        1
    )

    out = audio.copy()

    n_packets = len(out) // packet_len

    for i in range(n_packets):
        if rng.random() < loss_rate:
            out[
                i * packet_len:
                (i + 1) * packet_len
            ] = 0.0

    return out


def add_noise_floor(
    audio: np.ndarray,
    noise_db: float,
    seed: int = None
) -> np.ndarray:
    """
    Adds a fixed-level noise floor representing line noise / codec
    quantization hiss relative to the signal's own RMS level.
    """
    rng = np.random.default_rng(seed)

    signal_rms = (
        np.sqrt(np.mean(audio ** 2))
        + 1e-12
    )

    noise_rms = (
        signal_rms
        * (10 ** (noise_db / 20))
    )

    noise = rng.normal(
        0,
        noise_rms,
        size=audio.shape
    )

    return audio + noise


# --- full pipeline -----------------------------------------------------------

def apply_codec_pipeline(
    audio: np.ndarray,
    sr: int,
    profile: CodecProfile,
    seed: int = None
) -> np.ndarray:
    """
    Runs audio through:

    bandpass
    -> resample roundtrip
    -> mu-law
    -> packet loss
    -> noise floor

    Returns audio at the SAME sample rate and length as the input,
    so it can be fed straight into your existing feature-extraction stage.
    """

    out = bandpass_filter(
        audio,
        sr,
        profile.band_low,
        profile.band_high
    )

    out, _ = resample_roundtrip(
        out,
        sr,
        profile.target_sr
    )

    if profile.mu_law:
        out = mu_law_roundtrip(
            out,
            profile.mu
        )

    out = simulate_packet_loss(
        out,
        sr,
        profile.packet_loss_rate,
        seed=seed
    )

    out = add_noise_floor(
        out,
        profile.extra_noise_db,
        seed=seed
    )

    return np.clip(
        out,
        -1.0,
        1.0
    )


# --- metrics: quantify why detection degrades -------------------------------

def high_frequency_energy_ratio(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float = 4000
) -> float:
    """
    Fraction of total spectral energy that lies above cutoff_hz.
    """

    spectrum = np.abs(
        np.fft.rfft(audio)
    ) ** 2

    freqs = np.fft.rfftfreq(
        len(audio),
        d=1 / sr
    )

    total = spectrum.sum() + 1e-12

    high = spectrum[
        freqs >= cutoff_hz
    ].sum()

    return float(
        high / total
    )


def signal_quality_report(
    clean: np.ndarray,
    degraded: np.ndarray,
    sr: int
) -> Dict:
    """
    Quick report you can log per-call:
    how much high-frequency detail survived,
    and how much noise was introduced.
    """

    hf_before = high_frequency_energy_ratio(
        clean,
        sr
    )

    hf_after = high_frequency_energy_ratio(
        degraded,
        sr
    )

    noise_floor = float(
        np.sqrt(
            np.mean(
                (degraded - clean) ** 2
            )
        )
    )

    return {
        "hf_ratio_before": round(
            hf_before,
            5
        ),

        "hf_ratio_after": round(
            hf_after,
            5
        ),

        "hf_energy_retained_pct": round(
            100 * hf_after / (hf_before + 1e-12),
            1
        ),

        "added_noise_rms": round(
            noise_floor,
            5
        ),
    }


# ---------------------------------------------------------------------------
# Demo: synthesize a speech-like test signal
# ---------------------------------------------------------------------------

def _synthesize_test_signal(
    sr: int = 16000,
    duration: float = 2.0
) -> np.ndarray:

    t = np.linspace(
        0,
        duration,
        int(sr * duration),
        endpoint=False
    )

    signal = np.zeros_like(t)

    f0 = 150.0

    for k in range(1, 45):

        freq = f0 * k

        if freq > sr / 2:
            break

        amp = 1.0 / k

        signal += amp * np.sin(
            2 * np.pi * freq * t
        )

    signal += (
        0.02
        * np.random.default_rng(0).normal(
            size=t.shape
        )
    )

    return signal / np.max(
        np.abs(signal)
    )


# Public alias — used by server.py's diagnostic endpoint.

def synthesize_test_signal(
    sr: int = 16000,
    duration: float = 2.0
):
    return _synthesize_test_signal(
        sr=sr,
        duration=duration
    )


def run_demo():

    sr = 16000

    clean = _synthesize_test_signal(
        sr=sr
    )

    print(
        f"{'Profile':<35}"
        f"{'HF energy retained':<22}"
        f"{'Added noise RMS':<18}"
    )

    print("-" * 75)

    for key, profile in CODEC_PROFILES.items():

        degraded = apply_codec_pipeline(
            clean,
            sr,
            profile,
            seed=42
        )

        report = signal_quality_report(
            clean,
            degraded,
            sr
        )

        print(
            f"{profile.name:<35}"
            f"{report['hf_energy_retained_pct']:>6}%"
            f"{'':<15}"
            f"{report['added_noise_rms']:<18}"
        )

    try:

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            len(CODEC_PROFILES),
            1,
            figsize=(9, 2.2 * len(CODEC_PROFILES))
        )

        for ax, (key, profile) in zip(
            axes,
            CODEC_PROFILES.items()
        ):

            degraded = apply_codec_pipeline(
                clean,
                sr,
                profile,
                seed=42
            )

            ax.specgram(
                degraded,
                Fs=sr,
                NFFT=512,
                noverlap=256,
                cmap="magma"
            )

            ax.set_title(
                profile.name,
                fontsize=10
            )

            ax.set_ylabel("Hz")

        axes[-1].set_xlabel(
            "time (s)"
        )

        plt.tight_layout()

        plt.savefig(
            "codec_degradation_comparison.png",
            dpi=130
        )

        print(
            "\nSaved spectrogram comparison to "
            "codec_degradation_comparison.png"
        )

    except ImportError:
        pass


if __name__ == "__main__":
    run_demo()
