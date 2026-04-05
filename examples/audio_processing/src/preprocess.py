"""
ActiveReactivePreprocessor
==========================
Feature engineering for 4-channel first-order Ambisonics (FOA) waveforms.

Computes **active** and **reactive acoustic intensity** components in the
STFT domain following Jacobsen & Rydén (1990), then slices the waveform
into fixed-length CPU chunks before they are batched onto the GPU.

Typical usage
-------------
>>> from audio_processing.src.preprocess import ActiveReactivePreprocessor
>>> pre = ActiveReactivePreprocessor(sample_rate=16_000, chunk_duration=0.5)
>>> features = pre(waveform)   # waveform: (4, T) float32 CPU tensor
>>> # features: (N_chunks, 6, F, T_frames) float32 CPU tensor

Output feature channels (axis 1)
----------------------------------
Index  Name    Formula                   Physical meaning
-----  ------  ------------------------  --------------------------------
0      Ia_x    Re( W · X* )              Active intensity, front–back
1      Ia_y    Re( W · Y* )              Active intensity, left–right
2      Ia_z    Re( W · Z* )              Active intensity, up–down
3      Ir_x    Im( W · X* )             Reactive intensity, front–back
4      Ir_y    Im( W · Y* )             Reactive intensity, left–right
5      Ir_z    Im( W · Z* )             Reactive intensity, up–down

where W, X, Y, Z are the complex STFT coefficients of the four B-format
channels and * denotes complex conjugation.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# B-format channel indices (ACN ordering as used in Spatial LibriSpeech)
# ---------------------------------------------------------------------------
_CH_W, _CH_X, _CH_Y, _CH_Z = 0, 1, 2, 3


class ActiveReactivePreprocessor:
    """Convert 4-channel FOA waveforms into active/reactive intensity features.

    The pipeline consists of three stages, all executed on the **CPU**:

    1. **Chunking** — the raw waveform ``(4, T)`` is sliced into
       ``N_chunks`` non-overlapping segments of ``chunk_duration`` seconds
       each, producing a ``(N_chunks, 4, chunk_samples)`` tensor.  Padding
       with zeros is applied to the final chunk if the recording length is
       not an exact multiple of ``chunk_duration``.

    2. **STFT** — a Short-Time Fourier Transform is computed for every
       channel of every chunk simultaneously via a batched call to
       ``torch.stft``.  The result is a complex tensor of shape
       ``(N_chunks, 4, F, T_frames)`` where ``F = n_fft // 2 + 1``.

    3. **Active / Reactive extraction** — the cross-spectrum between the
       omnidirectional W channel and each of the three velocity channels
       (X, Y, Z) is computed.  The real part gives the active intensity
       (time-averaged power flow) and the imaginary part gives the reactive
       intensity (stored / circulating energy flow).  Optional W-power
       normalisation yields a unit-magnitude DOA estimate that is robust to
       level variations.

    Parameters
    ----------
    sample_rate:
        Sample rate of the input waveform in Hz.  Defaults to ``16_000``.
    chunk_duration:
        Duration of each chunk in seconds.  Defaults to ``0.5``.
    n_fft:
        FFT size.  Determines frequency resolution (``F = n_fft // 2 + 1``
        bins).  Defaults to ``512`` (32 ms at 16 kHz).
    hop_length:
        Number of samples between successive STFT frames.  Defaults to
        ``128`` (8 ms at 16 kHz, 75 % overlap with a 512-point window).
    win_length:
        Length of the analysis window in samples.  Defaults to ``n_fft``.
    window_fn:
        Name of the window function passed to ``torch.hann_window`` or
        ``torch.hamming_window``.  Accepted values: ``"hann"``,
        ``"hamming"``.  Defaults to ``"hann"``.
    center:
        If ``True`` (default), the signal is padded by ``n_fft // 2`` on
        both sides before the STFT so that frame 0 is centred at sample 0.
    pad_mode:
        Padding mode used when ``center=True``.  Passed directly to
        ``torch.nn.functional.pad``.  Defaults to ``"reflect"``.
    normalize_intensity:
        When ``True`` (default), each intensity component is divided by
        ``|W(f,t)|² + eps``, yielding a unit DOA direction estimate per
        time-frequency bin.
    eps:
        Small constant added to the W-power denominator to avoid division
        by zero.  Defaults to ``1e-8``.
    drop_last_chunk:
        When ``True``, the final (possibly zero-padded) chunk is discarded
        if the recording is not an exact multiple of ``chunk_duration``.
        Defaults to ``False``.

    Raises
    ------
    ValueError
        If ``window_fn`` is not one of ``"hann"`` or ``"hamming"``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        sample_rate: int = 16_000,
        chunk_duration: float = 0.5,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: Optional[int] = None,
        window_fn: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        normalize_intensity: bool = True,
        eps: float = 1e-8,
        drop_last_chunk: bool = False,
    ) -> None:
        _valid_windows = {"hann", "hamming"}
        if window_fn not in _valid_windows:
            raise ValueError(
                f"window_fn must be one of {_valid_windows!r}, got {window_fn!r}"
            )

        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples: int = int(round(sample_rate * chunk_duration))
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length: int = win_length if win_length is not None else n_fft
        self.window_fn = window_fn
        self.center = center
        self.pad_mode = pad_mode
        self.normalize_intensity = normalize_intensity
        self.eps = eps
        self.drop_last_chunk = drop_last_chunk

        # Number of frequency bins
        self.n_freq: int = n_fft // 2 + 1

        # Pre-compute the analysis window once; kept on CPU.
        if window_fn == "hann":
            self._window: Tensor = torch.hann_window(self.win_length)
        else:
            self._window = torch.hamming_window(self.win_length)

        logger.debug(
            "ActiveReactivePreprocessor | sr=%d | chunk=%.2fs (%d samples) | "
            "n_fft=%d | hop=%d | win=%d | normalize=%s",
            sample_rate,
            chunk_duration,
            self.chunk_samples,
            n_fft,
            hop_length,
            self.win_length,
            normalize_intensity,
        )

    # ------------------------------------------------------------------
    # Stage 1 — Chunking (CPU)
    # ------------------------------------------------------------------

    def chunk_waveform(self, waveform: Tensor) -> Tensor:
        """Slice a variable-length FOA waveform into fixed-size chunks.

        Operates **entirely on the CPU**.  The last chunk is zero-padded
        to ``chunk_samples`` unless ``drop_last_chunk=True``.

        Parameters
        ----------
        waveform:
            Raw FOA waveform, shape ``(4, T)``, dtype ``float32``.
            Must reside on CPU.

        Returns
        -------
        torch.Tensor
            Shape ``(N_chunks, 4, chunk_samples)``, dtype ``float32``,
            on CPU.

        Raises
        ------
        ValueError
            If ``waveform`` does not have exactly 4 channels.
        RuntimeError
            If ``waveform`` is not on the CPU.
        """
        if waveform.device.type != "cpu":
            raise RuntimeError(
                "chunk_waveform requires a CPU tensor.  "
                f"Got device={waveform.device}."
            )
        if waveform.shape[0] != 4:
            raise ValueError(
                f"Expected 4-channel FOA waveform (shape[0]==4), "
                f"got shape {tuple(waveform.shape)}."
            )

        n_channels, total_samples = waveform.shape
        chunk_size = self.chunk_samples

        # Pad to a multiple of chunk_size
        remainder = total_samples % chunk_size
        if remainder != 0:
            pad_len = chunk_size - remainder
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        n_chunks = waveform.shape[-1] // chunk_size

        # Reshape to (N_chunks, 4, chunk_samples)
        chunks = waveform.reshape(n_channels, n_chunks, chunk_size)
        chunks = chunks.permute(1, 0, 2).contiguous()  # (N_chunks, 4, chunk_samples)

        if self.drop_last_chunk and remainder != 0:
            # The last chunk was zero-padded; discard it
            chunks = chunks[:-1]

        return chunks  # (N_chunks, 4, chunk_samples)

    # ------------------------------------------------------------------
    # Stage 2 — STFT
    # ------------------------------------------------------------------

    def compute_stft(self, chunks: Tensor) -> Tensor:
        """Compute a batched STFT for all chunks and all FOA channels.

        Parameters
        ----------
        chunks:
            Shape ``(N_chunks, 4, chunk_samples)``, CPU float32.

        Returns
        -------
        torch.Tensor
            Complex tensor, shape ``(N_chunks, 4, F, T_frames)``,
            dtype ``torch.complex64``, on CPU.
            ``F = n_fft // 2 + 1``.
        """
        n_chunks, n_channels, chunk_samples = chunks.shape
        window = self._window  # CPU tensor, reused across calls

        # Flatten batch and channel dimensions for a single batched STFT call:
        # (N_chunks * 4, chunk_samples)
        flat = chunks.reshape(n_chunks * n_channels, chunk_samples)

        # torch.stft: (batch, T) → (batch, F, T_frames) complex
        stft_flat = torch.stft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=False,
            onesided=True,
            return_complex=True,
        )  # (N_chunks*4, F, T_frames)

        _, n_freq, n_frames = stft_flat.shape
        # Unflatten → (N_chunks, 4, F, T_frames)
        stft = stft_flat.reshape(n_chunks, n_channels, n_freq, n_frames)
        return stft

    # ------------------------------------------------------------------
    # Stage 3 — Active / Reactive Intensity Extraction
    # ------------------------------------------------------------------

    def extract_active_reactive(self, stft: Tensor) -> Tensor:
        """Compute active and reactive acoustic intensity from FOA STFT.

        Uses the Jacobsen & Rydén (1990) formulation:

            Active intensity   Ia_d = Re( W(f,t) · Xd(f,t)* )
            Reactive intensity Ir_d = Im( W(f,t) · Xd(f,t)* )

        where ``d ∈ {x, y, z}`` and ``*`` is complex conjugation.

        Optional W-power normalisation divides each component by
        ``|W(f,t)|² + eps``, producing a unit-normalised DOA estimate
        independent of source level.

        Parameters
        ----------
        stft:
            Complex STFT tensor, shape ``(N_chunks, 4, F, T_frames)``.

        Returns
        -------
        torch.Tensor
            Real float32 tensor, shape ``(N_chunks, 6, F, T_frames)``.

            Channel layout:
            ┌───────┬───────────────────────────────────────────────────┐
            │ Index │ Description                                        │
            ├───────┼───────────────────────────────────────────────────┤
            │   0   │ Ia_x  Re(W · X*)  active intensity, front–back    │
            │   1   │ Ia_y  Re(W · Y*)  active intensity, left–right    │
            │   2   │ Ia_z  Re(W · Z*)  active intensity, up–down       │
            │   3   │ Ir_x  Im(W · X*)  reactive intensity, front–back  │
            │   4   │ Ir_y  Im(W · Y*)  reactive intensity, left–right  │
            │   5   │ Ir_z  Im(W · Z*)  reactive intensity, up–down     │
            └───────┴───────────────────────────────────────────────────┘
        """
        W = stft[:, _CH_W, :, :]  # (N_chunks, F, T_frames) complex
        X = stft[:, _CH_X, :, :]
        Y = stft[:, _CH_Y, :, :]
        Z = stft[:, _CH_Z, :, :]

        # Cross-spectrum: W · conj(d) for d in {X, Y, Z}
        cs_x = W * X.conj()  # (N_chunks, F, T_frames) complex
        cs_y = W * Y.conj()
        cs_z = W * Z.conj()

        # Active (real part) and reactive (imaginary part)
        Ia_x = cs_x.real
        Ia_y = cs_y.real
        Ia_z = cs_z.real
        Ir_x = cs_x.imag
        Ir_y = cs_y.imag
        Ir_z = cs_z.imag

        if self.normalize_intensity:
            # Divide by W-channel power to obtain a level-independent
            # DOA estimate (unit-normalised intensity vector per TF bin).
            W_power = (W.real ** 2 + W.imag ** 2) + self.eps  # |W|² + ε
            Ia_x = Ia_x / W_power
            Ia_y = Ia_y / W_power
            Ia_z = Ia_z / W_power
            Ir_x = Ir_x / W_power
            Ir_y = Ir_y / W_power
            Ir_z = Ir_z / W_power

        # Stack along channel dimension → (N_chunks, 6, F, T_frames)
        features = torch.stack([Ia_x, Ia_y, Ia_z, Ir_x, Ir_y, Ir_z], dim=1)
        return features.float()

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def __call__(self, waveform: Tensor) -> Tensor:
        """Run the full preprocessing pipeline on a single FOA recording.

        Executes all three stages (chunk → STFT → active/reactive) on
        the **CPU**.  The result is a ready-to-collate tensor that the
        ``DataLoader`` can move to the GPU one chunk-batch at a time,
        keeping peak VRAM usage proportional to ``batch_size`` rather than
        to the full recording length.

        Parameters
        ----------
        waveform:
            4-channel FOA waveform, shape ``(4, T)``, dtype ``float32``,
            on CPU.

        Returns
        -------
        torch.Tensor
            Shape ``(N_chunks, 6, F, T_frames)``, dtype ``float32``,
            on CPU.

            ``N_chunks = ceil(T / chunk_samples)`` (or one fewer if
            ``drop_last_chunk=True`` and ``T % chunk_samples != 0``).
            ``F = n_fft // 2 + 1``.
            ``T_frames = ceil(chunk_samples / hop_length)`` (approximately,
            exact value depends on ``center`` and padding).
        """
        # Stage 1 — CPU chunking
        chunks = self.chunk_waveform(waveform)           # (N_chunks, 4, S)

        # Stage 2 — batched STFT (CPU)
        stft = self.compute_stft(chunks)                 # (N_chunks, 4, F, T_f) complex

        # Stage 3 — active / reactive extraction
        features = self.extract_active_reactive(stft)    # (N_chunks, 6, F, T_f)

        return features

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def output_channels(self) -> int:
        """Number of feature channels in the output (always 6)."""
        return 6

    @property
    def feature_names(self) -> list[str]:
        """Human-readable names for the 6 output feature channels."""
        return ["Ia_x", "Ia_y", "Ia_z", "Ir_x", "Ir_y", "Ir_z"]

    def output_shape(self, waveform_length: int) -> tuple[int, int, int, int]:
        """Return the expected output shape for a given waveform length.

        Parameters
        ----------
        waveform_length:
            Number of time samples ``T`` in the input waveform.

        Returns
        -------
        tuple
            ``(N_chunks, 6, F, T_frames)``
        """
        import math

        if self.drop_last_chunk:
            n_chunks = waveform_length // self.chunk_samples
        else:
            n_chunks = math.ceil(waveform_length / self.chunk_samples)

        if self.center:
            # torch.stft pads n_fft//2 on each side
            padded = self.chunk_samples + self.n_fft
            n_frames = (padded - self.win_length) // self.hop_length + 1
        else:
            n_frames = (self.chunk_samples - self.win_length) // self.hop_length + 1

        return (n_chunks, self.output_channels, self.n_freq, n_frames)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sr={self.sample_rate}, "
            f"chunk={self.chunk_duration}s ({self.chunk_samples} samples), "
            f"n_fft={self.n_fft}, "
            f"hop={self.hop_length}, "
            f"win={self.win_length}, "
            f"window={self.window_fn!r}, "
            f"normalize={self.normalize_intensity}, "
            f"drop_last={self.drop_last_chunk}"
            f")"
        )
