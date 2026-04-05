"""
SpatialLibriSpeechDataset
=========================
A robust OOP PyTorch Dataset for the Spatial LibriSpeech dataset targeted at
the 3D Source Localization task.

The dataset provides 4-channel first-order Ambisonics (FOA) speech recordings
paired with azimuth / elevation targets derived from the parquet metadata file.

Typical usage
-------------
>>> from audio_processing.src.dataset import SpatialLibriSpeechDataset
>>> train_ds = SpatialLibriSpeechDataset(
...     audio_root="/data/spatial_librispeech/ambisonics",
...     metadata_path="/data/spatial_librispeech/metadata.parquet",
...     split="train",
... )
>>> waveform, azimuth, elevation = train_ds[0]
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Split = Literal["train", "test"]

# Metadata URL for convenience (mirrors Apple ML Research hosting)
METADATA_URL = (
    "https://docs-assets.developer.apple.com/ml-research/datasets/"
    "spatial-librispeech/v1/metadata.parquet"
)


class SpatialLibriSpeechDataset(Dataset):
    """PyTorch Dataset for the Spatial LibriSpeech 3D Source Localization task.

    Each sample consists of:
    * a 4-channel first-order Ambisonics (FOA) waveform tensor  ``(4, T)``
    * a scalar azimuth target  (radians)
    * a scalar elevation target (radians)

    Parameters
    ----------
    audio_root:
        Root directory that contains the ``.flac`` audio files.
        Each file is named ``{sample_id:06d}.flac`` (zero-padded to 6 digits),
        e.g. ``000042.flac``.
    metadata_path:
        Path to the ``metadata.parquet`` file **or** a URL from which it will
        be downloaded on first use.  Defaults to the Apple ML Research URL.
    split:
        One of ``"train"`` or ``"test"``.  Only samples whose ``split``
        column matches this value are included.
    lite_version_only:
        When ``True``, restrict to samples that belong to the *lite* version
        of the dataset (``lite_version == True``).  Useful for quick
        experiments.  Defaults to ``False``.
    transform:
        Optional callable applied to the raw waveform tensor ``(4, T)``
        **after** loading.  Use this for feature extraction (e.g. STFT,
        mel-spectrogram).
    target_transform:
        Optional callable applied to the ``(azimuth, elevation)`` tuple.
    verify_checksums:
        When ``True``, verify the SHA-256 checksum of each audio file against
        the value stored in the metadata.  Adds I/O overhead; recommended only
        during data validation.  Defaults to ``False``.
    sample_rate:
        Expected sample rate of the audio files (Hz).  ``torchaudio`` will
        raise if the actual rate differs.  The Spatial LibriSpeech dataset is
        recorded at 16 kHz.  Defaults to ``16000``.

    Raises
    ------
    ValueError
        If ``split`` is not ``"train"`` or ``"test"``.
    FileNotFoundError
        If ``audio_root`` does not exist on the file system.
    RuntimeError
        If the parquet file cannot be loaded from the given path/URL.
    """

    _VALID_SPLITS: Tuple[str, ...] = ("train", "test")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        audio_root: str | os.PathLike,
        metadata_path: str | os.PathLike = METADATA_URL,
        split: Split = "train",
        lite_version_only: bool = False,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transform: Optional[
            Callable[[Tuple[float, float]], Tuple[float, float]]
        ] = None,
        verify_checksums: bool = False,
        sample_rate: int = 16_000,
    ) -> None:
        super().__init__()

        # ---- validate arguments ----------------------------------------
        if split not in self._VALID_SPLITS:
            raise ValueError(
                f"split must be one of {self._VALID_SPLITS!r}, got {split!r}"
            )

        self.audio_root = Path(audio_root)
        if not self.audio_root.exists():
            raise FileNotFoundError(
                f"audio_root does not exist: {self.audio_root}"
            )

        self.split = split
        self.lite_version_only = lite_version_only
        self.transform = transform
        self.target_transform = target_transform
        self.verify_checksums = verify_checksums
        self.sample_rate = sample_rate

        # ---- load & filter metadata ------------------------------------
        self.metadata: pd.DataFrame = self._load_metadata(str(metadata_path))
        self.metadata = self._filter_metadata(self.metadata)

        # Reset integer index so __getitem__ can use positional lookup
        self.metadata = self.metadata.reset_index(drop=True)

        logger.info(
            "SpatialLibriSpeechDataset | split=%s | samples=%d | "
            "audio_root=%s",
            self.split,
            len(self.metadata),
            self.audio_root,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_metadata(path_or_url: str) -> pd.DataFrame:
        """Load the parquet metadata from a local path or a remote URL."""
        try:
            df = pd.read_parquet(path_or_url)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load metadata from {path_or_url!r}. "
                "Ensure the path is correct or the URL is reachable."
            ) from exc
        return df

    def _filter_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply split (and optional lite-version) filters."""
        mask = df["split"] == self.split
        if self.lite_version_only:
            mask &= df["lite_version"] == True  # noqa: E712
        filtered = df[mask].copy()
        if filtered.empty:
            logger.warning(
                "No samples found for split=%r lite_version_only=%s",
                self.split,
                self.lite_version_only,
            )
        return filtered

    def _build_audio_path(self, sample_id: int) -> Path:
        """Return the .flac file path for a given sample_id.

        The filename convention is ``{sample_id:06d}.flac``.
        """
        filename = f"{int(sample_id):06d}.flac"
        return self.audio_root / filename

    @staticmethod
    def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
        """Compute the hex SHA-256 digest of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()

    def _verify_checksum(self, path: Path, expected_hex: str) -> None:
        """Raise ``RuntimeError`` if the file checksum does not match."""
        actual = self._sha256(path)
        if actual != expected_hex:
            raise RuntimeError(
                f"Checksum mismatch for {path}.\n"
                f"  Expected : {expected_hex}\n"
                f"  Actual   : {actual}"
            )

    def _load_audio(self, path: Path) -> torch.Tensor:
        """Load a .flac file and return a ``(channels, time)`` float32 tensor.

        Parameters
        ----------
        path:
            Absolute path to the ``.flac`` file.

        Returns
        -------
        torch.Tensor
            Shape ``(4, T)`` — 4-channel FOA waveform, dtype ``float32``.

        Raises
        ------
        FileNotFoundError
            If the audio file is missing from ``audio_root``.
        RuntimeError
            If the file's sample rate does not match ``self.sample_rate``.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {path}\n"
                "Ensure the dataset has been downloaded and "
                "audio_root points to the correct directory."
            )

        waveform, sr = torchaudio.load(str(path))  # (C, T), int16 → float32

        if sr != self.sample_rate:
            raise RuntimeError(
                f"Unexpected sample rate {sr} Hz for {path}. "
                f"Expected {self.sample_rate} Hz."
            )

        # Normalise to [-1, 1] float32 (torchaudio already returns float32)
        return waveform.float()

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the sample at ``index``.

        Returns
        -------
        waveform : torch.Tensor
            4-channel FOA waveform, shape ``(4, T)``, dtype ``float32``.
        azimuth : torch.Tensor
            Scalar tensor (radians).  Horizontal angle between speech source
            and microphone array.
        elevation : torch.Tensor
            Scalar tensor (radians).  Vertical angle between speech source
            and microphone array.
        """
        row = self.metadata.iloc[index]

        # ---- targets ---------------------------------------------------
        azimuth = torch.tensor(float(row["speech/azimuth"]), dtype=torch.float32)
        elevation = torch.tensor(float(row["speech/elevation"]), dtype=torch.float32)

        # ---- audio path ------------------------------------------------
        audio_path = self._build_audio_path(row["sample_id"])

        # ---- optional checksum verification ----------------------------
        if self.verify_checksums:
            expected_checksum = row.get("audio_info/checksum/ambisonics")
            if expected_checksum:
                self._verify_checksum(audio_path, str(expected_checksum))

        # ---- load waveform ---------------------------------------------
        waveform = self._load_audio(audio_path)

        # ---- optional transforms ---------------------------------------
        if self.transform is not None:
            waveform = self.transform(waveform)

        if self.target_transform is not None:
            azimuth_f, elevation_f = self.target_transform(
                (azimuth.item(), elevation.item())
            )
            azimuth = torch.tensor(azimuth_f, dtype=torch.float32)
            elevation = torch.tensor(elevation_f, dtype=torch.float32)

        return waveform, azimuth, elevation

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_sample_id(self, index: int) -> int:
        """Return the dataset ``sample_id`` for a given positional index."""
        return int(self.metadata.iloc[index]["sample_id"])

    def get_audio_path(self, index: int) -> Path:
        """Return the resolved ``.flac`` ``Path`` for a given positional index."""
        return self._build_audio_path(self.get_sample_id(index))

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame with key columns for the active split.

        Includes: ``sample_id``, ``split``, ``speech/azimuth``,
        ``speech/elevation``, ``speech/distance``,
        ``audio_info/duration``.
        """
        cols = [
            "sample_id",
            "split",
            "speech/azimuth",
            "speech/elevation",
            "speech/distance",
            "audio_info/duration",
        ]
        available = [c for c in cols if c in self.metadata.columns]
        return self.metadata[available].copy()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split={self.split!r}, "
            f"n_samples={len(self)}, "
            f"audio_root={self.audio_root!r}, "
            f"lite_version_only={self.lite_version_only}, "
            f"verify_checksums={self.verify_checksums}"
            f")"
        )
