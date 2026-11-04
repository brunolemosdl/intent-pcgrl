from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

NPZ_FILENAME = "chunk_library.npz"
JSON_FILENAME = "chunk_library.json"
LIBRARY_FORMAT_VERSION = 1


class ChunkFamily(StrEnum):
    """Structural role of a chunk inside a composed level.

    Description:
        Labels where a chunk sits in the SMB1 level template: spawn region,
        middle traversal, or staircase/flag ending.
    """

    START = "start"
    MIDDLE = "middle"
    ENDING = "ending"


class Chunk(BaseModel):
    """Vertical slice of a real corpus level with provenance and annotations.

    Description:
        ``tiles`` keeps all native rows of the source level for a few columns.
        ``entry_height``/``exit_height`` are standable surface heights at the
        first and last columns (rows above the grid bottom; 0 = bottomless gap);
        the composer stitching mask compares exit-to-entry of consecutive
        chunks. ``gap_at_left``/``gap_at_right`` flag edges that begin or end
        mid-gap. ``category_counts`` caches tile counts per category for intent
        budget arithmetic.

    Attributes:
        chunk_id: Contiguous library index assigned at extraction time.
        tiles: Tile-id grid with shape ``(height, width)``.
        family: Structural role (start, middle, or ending).
        level_id: Source corpus level identifier.
        column_offset: Left column index in the source level.
        entry_height: Standable surface height at the left edge.
        exit_height: Standable surface height at the right edge.
        gap_at_left: Whether the left edge starts inside a floor gap.
        gap_at_right: Whether the right edge ends inside a floor gap.
        category_counts: Per-category tile counts inside the chunk.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    chunk_id: int
    tiles: np.ndarray
    family: ChunkFamily
    level_id: str
    column_offset: int
    entry_height: int
    exit_height: int
    gap_at_left: bool
    gap_at_right: bool
    category_counts: dict[str, int] = Field(default_factory=dict)

    @property
    def width(self) -> int:
        """Number of columns in the chunk."""
        return int(self.tiles.shape[1])

    @property
    def height(self) -> int:
        """Number of rows in the chunk."""
        return int(self.tiles.shape[0])

    def annotations_dict(self) -> dict[str, Any]:
        """Serialise every non-tile field for the JSON sidecar.

        Returns:
            Mapping suitable for the chunk-library JSON payload.
        """
        return {
            "chunk_id": self.chunk_id,
            "family": self.family.value,
            "level_id": self.level_id,
            "column_offset": self.column_offset,
            "entry_height": self.entry_height,
            "exit_height": self.exit_height,
            "gap_at_left": self.gap_at_left,
            "gap_at_right": self.gap_at_right,
            "category_counts": dict(self.category_counts),
        }

    @classmethod
    def from_annotations(cls, tiles: np.ndarray, data: dict[str, Any]) -> Chunk:
        """Rebuild a chunk from a tile grid plus its JSON annotations.

        Args:
            tiles: Tile-id grid for the chunk.
            data: Annotation mapping produced by ``annotations_dict``.

        Returns:
            Reconstructed ``Chunk`` instance.
        """
        return cls(
            chunk_id=int(data["chunk_id"]),
            tiles=tiles,
            family=ChunkFamily(data["family"]),
            level_id=str(data["level_id"]),
            column_offset=int(data["column_offset"]),
            entry_height=int(data["entry_height"]),
            exit_height=int(data["exit_height"]),
            gap_at_left=bool(data["gap_at_left"]),
            gap_at_right=bool(data["gap_at_right"]),
            category_counts={
                str(key): int(value)
                for key, value in dict(data.get("category_counts", {})).items()
            },
        )


class ChunkLibrary:
    """Deduplicated, annotated chunk collection with split metadata.

    Description:
        Persisted as ``chunk_library.npz`` (tile grids) plus
        ``chunk_library.json`` (annotations). ``tile_names`` freezes the
        id-to-name mapping at extraction time so a saved library stays decodable
        even if the runtime catalog ordering changes later. Construction
        enforces the by-level split: chunks from heldout levels are rejected.

    Attributes:
        chunks: Ordered list of annotated chunks.
        train_level_ids: Corpus level ids included in training.
        heldout_level_ids: Corpus level ids reserved for evaluation.
        tile_names: Tile-name ordering frozen at extraction time.
        metadata: Optional extraction provenance and statistics.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        train_level_ids: list[str],
        heldout_level_ids: list[str],
        tile_names: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Validate id contiguity and split integrity before storing.

        Args:
            chunks: Annotated chunks with contiguous ``chunk_id`` values.
            train_level_ids: Level ids allowed in the library.
            heldout_level_ids: Level ids that must not appear in ``chunks``.
            tile_names: Tile-name ordering to persist with the library.
            metadata: Optional extraction metadata.

        Raises:
            ValueError: When ids are not contiguous, splits overlap, or heldout
                levels leak into ``chunks``.
        """
        ids = [chunk.chunk_id for chunk in chunks]
        if ids != list(range(len(chunks))):
            raise ValueError("chunk_id values must be contiguous and start at 0")
        overlap = set(train_level_ids) & set(heldout_level_ids)
        if overlap:
            raise ValueError(f"levels present in both splits: {sorted(overlap)}")
        leaked = {
            chunk.level_id for chunk in chunks if chunk.level_id in set(heldout_level_ids)
        }
        if leaked:
            raise ValueError(f"library contains chunks from heldout levels: {sorted(leaked)}")
        self.chunks = list(chunks)
        self.train_level_ids = list(train_level_ids)
        self.heldout_level_ids = list(heldout_level_ids)
        self.tile_names = list(tile_names)
        self.metadata = dict(metadata or {})

    def __len__(self) -> int:
        """Return the number of chunks in the library."""
        return len(self.chunks)

    def __getitem__(self, index: int) -> Chunk:
        """Return the chunk at ``index``."""
        return self.chunks[index]

    def by_family(self, family: ChunkFamily) -> list[Chunk]:
        """Return chunks belonging to one structural family.

        Args:
            family: Structural role filter.

        Returns:
            Matching chunks in library order.
        """
        return [chunk for chunk in self.chunks if chunk.family == family]

    def widths(self) -> tuple[int, ...]:
        """Return sorted distinct chunk widths present in the library."""
        return tuple(sorted({chunk.width for chunk in self.chunks}))

    def save(self, directory: str | Path) -> tuple[Path, Path]:
        """Write the npz tile grids and the JSON annotation sidecar.

        Args:
            directory: Output directory (created when missing).

        Returns:
            Paths to the npz and JSON files.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        npz_path = directory / NPZ_FILENAME
        json_path = directory / JSON_FILENAME
        arrays = {
            f"chunk_{chunk.chunk_id:05d}": chunk.tiles.astype(np.int16)
            for chunk in self.chunks
        }
        np.savez_compressed(npz_path, **arrays)  # type: ignore[arg-type]
        payload = {
            "format_version": LIBRARY_FORMAT_VERSION,
            "tile_names": self.tile_names,
            "train_level_ids": self.train_level_ids,
            "heldout_level_ids": self.heldout_level_ids,
            "metadata": self.metadata,
            "chunks": [chunk.annotations_dict() for chunk in self.chunks],
        }
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8"
        )
        return npz_path, json_path

    @classmethod
    def load(cls, directory: str | Path) -> ChunkLibrary:
        """Load a saved library, validating presence and format version.

        Args:
            directory: Directory containing ``chunk_library.npz`` and
                ``chunk_library.json``.

        Returns:
            Reconstructed library instance.

        Raises:
            FileNotFoundError: When either library file is missing.
            ValueError: When ``format_version`` is unsupported.
        """
        directory = Path(directory)
        npz_path = directory / NPZ_FILENAME
        json_path = directory / JSON_FILENAME
        if not npz_path.is_file() or not json_path.is_file():
            raise FileNotFoundError(
                f"chunk library not found in {directory} "
                f"(expected {NPZ_FILENAME} and {JSON_FILENAME}; run the 'library' stage)"
            )
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        version = int(payload.get("format_version", 0))
        if version != LIBRARY_FORMAT_VERSION:
            raise ValueError(
                f"unsupported chunk library format_version {version} "
                f"(expected {LIBRARY_FORMAT_VERSION})"
            )
        with np.load(npz_path) as data:
            chunks = [
                Chunk.from_annotations(
                    data[f"chunk_{int(entry['chunk_id']):05d}"], entry
                )
                for entry in payload["chunks"]
            ]
        return cls(
            chunks,
            train_level_ids=[str(item) for item in payload["train_level_ids"]],
            heldout_level_ids=[str(item) for item in payload["heldout_level_ids"]],
            tile_names=[str(item) for item in payload["tile_names"]],
            metadata=dict(payload.get("metadata", {})),
        )
