"""Physical column storage for TensorRecordBatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import torch

from tpch_torch.record_batch_types import AllocationOwner, StorageKind


@dataclass(frozen=True)
class ColumnStorage:
    """Physical tensor storage for one typed column."""

    kind: StorageKind
    data: torch.Tensor
    torch_dtype: torch.dtype
    validity: torch.Tensor | None = None
    children: Mapping[str, torch.Tensor] = field(default_factory=dict)
    dictionary: tuple[str, ...] | None = None
    owner: AllocationOwner | None = None
    is_view: bool = False
    parent_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", MappingProxyType(dict(self.children)))
        if self.owner is None:
            object.__setattr__(self, "owner", AllocationOwner.torch())
        _validate_storage(self)

    @classmethod
    def fixed(
        cls,
        data: torch.Tensor,
        *,
        validity: torch.Tensor | None = None,
        owner: AllocationOwner | None = None,
    ) -> "ColumnStorage":
        return cls(StorageKind.FIXED, data, data.dtype, validity=validity, owner=owner)

    @classmethod
    def decimal64(
        cls,
        data: torch.Tensor,
        *,
        validity: torch.Tensor | None = None,
        owner: AllocationOwner | None = None,
    ) -> "ColumnStorage":
        if data.dtype != torch.int64:
            raise TypeError("decimal64 storage requires torch.int64 data")
        return cls(StorageKind.DECIMAL64, data, data.dtype, validity=validity, owner=owner)

    @classmethod
    def dictionary_ids(
        cls,
        ids: torch.Tensor,
        vocabulary: Sequence[str],
        *,
        validity: torch.Tensor | None = None,
        owner: AllocationOwner | None = None,
    ) -> "ColumnStorage":
        if ids.dtype != torch.int64:
            raise TypeError("dictionary storage requires int64 ids")
        return cls(
            StorageKind.DICTIONARY,
            ids,
            ids.dtype,
            validity=validity,
            dictionary=tuple(vocabulary),
            owner=owner,
        )

    @classmethod
    def utf8_offsets(
        cls,
        values: Iterable[str | None],
        *,
        device: str | torch.device,
        owner: AllocationOwner | None = None,
    ) -> "ColumnStorage":
        offsets, chars, validity = _encode_utf8_offsets(tuple(values), torch.device(device))
        return cls(
            StorageKind.UTF8_OFFSETS,
            chars,
            chars.dtype,
            validity=validity,
            children={"offsets": offsets, "chars": chars},
            owner=owner,
        )

    @property
    def row_count(self) -> int:
        if self.kind == StorageKind.UTF8_OFFSETS:
            return int(self.children["offsets"].numel()) - 1
        return int(self.data.shape[0])

    @property
    def device(self) -> torch.device:
        return self.data.device

    def filter(self, mask: torch.Tensor) -> "ColumnStorage":
        if self.kind == StorageKind.UTF8_OFFSETS:
            return self._select_utf8(mask.nonzero(as_tuple=False).reshape(-1))
        validity = None if self.validity is None else self.validity[mask]
        return self._replace_data(self.data[mask], validity=validity)

    def gather(self, indices: torch.Tensor) -> "ColumnStorage":
        if indices.dtype != torch.int64:
            indices = indices.to(dtype=torch.int64)
        if self.kind == StorageKind.UTF8_OFFSETS:
            return self._select_utf8(indices)
        validity = None if self.validity is None else self.validity.index_select(0, indices)
        return self._replace_data(self.data.index_select(0, indices), validity=validity)

    def decode_utf8(self) -> list[str | None]:
        if self.kind != StorageKind.UTF8_OFFSETS:
            raise TypeError("decode_utf8 requires UTF8_OFFSETS storage")
        if self.data.device.type != "cpu":
            raise NotImplementedError("UTF8 decode is CPU-only; no implicit device fallback")
        offsets = self.children["offsets"].tolist()
        chars = bytes(self.children["chars"].tolist())
        validity = None if self.validity is None else self.validity.tolist()
        return [_decode_utf8_row(offsets, chars, validity, index) for index in range(len(offsets) - 1)]

    def _replace_data(self, data: torch.Tensor, *, validity: torch.Tensor | None) -> "ColumnStorage":
        return ColumnStorage(
            self.kind,
            data,
            data.dtype,
            validity=validity,
            dictionary=self.dictionary,
            owner=self.owner,
            is_view=False,
            parent_id=self.parent_id,
        )

    def _select_utf8(self, indices: torch.Tensor) -> "ColumnStorage":
        if self.data.device.type != "cpu" or indices.device.type != "cpu":
            raise NotImplementedError("UTF8 compaction is CPU-only; no implicit device fallback")
        decoded = self.decode_utf8()
        selected = [decoded[int(index)] for index in indices.tolist()]
        return ColumnStorage.utf8_offsets(selected, device=self.data.device, owner=self.owner)


def _validate_storage(storage: ColumnStorage) -> None:
    if storage.kind == StorageKind.UTF8_OFFSETS:
        _validate_utf8_storage(storage)
    elif storage.data.ndim == 0:
        raise ValueError("fixed-width storage data must be at least 1-D")
    if storage.validity is not None and storage.validity.dtype is not torch.bool:
        raise ValueError("validity mask must be boolean")


def _validate_utf8_storage(storage: ColumnStorage) -> None:
    if "offsets" not in storage.children or "chars" not in storage.children:
        raise ValueError("UTF8_OFFSETS storage requires offsets and chars children")
    offsets = storage.children["offsets"]
    chars = storage.children["chars"]
    if offsets.dtype != torch.int64 or chars.dtype != torch.uint8:
        raise TypeError("UTF8 offsets must be int64 and chars must be uint8")
    if offsets.device != chars.device or chars.device != storage.data.device:
        raise ValueError("UTF8 children must be on the storage device")


def _encode_utf8_offsets(
    values: tuple[str | None, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = [0]
    validity = []
    raw = bytearray()
    for value in values:
        validity.append(value is not None)
        if value is not None:
            raw.extend(value.encode("utf-8"))
        offsets.append(len(raw))
    chars = torch.tensor(list(raw), dtype=torch.uint8, device=device)
    return (
        torch.tensor(offsets, dtype=torch.int64, device=device),
        chars,
        torch.tensor(validity, dtype=torch.bool, device=device),
    )


def _decode_utf8_row(
    offsets: list[int],
    chars: bytes,
    validity: list[bool] | None,
    index: int,
) -> str | None:
    if validity is not None and not validity[index]:
        return None
    return chars[offsets[index] : offsets[index + 1]].decode("utf-8")
