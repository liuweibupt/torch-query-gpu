"""Typed slot references carried by TQP operator graphs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TQPSlotRef:
    """Reference to a node output slot, independent of name-vs-ordinal syntax."""

    slot_id: str
    node_id: str
    ordinal: int
    name: str


@dataclass(frozen=True)
class TQPSlot:
    """One typed output position produced by a TQP operator node."""

    slot_id: str
    node_id: str
    ordinal: int
    name: str
    type_name: str | None = None
    aliases: tuple[str, ...] = ()
    origin_slot_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", tuple(dict.fromkeys((self.name, *self.aliases))))

    @property
    def ref(self) -> TQPSlotRef:
        return TQPSlotRef(self.slot_id, self.node_id, self.ordinal, self.name)


@dataclass(frozen=True)
class TQPBoundExpression:
    """Expression text plus resolved input slot references."""

    raw: str
    canonical: str
    refs: tuple[TQPSlotRef, ...]
    unresolved: tuple[str, ...] = ()
    output_slot: TQPSlot | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "refs", tuple(self.refs))
        object.__setattr__(self, "unresolved", tuple(dict.fromkeys(self.unresolved)))
