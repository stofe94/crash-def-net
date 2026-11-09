"""
radioss_deck.py — Read an OpenRadioss starter deck next to a d3plot.

A d3plot converted from OpenRadioss animation files holds the deformation and the
results, not the model definition. ``read_deck`` takes that from the starter deck
(``starter_deck``: ``*_0000.rad`` in the run folder; the engine deck ``*_0001.rad``
holds nothing of it):

* ``shell_thickness``: the thickness of ``/PROP/SHELL`` of every ``/PART``, by part ID.
* ``walls``: stationary ``/RWALL/CYL`` as (W, 7): point on the axis, unit axis, radius.
  The solver computes contact with this analytic wall, so its geometry comes from here
  and not from the display model in the d3plot. A wall is treated as infinite; moving
  walls and other wall types are not read (warning).
* ``rigid_titles``: the titles of ``/RWALL`` and ``/RBODY``: the d3plot names the parts
  that show them by these titles.
* ``clamped``: node IDs that ``/BCS`` fixes in x, y and z.
* ``ties``: ``/INTER/TYPE2`` as (tied node IDs, part IDs of the surface they are tied
  to). Node groups: ``/GRNOD/NODE``, ``/GRNOD/GRNOD``; surfaces: ``/SURF/PART``; others
  are not read (warning).
* ``initial_velocity``: ``/INIVEL/TRA`` as (velocity, node IDs, part IDs) of its node group,
  which may also be ``/GRNOD/PART`` (the nodes of these parts); other kinds of ``/INIVEL``
  and velocities in a skew frame are not read (warning).

Cards in fixed format: integers in 10 characters, floats in 20. ``#include`` is not
followed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

DECK_PATTERN = "*_0000.rad"   # the starter deck of a run
WALL_COLUMNS = 7              # point on the axis, unit axis, radius
_COMMENT_MARKS = ("#", "$")


@dataclass
class RadiossDeck:
    shell_thickness: Dict[int, float] = field(default_factory=dict)   # part ID -> thickness
    walls: np.ndarray = field(default_factory=lambda: np.zeros((0, WALL_COLUMNS), np.float32))
    rigid_titles: Set[str] = field(default_factory=set)
    clamped: Set[int] = field(default_factory=set)
    ties: List[Tuple[Set[int], List[int]]] = field(default_factory=list)
    initial_velocity: List[Tuple[np.ndarray, Set[int], Set[int]]] = field(default_factory=list)


def starter_deck(run_dir) -> Optional[Path]:
    """The starter deck of a run, None without one."""
    decks = sorted(Path(run_dir).glob(DECK_PATTERN))
    if len(decks) > 1:
        raise ValueError(f"{run_dir}: several starter decks ({', '.join(d.name for d in decks)})")
    return decks[0] if decks else None


def _read_blocks(path) -> Dict[str, List[str]]:
    """Keyword line -> its cards, comments removed."""
    blocks: Dict[str, List[str]] = {}
    cards: Optional[List[str]] = None
    for line in Path(path).read_text(errors="replace", encoding="utf-8").splitlines():
        if line.startswith("/"):
            cards = blocks.setdefault(line.strip(), [])
        elif cards is not None and line.strip() and not line.startswith(_COMMENT_MARKS):
            cards.append(line)
    return blocks


def _items(blocks: Dict[str, List[str]], prefix: str) -> Iterator[Tuple[int, List[str]]]:
    """(ID, cards) of the keywords ``<prefix>/<ID>``."""
    for keyword, cards in blocks.items():
        if keyword.startswith(prefix + "/"):
            yield int(keyword[len(prefix) + 1:].split("/")[0]), cards


def _integer(card: str, start: int) -> int:
    """The integer in the 10 characters from ``start``; blank = 0."""
    value = card[start:start + 10].strip()
    return int(value) if value else 0


def _floats(card: str, count: int) -> List[float]:
    """The first ``count`` fields of 20 characters; blank = 0."""
    fields = (card[start:start + 20].strip() for start in range(0, 20 * count, 20))
    return [float(value) if value else 0.0 for value in fields]


def _ids(cards: List[str]) -> List[int]:
    """All IDs of the cards of a group, without its title card."""
    return [int(value) for card in cards[1:] for value in card.split()]


def _group_nodes(blocks: Dict[str, List[str]], group: int) -> Set[int]:
    """Node IDs of a node group."""
    if f"/GRNOD/NODE/{group}" in blocks:
        return set(_ids(blocks[f"/GRNOD/NODE/{group}"]))
    if f"/GRNOD/GRNOD/{group}" in blocks:
        return set().union(*(_group_nodes(blocks, member)
                             for member in _ids(blocks[f"/GRNOD/GRNOD/{group}"])))
    warnings.warn(f"node group {group} is no /GRNOD/NODE or /GRNOD/GRNOD; it is ignored")
    return set()


def _group_members(blocks: Dict[str, List[str]], group: int) -> Tuple[Set[int], Set[int]]:
    """Node IDs and part IDs of a node group that may also hold whole parts."""
    if f"/GRNOD/PART/{group}" in blocks:
        return set(), set(_ids(blocks[f"/GRNOD/PART/{group}"]))
    if f"/GRNOD/GRNOD/{group}" in blocks:
        nodes: Set[int] = set()
        parts: Set[int] = set()
        for member in _ids(blocks[f"/GRNOD/GRNOD/{group}"]):
            member_nodes, member_parts = _group_members(blocks, member)
            nodes |= member_nodes
            parts |= member_parts
        return nodes, parts
    return _group_nodes(blocks, group), set()


def _read_initial_velocity(blocks: Dict[str, List[str]]
                           ) -> List[Tuple[np.ndarray, Set[int], Set[int]]]:
    definitions = []
    for keyword, cards in blocks.items():
        if not keyword.startswith("/INIVEL/"):
            continue
        if not keyword.startswith("/INIVEL/TRA/"):
            warnings.warn(f"{keyword} is not supported; the initial velocity is ignored")
        elif _integer(cards[1], 70):
            warnings.warn(f"{keyword}: velocities in a skew frame are not supported; the "
                          "initial velocity is ignored")
        else:
            definitions.append((np.array(_floats(cards[1], 3), dtype=np.float32),
                                *_group_members(blocks, _integer(cards[1], 60))))
    return definitions


def _read_walls(blocks: Dict[str, List[str]]) -> Tuple[np.ndarray, Set[str]]:
    walls, titles = [], set()
    for keyword, cards in blocks.items():
        if not keyword.startswith("/RWALL/"):
            continue
        titles.add(cards[0].strip())
        if not keyword.startswith("/RWALL/CYL/"):
            warnings.warn(f"{keyword} is not supported; the wall is ignored")
        elif _integer(cards[1], 0):
            warnings.warn(f"{keyword}: moving walls are not supported; the wall is ignored")
        else:
            radius = _floats(cards[2], 3)[2] / 2
            point = np.array(_floats(cards[3], 3))
            axis = np.array(_floats(cards[4], 3)) - point
            walls.append(np.concatenate([point, axis / np.linalg.norm(axis), [radius]]))
    return np.array(walls, dtype=np.float32).reshape(-1, WALL_COLUMNS), titles


def read_deck(path) -> RadiossDeck:
    """What a starter deck adds to a d3plot (module docstring)."""
    blocks = _read_blocks(path)
    thickness = {prop: _floats(cards[3][20:], 1)[0] for prop, cards in _items(blocks, "/PROP/SHELL")}
    deck = RadiossDeck()
    for part, cards in _items(blocks, "/PART"):
        prop = _integer(cards[1], 0)
        if prop in thickness:
            deck.shell_thickness[part] = thickness[prop]
    deck.walls, deck.rigid_titles = _read_walls(blocks)
    deck.rigid_titles |= {cards[0].strip() for _, cards in _items(blocks, "/RBODY")}
    deck.initial_velocity = _read_initial_velocity(blocks)
    for _, cards in _items(blocks, "/BCS"):
        if cards[1][3:6] == "111":
            deck.clamped |= _group_nodes(blocks, _integer(cards[1], 20))
    for keyword, cards in blocks.items():
        if keyword.startswith("/INTER/TYPE2/"):
            surface = f"/SURF/PART/{_integer(cards[1], 10)}"
            if surface in blocks:
                deck.ties.append((_group_nodes(blocks, _integer(cards[1], 0)),
                                  _ids(blocks[surface])))
            else:
                warnings.warn(f"{keyword}: its surface is no /SURF/PART; the tie is ignored")
    return deck
