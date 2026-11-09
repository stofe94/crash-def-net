"""
export_config.py — The TOML file of an export and the roles of the components.

``export.py --config FILE`` takes its settings from a TOML file, so that a dataset is
defined before it is generated: which components it holds, what each of them is, and
which features it has. ``EXPORT_CONFIG_KEYS`` maps every key to an option of
``export.py``, whose help text describes it; options given on the command line override
the file, unknown keys are rejected.

Components
----------
``[components]`` (``--components``) lists what goes into the dataset, as
``pattern = role``. Patterns are ``fnmatch`` patterns on the component names
(``--list-parts`` shows them). A component that no pattern matches is left out, with the
nodes only it holds. A pattern without a match and a component with two roles are errors.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import toml

ROLE_DEFORMABLE = "deformable"   # the model predicts the motion
ROLE_ACTUATOR = "actuator"       # motion given by the run
ROLE_FIXED = "fixed"             # clamped
ROLE_FREE_RIGID = "free_rigid"   # the model predicts the motion, the rollout keeps it rigid
ROLES = (ROLE_DEFORMABLE, ROLE_ACTUATOR, ROLE_FIXED, ROLE_FREE_RIGID)

# TOML key -> option of export.py (dest of its parser), in the order of the reference.
EXPORT_CONFIG_KEYS = {
    "raw.format": "raw_format",
    "raw.dir": "raw_dir",
    "raw.sources": "sources",
    "raw.vtk.stress_field": "stress_field",
    "raw.vtk.node_type_field": "node_type_field",
    "raw.vtk.displacement_field": "displacement_field",
    "raw.vtk.node_id_field": "node_id_field",
    "raw.vtk.part_field": "part_field",
    "dataset.out_dir": "out_dir",
    "dataset.n_train": "n_train",
    "dataset.n_val": "n_val",
    "dataset.n_test": "n_test",
    "dataset.select_seed": "select_seed",
    "dataset.num_steps": "num_steps",
    "dataset.time_stride": "time_stride",
    "dataset.history": "history",
    "dataset.prev_stress": "prev_stress",
    "units.length": "length_unit",
    "units.stress": "stress_unit",
    "contact.radius": "contact_radius",
    "contact.pairs": "contact_pairs",
    "contact.use_deck_walls": "use_deck_walls",
    "features.local": "static_fields",
    "features.global": "global_features",
    "features.globals_file": "globals_file",
    "features.targets": "fields",
    "large.block_mb": "block_mb",
    "large.edge_stats_stride": "edge_stats_stride",
}
COMPONENTS_SECTION = "components"


def _flatten(table: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in table.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and not (not prefix and key == COMPONENTS_SECTION):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def read_export_config(path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The content of an export config and the parser defaults it sets."""
    config = toml.loads(Path(path).read_text(encoding="utf-8"))
    flat = _flatten(config)
    unknown = sorted(set(flat) - set(EXPORT_CONFIG_KEYS) - {COMPONENTS_SECTION})
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}; these are known: "
                         f"{list(EXPORT_CONFIG_KEYS)} and [{COMPONENTS_SECTION}]")
    defaults = {EXPORT_CONFIG_KEYS[key]: value for key, value in flat.items()
                if key != COMPONENTS_SECTION}
    if COMPONENTS_SECTION in flat:
        defaults[COMPONENTS_SECTION] = flat[COMPONENTS_SECTION]
    if isinstance(defaults.get("sources"), str):
        defaults["sources"] = [defaults["sources"]]
    if "contact_pairs" in defaults:
        defaults["contact_pairs"] = [":".join(pair) for pair in defaults["contact_pairs"]]
    return config, defaults


def parse_components(value: Union[Dict[str, str], Sequence[str], None]) -> Dict[str, str]:
    """``pattern -> role`` of the config section, or of ``PATTERN=ROLE`` options."""
    if not value:
        return {}
    pairs = dict(value) if isinstance(value, dict) else dict(item.rsplit("=", 1) for item in value
                                                             if "=" in item)
    if not isinstance(value, dict) and len(pairs) != len(value):
        raise ValueError(f"components need the form PATTERN=ROLE, got {list(value)}")
    unknown = {pattern: role for pattern, role in pairs.items() if role not in ROLES}
    if unknown:
        raise ValueError(f"components with unknown roles {unknown}; the roles are {list(ROLES)}")
    return pairs


def _matches(name: str, pattern: str) -> bool:
    return name == pattern or fnmatchcase(name, pattern)


def component_roles(names: Sequence[str], components: Dict[str, str],
                    where: str = "") -> Dict[str, str]:
    """The role of every component in ``names`` that a pattern matches; the others are
    left out."""
    roles: Dict[str, str] = {}
    for pattern, role in components.items():
        hits = [name for name in names if _matches(name, pattern)]
        if not hits:
            raise ValueError(f"{where}: component pattern {pattern!r} matches no part of the "
                             f"run; these are {sorted(names)}")
        for name in hits:
            if roles.setdefault(name, role) != role:
                raise ValueError(f"{where}: part {name!r} has two roles, {roles[name]} and "
                                 f"{role} (pattern {pattern!r})")
    return roles


def expand_contact_pairs(pairs: Sequence[Sequence[str]], names: Sequence[str]) -> List[List[str]]:
    """Contact pairs with patterns as pairs of component names, each pair once."""
    expanded: List[List[str]] = []
    for first, second in pairs:
        hits = [[name for name in names if _matches(name, pattern)] for pattern in (first, second)]
        if not all(hits):
            raise ValueError(f"contact pair {first}:{second} matches no component; "
                             f"these are {list(names)}")
        if first == second:
            found = list(combinations_with_replacement(hits[0], 2))
        else:
            found = [(a, b) for a in hits[0] for b in hits[1]]
        for pair in found:
            if list(pair) not in expanded and [pair[1], pair[0]] not in expanded:
                expanded.append(list(pair))
    return expanded
