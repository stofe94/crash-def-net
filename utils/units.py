"""
units.py — Units of a dataset and their conversion to SI.

The exporter keeps the raw data in their own units (LS-DYNA models mostly in mm and
MPa) and records the units of length and stress (``--length-unit``,
``--stress-unit``) in ``metadata.json`` and in the ``info`` table of ``dataset.db``
(``length_unit``, ``stress_unit``; missing: m and Pa, the units of
``deforming_plate``). Models, contact radius and position noise work in the units of
the data. The rollout converts its metrics to SI (positions and velocities in m, the
stress in Pa), so that every report reads them alike. The reports show lengths in mm
and the stress in kPa for data in Pa or kPa (the units of the paper tables), else in
the stress unit of the data, e.g. MPa (``stress_display``). Result fields other than
the stress keep the units of the data.
"""

from __future__ import annotations

from typing import Dict, Tuple

LENGTH_UNITS = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}                 # factor to m
STRESS_UNITS = {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9}  # factor to Pa
SI_LENGTH, SI_STRESS = "m", "Pa"


def si_factors(info: Dict) -> Tuple[float, float]:
    """Factors that turn lengths and stresses of a dataset into m and Pa; ``info``:
    its ``metadata.json`` or ``info`` table."""
    return (LENGTH_UNITS[info.get("length_unit", SI_LENGTH)],
            STRESS_UNITS[info.get("stress_unit", SI_STRESS)])


def stress_display(stress_unit: str = SI_STRESS) -> Tuple[float, str]:
    """Factor from Pa and the unit in which the reports show the stress of data in
    ``stress_unit``: kPa for data in Pa or kPa, else the unit of the data."""
    unit = "kPa" if STRESS_UNITS[stress_unit] < STRESS_UNITS["MPa"] else stress_unit
    return 1.0 / STRESS_UNITS[unit], unit
