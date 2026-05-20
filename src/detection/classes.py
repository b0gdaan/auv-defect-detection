"""
Defect class definitions for the Intelligent Defectoscope.

The CNN was trained to distinguish four states of hydraulic structure surfaces:
    0  normal         – no visible damage
    1  crack           – linear crack in concrete / masonry
    2  concrete_damage – spalling, crumbling, or section loss
    3  empty_joint     – missing mortar in stone / brick joints

Reference: bachelor's thesis, SamGTU 2023; RZD grant 2/22-РЖД/2022
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DefectClass:
    id:          int
    name:        str
    label_ru:    str
    color_bgr:   tuple[int, int, int]   # for OpenCV overlay
    severity:    int                    # 0 = info, 1 = warning, 2 = critical


CLASSES: list[DefectClass] = [
    DefectClass(0, "normal",         "Норма",                   (0, 200, 0),   0),
    DefectClass(1, "crack",          "Трещина",                 (0, 80, 255),  2),
    DefectClass(2, "concrete_damage","Разрушение бетона",       (0, 0, 220),   2),
    DefectClass(3, "empty_joint",    "Пустой шов",              (0, 165, 255), 1),
]

NUM_CLASSES = len(CLASSES)
CLASS_NAMES = [c.name for c in CLASSES]


def get_class(idx: int) -> DefectClass:
    return CLASSES[idx]


def get_color(idx: int) -> tuple[int, int, int]:
    return CLASSES[idx].color_bgr
