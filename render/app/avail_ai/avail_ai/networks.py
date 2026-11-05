# -*- coding: utf-8 -*-
"""Yeo-17 -> Yeo-7 network abbreviation mapping (matches the previous paper).

The previous paper's fMRI naming uses short Yeo-7 canonical labels, e.g.
`DMN`, `DAN`, `VAN`, `SMN`, `LIM`, `FPN`, `VIS`, `TP`, with the convention:
  - within-network edge  -> "within DMN"
  - between-network edge -> "DAN-DMN" (alphabetically sorted)
"""
NETWORK_FAMILIES = {
    "Default A": "DMN", "Default B": "DMN", "Default C": "DMN",
    "Dorsal Attention A": "DAN", "Dorsal Attention B": "DAN",
    "Salience Ventral Attention A": "VAN", "Salience Ventral Attention B": "VAN",
    "Somatomotor A": "SMN", "Somatomotor B": "SMN", "Somatomotor B Auditory": "SMN",
    "Limbic": "LIM", "Limbic Orbitofrontal": "LIM", "Limbic Temporopolar": "LIM",
    "Frontoparietal Control A": "FPN", "Frontoparietal Control B": "FPN",
    "Frontoparietal Control C": "FPN",
    "Visual Central": "VIS", "Visual Peripheral": "VIS",
    "Temporoparietal": "TP",
}


def abbrev_network(name):
    """Map a Yeo-17 network name to its short Yeo-7 label ('Default A' -> 'DMN')."""
    key = str(name).strip()
    if key.startswith("Somatomotor B"):
        key = "Somatomotor B Auditory" if "Auditory" in key else "Somatomotor B"
    return NETWORK_FAMILIES.get(key, key)


def abbrev_brain_label(label):
    """Shorten an fc summary column label to a Yeo-7 family label.

    'within_Default A'                        -> 'within DMN'
    'between_Default A__Default B'            -> 'within DMN' (same family)
    'between_Default A__Dorsal Attention B'   -> 'DAN-DMN'
    """
    s = str(label)
    if s.startswith("within_"):
        return "within " + abbrev_network(s[len("within_"):])
    if s.startswith("between_"):
        parts = s[len("between_"):].split("__")
        if len(parts) == 2:
            a, b = abbrev_network(parts[0]), abbrev_network(parts[1])
            return ("within " + a) if a == b else "-".join(sorted([a, b]))
    return s


def fam_label(col):
    """Alias of :func:`abbrev_brain_label` for fc column names."""
    return abbrev_brain_label(col)
