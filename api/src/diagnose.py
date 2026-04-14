from typing import Optional

SEGMENTS = [
    "furnace_to_2nd_strike",
    "2nd_to_3rd_strike",
    "3rd_to_4th_strike",
    "4th_strike_to_aux_press",
    "aux_press_to_bath",
]

CAUSES = {
    "furnace_to_2nd_strike": [
        "Billet pick", "gripper close", "grip retries",
        "trajectory", "permissions", "queues",
    ],
    "2nd_to_3rd_strike": [
        "Retraction", "gripper", "press/PLC handshake",
        "wait points", "regrip",
    ],
    "3rd_to_4th_strike": [
        "Retraction", "conservative trajectory", "synchronization",
        "positioning", "confirmations",
    ],
    "4th_strike_to_aux_press": [
        "Pick micro-corrections", "transfer",
        "queue at Auxiliary Press entry", "interlocks",
    ],
    "aux_press_to_bath": [
        "Retraction", "transport", "bath queues",
        "permissions", "bath deposit",
    ],
}


def _compute_partials(piece: dict) -> dict:
    t2 = piece.get("lifetime_2nd_strike_s")
    t3 = piece.get("lifetime_3rd_strike_s")
    t4 = piece.get("lifetime_4th_strike_s")
    ta = piece.get("lifetime_auxiliary_press_s")
    tb = piece.get("lifetime_bath_s")

    def diff(a, b):
        return None if (a is None or b is None) else round(a - b, 1)

    return {
        "furnace_to_2nd_strike": None if t2 is None else round(t2, 1),
        "2nd_to_3rd_strike": diff(t3, t2),
        "3rd_to_4th_strike": diff(t4, t3),
        "4th_strike_to_aux_press": diff(ta, t4),
        "aux_press_to_bath": diff(tb, ta),
    }


def diagnose(piece: dict, reference_times: dict) -> dict:
    matrix_key = str(piece["die_matrix"])
    refs = reference_times[matrix_key]
    partials = _compute_partials(piece)

    segments_out = []
    for seg in SEGMENTS:
        actual = partials[seg]
        ref = refs[seg]

        if actual is None:
            segments_out.append({
                "segment": seg,
                "actual_s": None,
                "reference_s": ref,
                "deviation_s": None,
                "penalized": None,
            })
            continue

        deviation = round(actual - ref, 1)

        if deviation > 5.0:
            penalized = None
        elif deviation > 1.0:
            penalized = True
        else:
            penalized = False

        segments_out.append({
            "segment": seg,
            "actual_s": actual,
            "reference_s": ref,
            "deviation_s": deviation,
            "penalized": penalized,
        })

    delay = any(s["penalized"] is True for s in segments_out)

    probable_causes: list[str] = []
    for seg in SEGMENTS:
        for s in segments_out:
            if s["segment"] == seg and s["penalized"] is True:
                probable_causes.extend(CAUSES[seg])

    return {
        "piece_id": piece["piece_id"],
        "die_matrix": piece["die_matrix"],
        "delay": delay,
        "segments": segments_out,
        "probable_causes": probable_causes,
    }
