"""Почему Spectra-0 даёт ложную синтетику на PACK2.

Сравниваются ложные срабатывания внутри пакета с верными живыми и с TTS.
Обучения нет. Порог не двигается.
"""

from __future__ import annotations

import pandas as pd

from src.research.model_assisted_annotation.common_misses import (
    AASIST_THR,
    KEY_FEATURES,
    ORIGIN,
    REPORT,
    ROOT,
    SPECTRA_THR,
    audio_levels,
    median,
    read_csv,
    windows,
)


def cohort() -> dict[str, list[str]]:
    pred = {row["call_id"]: row for row in read_csv(REPORT / "per_call_predictions.csv")}
    labels = {
        row["call_id"]: row
        for row in read_csv(ROOT / "data" / "annotations" / "priority_20261111_20261112_calls.csv")
    }
    groups = {
        "fp_pack2": [],
        "human_pack2": [],
        "tts_pack2": [],
        "human_pack1": [],
        "tts_pack1": [],
    }
    for call_id, label in labels.items():
        row = pred[call_id]
        pack = "pack1" if str(row["campaign"]).endswith("PACK1") else "pack2" if str(row["campaign"]).endswith("PACK2") else ""
        if not pack:
            continue
        primary = label["primary_label"]
        spectra = row.get("spectra0_prediction")
        if primary == "human_live" and pack == "pack2" and spectra == "synthetic":
            groups["fp_pack2"].append(call_id)
        elif primary == "human_live" and pack == "pack2" and spectra == "human":
            groups["human_pack2"].append(call_id)
        elif primary == "synthetic_tts" and pack == "pack2":
            groups["tts_pack2"].append(call_id)
        elif primary == "human_live" and pack == "pack1" and spectra == "human":
            groups["human_pack1"].append(call_id)
        elif primary == "synthetic_tts" and pack == "pack1":
            groups["tts_pack1"].append(call_id)
    return groups


def main() -> None:
    groups = cohort()
    for name, ids in groups.items():
        print(name, len(ids))
    pred = pd.DataFrame(read_csv(REPORT / "per_call_predictions.csv")).set_index("call_id")
    ids = [call_id for values in groups.values() for call_id in values]
    levels = audio_levels(ids).set_index("call_id")
    acoustic_cols = ["spectra_speech_duration_s", "spectra_n_segments", "spectra_n_windows", "padding_ratio", "spectra0_score"]
    acoustic = pred.loc[ids, acoustic_cols].apply(pd.to_numeric, errors="coerce").join(levels)
    print("\n--- медианы ---")
    for column in acoustic.columns:
        parts = [f"{name} {median(acoustic.loc[groups[name], column]):.3f}" for name in groups if groups[name]]
        print(column, " | ".join(parts))

    print("\n--- согласие других моделей на 17 FP ---")
    fp = groups["fp_pack2"]
    for model in ORIGIN:
        if model == "spectra0":
            continue
        synthetic = sum(pred.loc[call_id, f"{model}_prediction"] == "synthetic" for call_id in fp)
        print(model, synthetic, "из", len(fp))

    print("\n--- дни FP ---")
    print(pred.loc[fp, "date"].value_counts().to_string())
    print("amd", pred.loc[fp, "amd_campaign"].value_counts().to_string())

    print("\n--- окна FP ---")
    frame = windows(set(fp) | set(groups["human_pack2"][:0]) | set(groups["tts_pack2"]))
    spectra = frame[frame["model"] == "spectra0"]
    for call_id in fp:
        part = spectra[spectra["call_id"] == call_id]
        scores = part["decision_score"]
        spoof = scores < SPECTRA_THR
        padded = part["padded_to_model_length"].astype(str).str.lower().eq("true")
        valid = pd.to_numeric(part["valid_speech_audio_sec"], errors="coerce")
        print(
            call_id[:8],
            "score",
            f"{float(pred.loc[call_id, 'spectra0_score']):.3f}",
            "окон",
            len(part),
            "spoof",
            int(spoof.sum()),
            "min",
            f"{scores.min():.2f}",
            "med",
            f"{scores.median():.2f}",
            "padded",
            int(padded.sum()),
            "valid_med",
            f"{valid.median():.2f}",
            "speech",
            pred.loc[call_id, "spectra_speech_duration_s"],
            "pad",
            pred.loc[call_id, "padding_ratio"],
        )

    print("\n--- доля spoof-окон по группам ---")
    all_ids = groups["fp_pack2"] + groups["human_pack2"] + groups["tts_pack2"] + groups["human_pack1"] + groups["tts_pack1"]
    all_windows = windows(set(all_ids))
    spectra_all = all_windows[all_windows["model"] == "spectra0"]
    for name, call_ids in groups.items():
        part = spectra_all[spectra_all["call_id"].isin(call_ids)]
        scores = part["decision_score"]
        spoof = (scores < SPECTRA_THR).mean() if len(scores) else float("nan")
        padded = part["padded_to_model_length"].astype(str).str.lower().eq("true").mean() if len(part) else float("nan")
        valid = pd.to_numeric(part["valid_speech_audio_sec"], errors="coerce").median()
        print(name, "окон", len(part), "spoof", f"{spoof:.3f}", "padded", f"{padded:.3f}", "valid_med", f"{valid:.2f}", "margin_med", f"{scores.median():.2f}")

    features = pd.read_csv(REPORT / "new_classical_features.csv", usecols=lambda name: name == "call_id" or name in KEY_FEATURES).set_index("call_id")
    print("\n--- признаки ---")
    for column in KEY_FEATURES:
        if column not in features.columns:
            continue
        parts = []
        for name, call_ids in groups.items():
            present = features.index.intersection(call_ids)
            parts.append(f"{name} {median(features.loc[present, column]):.3f}")
        print(column, " | ".join(parts))

    print("\naasist thr reference", AASIST_THR)


if __name__ == "__main__":
    main()
