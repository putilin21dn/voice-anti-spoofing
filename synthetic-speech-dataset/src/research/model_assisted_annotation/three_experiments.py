"""Три эксперимента на сохранённых окнах и прежних порогах.

Агрегации Spectra-0 сравниваются при границе 0.227. Голос 2 из 3 не переобучается.
Семь общих пропусков и 17 ложных PACK2 сравниваются по уже посчитанным признакам.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.model_assisted_annotation.common_misses import KEY_FEATURES, ORIGIN, REPORT, ROOT, read_csv

THR = 0.22710928916931153
WINDOW = 4.0375
SPEECH_LOW = 9.68
SPEECH_HIGH = 17.04
PAD_LOW = 0.432021
PAD_HIGH = 0.563181
HUMAN = {"human_live", "human_recorded"}
SYNTHETIC = {"synthetic_tts", "synthetic_voice_conversion", "synthetic_playback"}
RULES = ("mean", "median", "trimmed", "weighted", "mean_ge1")
RULE_NAMES = {
    "mean": "mean",
    "median": "median",
    "trimmed": "trimmed 10%",
    "weighted": "weighted",
    "mean_ge1": "mean, valid≥1 с",
}


def trimmed_mean(values: np.ndarray, proportion: float = 0.1) -> float:
    ordered = np.sort(values.astype(float))
    count = len(ordered)
    cut = int(proportion * count)
    if cut == 0 or count - 2 * cut <= 0:
        return float(ordered.mean())
    return float(ordered[cut : count - cut].mean())


def aggregate_call(group: pd.DataFrame) -> pd.Series:
    margin = group["margin"].to_numpy(dtype=float)
    valid = group["valid"].to_numpy(dtype=float)
    kept = valid >= 1.0
    weights = np.clip(valid, 0, None)
    weighted = float(np.average(margin, weights=weights)) if weights.sum() > 0 else float(margin.mean())
    padded = group["padded_to_model_length"].astype(str).str.lower().eq("true").to_numpy()
    extra = np.maximum(WINDOW - valid, 0.0) * padded
    denominator = float(valid.sum() + extra.sum())
    segments = group.groupby("segment_index")["valid"].sum()
    return pd.Series(
        {
            "mean": float(margin.mean()),
            "median": float(np.median(margin)),
            "trimmed": trimmed_mean(margin),
            "weighted": weighted,
            "mean_ge1": float(margin[kept].mean()) if kept.any() else np.nan,
            "window_count": int(len(margin)),
            "n_ge1": int(kept.sum()),
            "valid_sum": float(valid.sum()),
            "short_share": float((valid < 1.0).mean()),
            "spoof_share": float((margin < THR).mean()),
            "margin_min": float(margin.min()),
            "margin_p25": float(np.percentile(margin, 25)),
            "margin_med": float(np.median(margin)),
            "margin_p75": float(np.percentile(margin, 75)),
            "margin_max": float(margin.max()),
            "segment_median_s": float(segments.median()) if len(segments) else np.nan,
            "segment_min_s": float(segments.min()) if len(segments) else np.nan,
            "padding_calc": float(extra.sum() / denominator) if denominator else np.nan,
        }
    )


def windows() -> pd.DataFrame:
    paths = [
        ROOT / "evaluation" / "day_20261111" / "20261116T195124" / "window_results" / "spectra0.csv",
        ROOT / "evaluation" / "day_20261112" / "20261116T195754" / "window_results" / "spectra0.csv",
        ROOT / "evaluation" / "day_20261116" / "20261116T152222" / "window_results" / "spectra0.csv",
    ]
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame["margin"] = pd.to_numeric(frame["margin"], errors="coerce")
    frame["valid"] = pd.to_numeric(frame["valid_speech_audio_sec"], errors="coerce")
    frame = frame.dropna(subset=["margin", "valid"])
    return frame


def labels() -> pd.DataFrame:
    early_pred = pd.DataFrame(read_csv(REPORT / "per_call_predictions.csv"))
    early_labels = pd.DataFrame(read_csv(ROOT / "data" / "annotations" / "priority_20261111_20261112_calls.csv"))
    early_labels = early_labels.drop_duplicates("call_id")
    early = early_labels.merge(early_pred, on="call_id", how="inner")
    early = early[early["primary_label"].isin(HUMAN | SYNTHETIC)].copy()
    early["day"] = "11-12"
    early["synthetic"] = early["primary_label"].isin(SYNTHETIC)
    early["pack"] = np.where(early["campaign"].str.endswith("PACK1"), "PACK1", np.where(early["campaign"].str.endswith("PACK2"), "PACK2", ""))

    late_long = pd.read_csv(ROOT / "reports" / "day_20261116_model_error_overlap" / "per_call_predictions.csv", encoding="utf-8-sig")
    truth = late_long.drop_duplicates("call_id")[["call_id", "ground_truth", "primary_label"]]
    wide = late_long.pivot(index="call_id", columns="model", values="predicted_label").reset_index()
    wide = wide.rename(columns={name: f"{name}_prediction" for name in wide.columns if name != "call_id"})
    manifest = pd.DataFrame(read_csv(ROOT / "data" / "manifests" / "day_20261116.csv"))[["call_id", "campaign_name"]]
    spectra = pd.read_csv(
        ROOT / "evaluation" / "day_20261116" / "20261116T152222" / "call_results" / "spectra0.csv",
        usecols=["call_id", "score", "speech_duration_s", "speech_segments", "n_windows"],
    )
    late = truth.merge(wide, on="call_id").merge(manifest, on="call_id").merge(spectra, on="call_id", how="left")
    late["day"] = "16"
    late["synthetic"] = late["ground_truth"].eq("synthetic")
    late["pack"] = np.where(late["campaign_name"].str.endswith("PACK1"), "PACK1", np.where(late["campaign_name"].str.endswith("PACK2"), "PACK2", ""))
    late["spectra0_score"] = late["score"]
    late["spectra_speech_duration_s"] = late["speech_duration_s"]
    return pd.concat([early, late], ignore_index=True)


def roc_auc(synthetic: np.ndarray, score: np.ndarray) -> float:
    """Больший score — более синтетический. Здесь score = −margin."""

    mask = np.isfinite(score)
    synthetic = synthetic[mask]
    score = score[mask]
    positive = score[synthetic]
    negative = score[~synthetic]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    diff = positive[:, None] - negative[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def rates(frame: pd.DataFrame, column: str) -> str:
    part = frame[np.isfinite(pd.to_numeric(frame[column], errors="coerce"))].copy()
    score = pd.to_numeric(part[column], errors="coerce")
    pred = score < THR
    y = part["synthetic"].astype(bool)
    tp = int((y & pred).sum())
    fn = int((y & ~pred).sum())
    fp = int((~y & pred).sum())
    n_syn = int(y.sum())
    n_hum = int((~y).sum())
    classical = y & part["svm_new_prediction"].eq("human") & part["lightgbm_new_prediction"].eq("human")
    held = int((classical & pred).sum())
    classical_n = int(classical.sum())
    recall = tp / n_syn if n_syn else float("nan")
    fpr = fp / n_hum if n_hum else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    ranking = roc_auc(y.to_numpy(), -score.to_numpy())
    return (
        f"n {len(part)} syn {n_syn} hum {n_hum} auc {ranking:.3f} "
        f"recall {recall:.3f} ({tp}/{n_syn}) fpr {fpr:.3f} ({fp}/{n_hum}) prec {precision:.3f} "
        f"classical_miss_held {held}/{classical_n}"
    )


def speech_bin(values: pd.Series) -> pd.Series:
    number = pd.to_numeric(values, errors="coerce")
    out = pd.Series("средняя", index=values.index)
    out[number <= SPEECH_LOW] = "короткая"
    out[number > SPEECH_HIGH] = "длинная"
    out[number.isna()] = pd.NA
    return out


def pad_bin(values: pd.Series) -> pd.Series:
    number = pd.to_numeric(values, errors="coerce")
    out = pd.Series("средний", index=values.index)
    out[number <= PAD_LOW] = "низкий"
    out[number > PAD_HIGH] = "высокий"
    out[number.isna()] = pd.NA
    return out


def experiment_one(frame: pd.DataFrame) -> None:
    print("\n===== эксперимент 1 =====")
    stored = pd.to_numeric(frame["spectra0_score"], errors="coerce")
    gap = (frame["mean"] - stored).abs()
    print("mean vs stored score, max abs", float(gap.max()), "median", float(gap.median()))
    changed = (frame["trimmed"] - frame["mean"]).abs() > 1e-9
    print("trimmed отличается от mean", int(changed.sum()), "из", len(frame), "медиана окон у них", float(frame.loc[changed, "window_count"].median()) if changed.any() else 0)
    empty = frame["mean_ge1"].isna().sum()
    print("без окон valid≥1 с", int(empty))
    for day in ("11-12", "16"):
        day_frame = frame[frame["day"] == day]
        print(f"\n-- {day} --")
        for rule in RULES:
            print(f"{RULE_NAMES[rule]:16} {rates(day_frame, rule)}")
        for column, order in (
            ("pack", ["PACK1", "PACK2"]),
            ("speech_bin", ["короткая", "длинная"]),
            ("pad_bin", ["низкий", "высокий"]),
        ):
            for name in order:
                part = day_frame[day_frame[column] == name]
                print(f"\n[{day} {column} {name}]")
                for rule in RULES:
                    print(f"  {RULE_NAMES[rule]:16} {rates(part, rule)}")


def experiment_two(frame: pd.DataFrame) -> None:
    print("\n===== эксперимент 2 =====")
    ready = frame[
        frame["spectra0_prediction"].isin(["human", "synthetic"])
        & frame["svm_new_prediction"].isin(["human", "synthetic"])
        & frame["lightgbm_new_prediction"].isin(["human", "synthetic"])
    ].copy()
    ready["spectra_syn"] = ready["spectra0_prediction"].eq("synthetic")
    ready["svm_syn"] = ready["svm_new_prediction"].eq("synthetic")
    ready["lgbm_syn"] = ready["lightgbm_new_prediction"].eq("synthetic")
    ready["vote"] = ready[["spectra_syn", "svm_syn", "lgbm_syn"]].sum(axis=1) >= 2

    def line(part: pd.DataFrame, column: str) -> str:
        y = part["synthetic"].astype(bool)
        pred = part[column].astype(bool)
        tp = int((y & pred).sum())
        fp = int((~y & pred).sum())
        n_syn = int(y.sum())
        n_hum = int((~y).sum())
        precision = tp / (tp + fp) if tp + fp else float("nan")
        return f"recall {tp}/{n_syn} fpr {fp}/{n_hum} prec {precision:.3f}"

    for day in ("11-12", "16"):
        day_frame = ready[ready["day"] == day]
        slices = [("все", day_frame)]
        for pack in ("PACK1", "PACK2"):
            slices.append((pack, day_frame[day_frame["pack"] == pack]))
        for name in ("короткая", "средняя", "длинная"):
            slices.append((name, day_frame[day_frame["speech_bin"] == name]))
        for title, part in slices:
            print(f"\n[{day} {title}] n {len(part)} syn {int(part['synthetic'].sum())}")
            for column, label in (
                ("spectra_syn", "Spectra-0"),
                ("svm_syn", "SVM"),
                ("lgbm_syn", "LightGBM"),
                ("vote", "голос 2/3"),
            ):
                print(f"  {label:12} {line(part, column)}")


def experiment_three(frame: pd.DataFrame, raw: pd.DataFrame) -> None:
    print("\n===== эксперимент 3 =====")
    early = frame[frame["day"] == "11-12"].copy()
    votes = [f"{name}_prediction" for name in ORIGIN]
    missed = early[(early["synthetic"]) & early[votes].eq("human").all(axis=1)]
    false_pack2 = early[(~early["synthetic"]) & early["pack"].eq("PACK2") & early["spectra0_prediction"].eq("synthetic")]
    human_pack2 = early[(~early["synthetic"]) & early["pack"].eq("PACK2") & early["spectra0_prediction"].eq("human")]
    tts_pack2 = early[early["synthetic"] & early["pack"].eq("PACK2")]
    print("пропуски всеми", len(missed), "ложные PACK2", len(false_pack2), "верные живые PACK2", len(human_pack2), "TTS PACK2", len(tts_pack2))
    features = pd.read_csv(REPORT / "new_classical_features.csv", usecols=lambda name: name == "call_id" or name in KEY_FEATURES)
    features = features.set_index("call_id")
    groups = {
        "7 пропусков": missed,
        "17 ложных PACK2": false_pack2,
        "верные живые PACK2": human_pack2,
        "TTS PACK2": tts_pack2,
    }
    columns = [
        "spectra_speech_duration_s",
        "window_count",
        "short_share",
        "spoof_share",
        "margin_med",
        "margin_min",
        "margin_max",
        "segment_median_s",
        "padding_ratio",
    ]
    print("-- медианы окон --")
    for name, part in groups.items():
        bits = []
        for column in columns:
            values = pd.to_numeric(part[column], errors="coerce")
            bits.append(f"{column} {values.median():.3f}")
        print(name, " | ".join(bits))
    print("-- LFCC CQCC --")
    for name, part in groups.items():
        present = features.index.intersection(part["call_id"])
        bits = []
        for column in KEY_FEATURES:
            if column not in features.columns:
                continue
            bits.append(f"{column} {pd.to_numeric(features.loc[present, column], errors='coerce').median():.3f}")
        print(name, " | ".join(bits))
    print("-- по звонкам: 7 и 17 --")
    focus = pd.concat([
        missed.assign(kind="miss"),
        false_pack2.assign(kind="fp"),
    ])
    for row in focus.sort_values(["kind", "call_id"]).itertuples(index=False):
        print(
            row.kind,
            str(row.call_id)[:8],
            "speech",
            row.spectra_speech_duration_s,
            "seg_med",
            f"{row.segment_median_s:.2f}",
            "short",
            f"{row.short_share:.2f}",
            "spoof",
            f"{row.spoof_share:.2f}",
            "med",
            f"{row.margin_med:.2f}",
            "min",
            f"{row.margin_min:.2f}",
            "max",
            f"{row.margin_max:.2f}",
            "pad",
            row.padding_ratio,
        )
    del raw


def main() -> None:
    raw = windows()
    print("окон", len(raw), "звонков", raw["call_id"].nunique())
    aggregated = raw.groupby("call_id", sort=False).apply(aggregate_call, include_groups=False).reset_index()
    frame = labels().merge(aggregated, on="call_id", how="left")
    frame["padding_ratio"] = frame["padding_calc"]
    frame["speech_bin"] = speech_bin(frame["spectra_speech_duration_s"])
    frame["pad_bin"] = pad_bin(frame["padding_ratio"])
    check = frame[frame["day"] == "11-12"]
    speech = pd.to_numeric(check["spectra_speech_duration_s"], errors="coerce")
    print("sum valid vs speech duration, median abs", float((check["valid_sum"] - speech).abs().median()))
    experiment_one(frame)
    experiment_two(frame)
    experiment_three(frame, raw)


if __name__ == "__main__":
    main()
