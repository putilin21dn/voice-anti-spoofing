"""Девять проверенных ошибок callback и ошибки голоса 2 из 3 на трёх наборах.

Пороги не двигаются. needs_review — аналитическая категория, не новая метка.
Телефоны и другие идентификаторы абонента в таблицы не пишутся.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.model_assisted_annotation.common_misses import KEY_FEATURES, ROOT, read_csv
from src.research.model_assisted_annotation.rules import prediction
from src.research.model_assisted_annotation.three_experiments import THR, aggregate_call

REPORT = ROOT / "reports" / "callback_20261116"
HUMAN = {"human_live", "human_recorded"}
SYNTHETIC = {"synthetic_tts", "synthetic_voice_conversion", "synthetic_playback"}
ROBOTS = (
    "e3e76e91-f1ca-406d-a5e5-865c6755f87f",
    "92da96f9-dde8-4c97-bf81-8ce410457a22",
    "1fd9f5cb-2e2d-403c-afa9-3d2e412eef8f",
    "905d9c60-ec8a-4bf7-b85f-96b5dfd9f651",
    "253ecc40-7a74-4707-80f1-b77ad8648925",
    "cbdf3ca2-a7db-421c-9ade-d393ed485c09",
    "cadcd2dd-c317-4f51-a6a5-5995a0500ece",
    "4c6902f3-f2c2-4ad1-9dc2-37253c0c36e2",
    "a09e80f9-d5fc-4705-a905-d91222090101",
)
PEOPLE = (
    "41c23f1e-8b1b-4a70-aa4e-dc58241cbdc0",
    "43ac422b-8b1e-4410-9460-b445de2214cf",
    "80cffc4d-51ac-485b-a87b-d6eb6c72f6b7",
    "b8084dd3-e790-45dc-ac5b-19b1759fea24",
    "c775f6a3-8603-4abc-b109-44829aaa8598",
    "ef777b54-3347-4d58-97a9-38db921b1150",
    "2160d2ae-64e0-43ff-a871-3fa1371efcc7",
    "5472647c-889a-4956-9631-d275f5e3f79d",
    "b2d940c6-49f2-4fbb-aca7-72136c91084e",
    "6ce4c4de-d977-4db6-997f-2288b60c05ea",
)
UNANIMOUS = (
    "cbdf3ca2-a7db-421c-9ade-d393ed485c09",
    "cadcd2dd-c317-4f51-a6a5-5995a0500ece",
    "4c6902f3-f2c2-4ad1-9dc2-37253c0c36e2",
    "a09e80f9-d5fc-4705-a905-d91222090101",
)


def windows() -> pd.DataFrame:
    paths = [
        ROOT / "evaluation" / "day_20261111" / "20261116T195124" / "window_results" / "spectra0.csv",
        ROOT / "evaluation" / "day_20261112" / "20261116T195754" / "window_results" / "spectra0.csv",
        ROOT / "evaluation" / "day_20261116" / "20261116T152222" / "window_results" / "spectra0.csv",
        ROOT / "evaluation" / "callback_20261116" / "20261117T024359" / "window_results" / "spectra0.csv",
    ]
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame["margin"] = pd.to_numeric(frame["margin"], errors="coerce")
    frame["valid"] = pd.to_numeric(frame["valid_speech_audio_sec"], errors="coerce")
    return frame.dropna(subset=["margin", "valid"])


def decide_margin(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return prediction("spectra0", value, "ok")


def audio_levels(call_ids: list[str], manifest: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for call_id in call_ids:
        source = ROOT / manifest[call_id]["source_file"]
        channel = int(manifest[call_id].get("client_channel") or 0)
        with wave.open(str(source), "rb") as handle:
            channels = handle.getnchannels()
            rate = handle.getframerate()
            frames = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        channel_audio = frames.reshape(-1, channels)[:, channel].astype(np.float64) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(channel_audio)))) if channel_audio.size else np.nan
        frame = int(0.02 * rate)
        usable = channel_audio[: channel_audio.size // frame * frame].reshape(-1, frame)
        frame_rms = np.sqrt(np.mean(np.square(usable), axis=1))
        db = 20.0 * np.log10(np.maximum(frame_rms, 1e-8))
        active = db[db > -45.0]
        dynamic = float(np.percentile(active, 95) - np.percentile(active, 5)) if active.size >= 8 else np.nan
        rows.append({"call_id": call_id, "rms_dbfs": 20.0 * np.log10(max(rms, 1e-8)), "dynamic_range_db": dynamic})
    return pd.DataFrame(rows)


def early_calls(aggregated: pd.DataFrame) -> pd.DataFrame:
    pred = pd.DataFrame(read_csv(ROOT / "reports" / "model_assisted_annotation_20261111_20261112" / "per_call_predictions.csv"))
    labels = pd.DataFrame(read_csv(ROOT / "data" / "annotations" / "priority_20261111_20261112_calls.csv"))
    labels = labels.drop_duplicates("call_id")[["call_id", "primary_label"]]
    frame = pred.merge(labels, on="call_id", how="inner")
    frame = frame[frame["primary_label"].isin(HUMAN | SYNTHETIC)].copy()
    frame["dataset"] = "11-12"
    frame["campaign"] = frame["campaign"]
    frame["label"] = frame["primary_label"]
    frame["is_synthetic"] = frame["primary_label"].isin(SYNTHETIC)
    frame = frame.merge(aggregated, on="call_id", how="left")
    return frame


def late_calls(aggregated: pd.DataFrame) -> pd.DataFrame:
    long = pd.read_csv(ROOT / "reports" / "day_20261116_model_error_overlap" / "per_call_predictions.csv", encoding="utf-8-sig")
    truth = long.drop_duplicates("call_id")[["call_id", "ground_truth", "primary_label"]]
    wide = long.pivot(index="call_id", columns="model", values="predicted_label").reset_index()
    wide = wide.rename(columns={name: f"{name}_prediction" for name in wide.columns if name != "call_id"})
    manifest = pd.DataFrame(read_csv(ROOT / "data" / "manifests" / "day_20261116.csv"))[["call_id", "campaign_name"]]
    frame = truth.merge(wide, on="call_id").merge(manifest, on="call_id")
    frame = frame[frame["primary_label"].isin(HUMAN | SYNTHETIC)].copy()
    frame["dataset"] = "16"
    frame["campaign"] = frame["campaign_name"]
    frame["label"] = frame["primary_label"]
    frame["is_synthetic"] = frame["ground_truth"].eq("synthetic")
    frame["spectra0_prediction"] = frame["spectra0_prediction"]
    frame["svm_new_prediction"] = frame["svm_new_prediction"]
    frame["lightgbm_new_prediction"] = frame["lightgbm_new_prediction"]
    return frame.merge(aggregated, on="call_id", how="left")


def callback_calls(aggregated: pd.DataFrame) -> pd.DataFrame:
    vote = pd.read_csv(REPORT / "vote.csv")
    verified = {call_id: True for call_id in ROBOTS}
    verified.update({call_id: False for call_id in PEOPLE})
    frame = vote[vote["call_id"].isin(verified)].copy()
    frame["dataset"] = "callback"
    frame["campaign"] = "B2X-16-11-2026-callback"
    frame["is_synthetic"] = frame["call_id"].map(verified)
    frame["label"] = np.where(frame["is_synthetic"], "проверено: не человек", "проверено: человек")
    frame["spectra0_prediction"] = frame["spectra_pred"]
    frame["svm_new_prediction"] = frame["svm_pred"]
    frame["lightgbm_new_prediction"] = frame["lgbm_pred"]
    return frame.merge(aggregated, on="call_id", how="left")


def with_rules(frame: pd.DataFrame) -> pd.DataFrame:
    ready = frame[
        frame["spectra0_prediction"].isin(["human", "synthetic"])
        & frame["svm_new_prediction"].isin(["human", "synthetic"])
        & frame["lightgbm_new_prediction"].isin(["human", "synthetic"])
    ].copy()
    ready["spectra_syn"] = ready["spectra0_prediction"].eq("synthetic")
    ready["svm_syn"] = ready["svm_new_prediction"].eq("synthetic")
    ready["lgbm_syn"] = ready["lightgbm_new_prediction"].eq("synthetic")
    ready["vote_syn"] = ready[["spectra_syn", "svm_syn", "lgbm_syn"]].sum(axis=1) >= 2
    ready["vote"] = np.where(ready["vote_syn"], "synthetic", "human")
    ready["review"] = ready["spectra_syn"] & ~ready["svm_syn"] & ~ready["lgbm_syn"]
    ready["weighted_pred"] = ready["weighted"].map(decide_margin)
    ready["mean_ge1_pred"] = ready["mean_ge1"].map(decide_margin)
    return ready


def disagreement(row: pd.Series) -> str:
    said = []
    if row["spectra_syn"]:
        said.append("Spectra-0")
    if row["svm_syn"]:
        said.append("SVM new")
    if row["lgbm_syn"]:
        said.append("LightGBM new")
    if row["is_synthetic"] and not row["vote_syn"]:
        core = "пропуск: все три модели human" if not said else "пропуск: synthetic только " + " и ".join(said)
    else:
        core = "ложный synthetic: " + " и ".join(said)
    notes = []
    if row["weighted_pred"] and row["weighted_pred"] != row["spectra0_prediction"]:
        notes.append(f"взвешенное среднее Spectra-0 даёт {row['weighted_pred']}")
    if row["mean_ge1_pred"] and row["mean_ge1_pred"] != row["spectra0_prediction"]:
        notes.append(f"среднее окон с речью ≥1 с даёт {row['mean_ge1_pred']}")
    elif not row["mean_ge1_pred"]:
        notes.append("нет окна с речью ≥1 с")
    if row["review"]:
        notes.append("попадает в needs_review")
    return "; ".join([core, *notes])


def error_table(frame: pd.DataFrame) -> pd.DataFrame:
    errors = frame[frame["is_synthetic"] != frame["vote_syn"]].copy()
    errors["reason"] = errors.apply(disagreement, axis=1)
    columns = [
        "dataset",
        "call_id",
        "campaign",
        "label",
        "spectra0_prediction",
        "svm_new_prediction",
        "lightgbm_new_prediction",
        "vote",
        "weighted_pred",
        "mean_ge1_pred",
        "reason",
    ]
    table = errors[columns].rename(
        columns={
            "dataset": "набор",
            "campaign": "кампания",
            "label": "метка",
            "spectra0_prediction": "spectra0",
            "svm_new_prediction": "svm_new",
            "lightgbm_new_prediction": "lightgbm_new",
            "vote": "голос_2_из_3",
            "weighted_pred": "spectra_взвешенное_среднее",
            "mean_ge1_pred": "spectra_среднее_окон_от_1с",
            "reason": "причина_расхождения",
        }
    )
    return table.sort_values(["набор", "call_id"])


def review_line(part: pd.DataFrame) -> str:
    y = part["is_synthetic"].astype(bool)
    review = part["review"].astype(bool)
    auto = ~review
    silent = y & ~part["vote_syn"] & auto
    caught = y & part["vote_syn"] & auto
    false_auto = ~y & part["vote_syn"] & auto
    review_syn = int((y & review).sum())
    review_hum = int((~y & review).sum())
    return (
        f"n {len(part)} syn {int(y.sum())} hum {int((~y).sum())} "
        f"review {int(review.sum())} ({review_syn} не человек / {review_hum} человек) "
        f"авто-пропуск {int(silent.sum())} авто-synthetic {int(caught.sum())} авто-ложный {int(false_auto.sum())}"
    )


def median_line(part: pd.DataFrame, columns: list[str]) -> str:
    bits = []
    for column in columns:
        values = pd.to_numeric(part[column], errors="coerce")
        bits.append(f"{column} {values.median():.3f}" if values.notna().any() else f"{column} —")
    return " | ".join(bits)


def main() -> None:
    raw = windows()
    aggregated = raw.groupby("call_id", sort=False).apply(aggregate_call, include_groups=False).reset_index()
    segments = raw.groupby("call_id")["segment_index"].nunique().rename("spectra_n_segments").reset_index()
    aggregated = aggregated.merge(segments, on="call_id", how="left")
    packs = pd.concat([early_calls(aggregated), late_calls(aggregated)], ignore_index=True)
    callback = callback_calls(aggregated)
    labeled = with_rules(pd.concat([packs, callback], ignore_index=True))

    table = error_table(labeled)
    REPORT.mkdir(parents=True, exist_ok=True)
    table_path = REPORT / "vote_errors_three_sets.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")

    print("ошибки голоса", len(table))
    print(table.groupby("набор").size().to_string())
    print(table["причина_расхождения"].str.split(";").str[0].value_counts().to_string())
    print("\n===== needs_review: Spectra synthetic, SVM и LightGBM human =====")
    for name in ("11-12", "16", "callback"):
        print(name, review_line(labeled[labeled["dataset"] == name]))

    vote = pd.read_csv(REPORT / "vote.csv")
    scored = vote[vote["vote"].isin(["human", "synthetic"])].copy()
    scored["review"] = scored["spectra_pred"].eq("synthetic") & scored["svm_pred"].eq("human") & scored["lgbm_pred"].eq("human")
    print("callback все 39 с голосом, needs_review", int(scored["review"].sum()), "из", len(scored))
    print(scored.loc[scored["review"], ["call_id", "prior_classification", "vote"]].to_string(index=False))

    focus_ids = list(ROBOTS) + list(PEOPLE)
    manifest = {row["call_id"]: row for row in read_csv(ROOT / "data" / "manifests" / "callback_20261116.csv")}
    levels = audio_levels(focus_ids, manifest)
    features = pd.read_csv(REPORT / "new_classical_features.csv", usecols=lambda name: name == "call_id" or name in KEY_FEATURES)
    detail = labeled[labeled["call_id"].isin(focus_ids)].merge(levels, on="call_id", how="left").merge(features, on="call_id", how="left")
    detail["group"] = np.where(detail["call_id"].isin(UNANIMOUS), "4 единогласных", np.where(detail["call_id"].isin(ROBOTS), "5 только Spectra", "10 людей"))

    window_columns = [
        "spectra_speech_s",
        "spectra_n_segments",
        "window_count",
        "short_share",
        "spoof_share",
        "margin_med",
        "margin_min",
        "margin_max",
        "segment_median_s",
        "padding_calc",
        "rms_dbfs",
        "dynamic_range_db",
    ]
    print("\n===== медианы групп callback =====")
    for name in ("4 единогласных", "5 только Spectra", "10 людей"):
        print(name, median_line(detail[detail["group"] == name], window_columns))
    print("-- признаки --")
    for name in ("4 единогласных", "5 только Spectra", "10 людей"):
        print(name, median_line(detail[detail["group"] == name], KEY_FEATURES))

    print("\n===== девять ошибок =====")
    robots = detail[detail["call_id"].isin(ROBOTS)].sort_values("call_id")
    show = [
        "call_id",
        "group",
        "prior_classification",
        "outcome",
        "live_amd",
        "duration_seconds",
        "spectra_speech_s",
        "spectra_n_segments",
        "window_count",
        "segment_median_s",
        "short_share",
        "spoof_share",
        "padding_calc",
        "spectra0_score",
        "svm_new_score",
        "lightgbm_new_score",
        "weighted",
        "weighted_pred",
        "mean_ge1",
        "mean_ge1_pred",
        "margin_min",
        "margin_med",
        "margin_max",
        "rms_dbfs",
        "dynamic_range_db",
        "f0_mean__mean",
        "f0_voiced_fraction__mean",
        "snr_db",
        "lfcc_static_00_mean__mean",
        "cqcc_static_00_mean__mean",
    ]
    robots[show].to_csv(REPORT / "nine_errors.csv", index=False, encoding="utf-8-sig")
    for row in robots.itertuples(index=False):
        print(
            str(row.call_id)[:8],
            row.group,
            row.prior_classification,
            row.outcome,
            "live",
            row.live_amd,
            "dur",
            f"{float(row.duration_seconds):.1f}",
            "speech",
            f"{float(row.spectra_speech_s):.2f}",
            "seg",
            int(row.spectra_n_segments),
            "win",
            int(row.window_count),
            "seg_med",
            f"{float(row.segment_median_s):.2f}",
            "short",
            f"{float(row.short_share):.2f}",
            "spoof",
            f"{float(row.spoof_share):.2f}",
            "pad",
            f"{float(row.padding_calc):.2f}",
            "S",
            f"{float(row.spectra0_score):.2f}",
            "svm",
            f"{float(row.svm_new_score):.2f}",
            "lgb",
            f"{float(row.lightgbm_new_score):.3f}",
            "w",
            row.weighted_pred,
            "ge1",
            row.mean_ge1_pred or "нет",
            "rms",
            f"{float(row.rms_dbfs):.1f}",
            "dyn",
            f"{float(row.dynamic_range_db):.1f}",
            "f0",
            f"{float(row.f0_mean__mean):.0f}",
            "voiced",
            f"{float(row.f0_voiced_fraction__mean):.2f}",
            "snr",
            f"{float(row.snr_db):.0f}",
            "lfcc",
            f"{float(row.lfcc_static_00_mean__mean):.1f}",
            "cqcc",
            f"{float(row.cqcc_static_00_mean__mean):.0f}",
        )

    flips = robots[(robots["weighted_pred"] != robots["spectra0_prediction"]) | (robots["mean_ge1_pred"] != robots["spectra0_prediction"])]
    print("агрегация меняет Spectra на девяти", len(flips))

    print("needs_review записан в stdout выше")


if __name__ == "__main__":
    main()
