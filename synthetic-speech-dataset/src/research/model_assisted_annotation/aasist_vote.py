"""Spectra-AASIST3 отдельно и в голосе 2 из 3 вместо Spectra-0.

Пороги прежние. Нового прогона нет: берутся уже посчитанные score.
"""

from __future__ import annotations

import pandas as pd

from src.research.model_assisted_annotation.common_misses import ROOT, read_csv
from src.research.model_assisted_annotation.rules import prediction

HUMAN = {"human_live", "human_recorded"}
SYNTHETIC = {"synthetic_tts", "synthetic_voice_conversion", "synthetic_playback"}
ROBOTS = {
    "e3e76e91-f1ca-406d-a5e5-865c6755f87f",
    "92da96f9-dde8-4c97-bf81-8ce410457a22",
    "1fd9f5cb-2e2d-403c-afa9-3d2e412eef8f",
    "905d9c60-ec8a-4bf7-b85f-96b5dfd9f651",
    "253ecc40-7a74-4707-80f1-b77ad8648925",
    "cbdf3ca2-a7db-421c-9ade-d393ed485c09",
    "cadcd2dd-c317-4f51-a6a5-5995a0500ece",
    "4c6902f3-f2c2-4ad1-9dc2-37253c0c36e2",
    "a09e80f9-d5fc-4705-a905-d91222090101",
}
PEOPLE = {
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
}


def rates(y: pd.Series, pred: pd.Series) -> str:
    y = y.astype(bool)
    pred = pred.astype(bool)
    tp = int((y & pred).sum())
    fp = int((~y & pred).sum())
    n_syn = int(y.sum())
    n_hum = int((~y).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    return f"recall {tp}/{n_syn} fp {fp}/{n_hum} prec {precision:.3f}"


def vote_of(a: pd.Series, b: pd.Series, c: pd.Series) -> pd.Series:
    votes = pd.concat([a.astype(int), b.astype(int), c.astype(int)], axis=1)
    return votes.sum(axis=1) >= 2


def early() -> pd.DataFrame:
    pred = pd.DataFrame(read_csv(ROOT / "reports" / "model_assisted_annotation_20261111_20261112" / "per_call_predictions.csv"))
    labels = pd.DataFrame(read_csv(ROOT / "data" / "annotations" / "priority_20261111_20261112_calls.csv"))
    labels = labels.drop_duplicates("call_id")[["call_id", "primary_label"]]
    frame = pred.merge(labels, on="call_id", how="inner")
    frame = frame[frame["primary_label"].isin(HUMAN | SYNTHETIC)].copy()
    frame["y"] = frame["primary_label"].isin(SYNTHETIC)
    frame["spectra"] = frame["spectra0_prediction"].eq("synthetic")
    frame["aasist"] = frame["aasist3_prediction"].eq("synthetic")
    frame["svm"] = frame["svm_new_prediction"].eq("synthetic")
    frame["lgbm"] = frame["lightgbm_new_prediction"].eq("synthetic")
    return frame


def late() -> pd.DataFrame:
    long = pd.read_csv(ROOT / "reports" / "day_20261116_model_error_overlap" / "per_call_predictions.csv", encoding="utf-8-sig")
    truth = long.drop_duplicates("call_id")[["call_id", "ground_truth", "primary_label"]]
    wide = long.pivot(index="call_id", columns="model", values="predicted_label").reset_index()
    frame = truth.merge(wide, on="call_id")
    frame = frame[frame["primary_label"].isin(HUMAN | SYNTHETIC)].copy()
    frame["y"] = frame["ground_truth"].eq("synthetic")
    frame["spectra"] = frame["spectra0"].eq("synthetic")
    frame["aasist"] = frame["aasist3"].eq("synthetic")
    frame["svm"] = frame["svm_new"].eq("synthetic")
    frame["lgbm"] = frame["lightgbm_new"].eq("synthetic")
    return frame


def callback() -> pd.DataFrame:
    vote = pd.read_csv(ROOT / "reports" / "callback_20261116" / "vote.csv")
    scores = pd.read_csv(
        ROOT / "evaluation" / "callback_20261116" / "20261117T024359" / "call_scores.csv",
        usecols=["call_id", "aasist3_status", "aasist3_score"],
    )
    frame = vote.merge(scores, on="call_id", how="left")
    verified = {call_id: True for call_id in ROBOTS}
    verified.update({call_id: False for call_id in PEOPLE})
    frame = frame[frame["call_id"].isin(verified)].copy()
    frame["y"] = frame["call_id"].map(verified)
    frame["spectra"] = frame["spectra_pred"].eq("synthetic")
    frame["svm"] = frame["svm_pred"].eq("synthetic")
    frame["lgbm"] = frame["lgbm_pred"].eq("synthetic")
    frame["aasist"] = [
        prediction("aasist3", score, status) == "synthetic"
        for score, status in zip(frame["aasist3_score"], frame["aasist3_status"])
    ]
    return frame


def report(name: str, frame: pd.DataFrame) -> None:
    spectra_vote = vote_of(frame["spectra"], frame["svm"], frame["lgbm"])
    aasist_vote = vote_of(frame["aasist"], frame["svm"], frame["lgbm"])
    print(f"\n===== {name} n {len(frame)} syn {int(frame['y'].sum())} =====")
    print("Spectra-0      ", rates(frame["y"], frame["spectra"]))
    print("AASIST3        ", rates(frame["y"], frame["aasist"]))
    print("голос Spectra  ", rates(frame["y"], spectra_vote))
    print("голос AASIST   ", rates(frame["y"], aasist_vote))
    differ = spectra_vote != aasist_vote
    y = frame["y"].astype(bool)
    print(
        "голоса расходятся",
        int(differ.sum()),
        "из них не человек",
        int((differ & y).sum()),
        "человек",
        int((differ & ~y).sum()),
    )
    only_spectra = differ & spectra_vote & ~aasist_vote
    only_aasist = differ & aasist_vote & ~spectra_vote
    print(
        "synthetic только голос Spectra",
        int(only_spectra.sum()),
        f"не человек {int((only_spectra & y).sum())} человек {int((only_spectra & ~y).sum())}",
    )
    print(
        "synthetic только голос AASIST",
        int(only_aasist.sum()),
        f"не человек {int((only_aasist & y).sum())} человек {int((only_aasist & ~y).sum())}",
    )


def main() -> None:
    report("11-12", early())
    report("16", late())
    checked = callback()
    report("callback 19", checked)
    print("\n-- callback 19, AASIST по звонкам --")
    for row in checked.sort_values("call_id").itertuples(index=False):
        kind = "робот" if row.y else "человек"
        print(
            str(row.call_id)[:8],
            kind,
            "AASIST",
            f"{float(row.aasist3_score):.3f}",
            "synthetic" if row.aasist else "human",
            "Spectra",
            "synthetic" if row.spectra else "human",
            "SVM",
            "synthetic" if row.svm else "human",
            "LGBM",
            "synthetic" if row.lgbm else "human",
        )


if __name__ == "__main__":
    main()
