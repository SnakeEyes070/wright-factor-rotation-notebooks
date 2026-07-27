from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(r"E:\wright_quant\data")
BASE_DIR = Path(r"C:\Users\sankalp.DESKTOP-1RPOQ63\Documents\Codex\2026-06-12\files-mentioned-by-the-user-pasted")
OUT_DIR = BASE_DIR / "outputs"
WORK_DIR = BASE_DIR / "work"
sys.path.insert(0, str(WORK_DIR))

import write_one_left_direct as scorecard  # noqa: E402
import paper_regime_similarity_strategy as regime  # noqa: E402


RANK_COLS = ["rank_momentum", "rank_quality", "rank_value"]
FWD_COLS = ["fwd_momentum", "fwd_quality", "fwd_value"]


def rank_score(ranks: np.ndarray) -> np.ndarray:
    return 4.0 - np.asarray(ranks, dtype=float)


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    out = np.empty(3, dtype=int)
    out[order] = np.arange(1, 4)
    return out


def spearman3(pred: np.ndarray, true: np.ndarray) -> float:
    return float(1.0 - np.sum((np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)) ** 2) / 4.0)


def scorecard_feature_frame(train: pd.DataFrame, test: pd.DataFrame | None = None):
    feature_cols = [c for c in train.columns if c not in ["date", *RANK_COLS, *FWD_COLS]]
    if test is None:
        raw = train[["date", *feature_cols]]
    else:
        raw = pd.concat([train[["date", *feature_cols]], test[["date", *feature_cols]]], ignore_index=True)
    x_all, added = scorecard.add_eda_features(raw)
    cols = [
        "us_ind_prod",
        "quant_capitulation_indicator",
        "quant_selling_intensity_indicator",
        "reverse_repo",
        "dii_net",
        "fii_net",
        "market_momentum",
        "market_volatility",
        "dxy",
        "us_credit_spread",
        "crude",
        "gold",
        "quant_risk_appetite_index",
        "quant_risk_aversion_index",
        *added,
    ]
    cols = [c for c in cols if c in x_all.columns]
    return x_all, cols


def paper_spec() -> regime.RegimeSpec:
    # Best robust pure regime-similarity spec from the previous train-only search.
    return regime.RegimeSpec(
        n_pca=3,
        k_neighbors=12,
        half_life=60.0,
        kernel_scale=1.0,
        target="rank_score",
        blend_base=0.0,
        blend_transition=0.0,
        blend_econ=0.25,
    )


def walk_forward_predictions(train: pd.DataFrame, min_train: int = 48) -> pd.DataFrame:
    x_score, score_cols = scorecard_feature_frame(train)
    feature_cols = [c for c in train.columns if c not in ["date", *RANK_COLS, *FWD_COLS]]
    x_regime = regime.build_macro_state(train[["date", *feature_cols]])
    spec = paper_spec()

    rows = []
    for idx in range(min_train, len(train)):
        sc = scorecard.predict_rank(
            train.iloc[:idx],
            x_score.iloc[:idx],
            x_score.iloc[idx],
            score_cols,
        )
        rg = regime.predict_with_spec(
            x_regime.iloc[:idx],
            train.iloc[:idx],
            x_regime.iloc[idx],
            spec,
        )
        prev = train.loc[idx - 1, RANK_COLS].to_numpy(dtype=int)
        const_m = np.array([1, 2, 3])
        true = train.loc[idx, RANK_COLS].to_numpy(dtype=int)
        rows.append(
            {
                "date": train.loc[idx, "date"],
                "true": true,
                "scorecard": sc,
                "regime": rg,
                "prev": prev,
                "const_m": const_m,
            }
        )
    return pd.DataFrame(rows)


def evaluate_candidate(preds: list[np.ndarray], trues: list[np.ndarray]) -> dict:
    s = np.array([spearman3(p, t) for p, t in zip(preds, trues)])
    years = pd.Series(s, index=pd.to_datetime([r for r in []]))
    return {
        "mean": float(s.mean()),
        "median": float(np.median(s)),
        "hit": float(np.mean(s > 0)),
    }


def search_hybrids(wf: pd.DataFrame) -> pd.DataFrame:
    trues = list(wf["true"])
    candidates = []

    base_models = ["scorecard", "regime", "prev", "const_m"]
    for model in base_models:
        preds = list(wf[model])
        scores = np.array([spearman3(p, t) for p, t in zip(preds, trues)])
        by_year = pd.Series(scores, index=pd.to_datetime(wf["date"])).groupby(lambda d: d.year).mean()
        candidates.append(
            {
                "name": model,
                "kind": "single",
                "weights": {model: 1.0},
                "mean": float(scores.mean()),
                "median": float(np.median(scores)),
                "hit": float(np.mean(scores > 0)),
                "y2020": float(by_year.get(2020, np.nan)),
                "y2021": float(by_year.get(2021, np.nan)),
                "y2022": float(by_year.get(2022, np.nan)),
            }
        )

    # Forecast-combination literature: combine weak forecasts, shrink away
    # from fragile extremes. We test a small, predeclared simplex grid.
    weight_grid = []
    vals = [0.0, 0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85, 1.0]
    for w_sc in vals:
        for w_rg in vals:
            for w_prev in [0.0, 0.10, 0.20]:
                for w_const in [0.0, 0.10, 0.20]:
                    total = w_sc + w_rg + w_prev + w_const
                    if total <= 0:
                        continue
                    weights = {
                        "scorecard": w_sc / total,
                        "regime": w_rg / total,
                        "prev": w_prev / total,
                        "const_m": w_const / total,
                    }
                    weight_grid.append(weights)

    seen = set()
    for weights in weight_grid:
        key = tuple(round(weights[m], 4) for m in base_models)
        if key in seen:
            continue
        seen.add(key)
        preds = []
        for _, row in wf.iterrows():
            score = np.zeros(3)
            for m in base_models:
                score += weights[m] * rank_score(row[m])
            preds.append(ranks_from_scores(score))
        scores = np.array([spearman3(p, t) for p, t in zip(preds, trues)])
        by_year = pd.Series(scores, index=pd.to_datetime(wf["date"])).groupby(lambda d: d.year).mean()
        candidates.append(
            {
                "name": "blend_" + "_".join(f"{m}{weights[m]:.2f}" for m in base_models if weights[m] > 0),
                "kind": "blend",
                "weights": weights,
                "mean": float(scores.mean()),
                "median": float(np.median(scores)),
                "hit": float(np.mean(scores > 0)),
                "y2020": float(by_year.get(2020, np.nan)),
                "y2021": float(by_year.get(2021, np.nan)),
                "y2022": float(by_year.get(2022, np.nan)),
            }
        )

    return pd.DataFrame(candidates).sort_values(["mean", "y2022", "hit"], ascending=False).reset_index(drop=True)


def generate_submission(train: pd.DataFrame, test: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    x_score, score_cols = scorecard_feature_frame(train, test)
    x_score_train = x_score.iloc[: len(train)].reset_index(drop=True)
    x_score_test = x_score.iloc[len(train) :].reset_index(drop=True)

    feature_cols = [c for c in test.columns if c != "date"]
    raw_all = pd.concat([train[["date", *feature_cols]], test[["date", *feature_cols]]], ignore_index=True)
    x_regime_all = regime.build_macro_state(raw_all)
    x_regime_train = x_regime_all.iloc[: len(train)].reset_index(drop=True)
    x_regime_test = x_regime_all.iloc[len(train) :].reset_index(drop=True)
    spec = paper_spec()

    history = train.copy()
    rows = []
    for i in range(len(test)):
        x_sc_hist = pd.concat([x_score_train, x_score_test.iloc[:i]], ignore_index=True)
        x_rg_hist = pd.concat([x_regime_train, x_regime_test.iloc[:i]], ignore_index=True)
        sc = scorecard.predict_rank(history, x_sc_hist, x_score_test.iloc[i], score_cols)
        rg = regime.predict_with_spec(x_rg_hist, history, x_regime_test.iloc[i], spec)
        prev = history.iloc[-1][RANK_COLS].to_numpy(dtype=int)
        const_m = np.array([1, 2, 3])
        model_ranks = {"scorecard": sc, "regime": rg, "prev": prev, "const_m": const_m}
        score = np.zeros(3)
        for m, w in weights.items():
            score += float(w) * rank_score(model_ranks[m])
        pred = ranks_from_scores(score)
        rows.append(pred)

        pseudo = {c: np.nan for c in train.columns}
        pseudo["date"] = test.loc[i, "date"]
        for j, col in enumerate(RANK_COLS):
            pseudo[col] = int(pred[j])
        history = pd.concat([history, pd.DataFrame([pseudo])], ignore_index=True)

    out = pd.DataFrame({"date": test["date"].dt.strftime("%Y-%m-%d")})
    for j, col in enumerate(RANK_COLS):
        out[col] = [int(r[j]) for r in rows]
    return out


def validate(out: pd.DataFrame, test: pd.DataFrame) -> None:
    assert list(out.columns) == ["date", *RANK_COLS]
    assert out["date"].tolist() == test["date"].dt.strftime("%Y-%m-%d").tolist()
    for row in out[RANK_COLS].to_numpy(dtype=int):
        assert sorted(map(int, row)) == [1, 2, 3]


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    wf = walk_forward_predictions(train)
    search = search_hybrids(wf)
    search.to_csv(WORK_DIR / "paper_hybrid_search.csv", index=False)

    # Selection rule: beat the scorecard if possible while requiring non-negative
    # 2022 and positive hit rate. This is a model-selection rule on train CV only.
    scorecard_mean = float(search[search["name"] == "scorecard"].iloc[0]["mean"])
    viable = search[(search["mean"] > scorecard_mean) & (search["y2022"] >= 0) & (search["hit"] >= 0.70)]
    selected = (viable if len(viable) else search).iloc[0]
    weights = selected["weights"]
    if isinstance(weights, str):
        weights = json.loads(weights.replace("'", '"'))

    out = generate_submission(train, test, weights)
    validate(out, test)
    out_path = OUT_DIR / "submission_paper_hybrid_dma.csv"
    out.to_csv(out_path, index=False)

    summary = {
        "output": str(out_path),
        "selected": selected.to_dict(),
        "scorecard_mean": scorecard_mean,
        "top_20": search.head(20).to_dict(orient="records"),
        "first_15_predictions": out.head(15).to_dict(orient="records"),
        "rank_pattern_counts": out[RANK_COLS].value_counts().reset_index(name="count").to_dict(orient="records"),
    }
    with open(WORK_DIR / "paper_hybrid_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps({k: summary[k] for k in ["output", "selected", "scorecard_mean", "first_15_predictions"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
