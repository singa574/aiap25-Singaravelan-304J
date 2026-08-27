from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier

from src.config import load_config, with_overrides
from src.data import clean_bookings, load_bookings
from src.evaluate import choose_threshold, classification_metrics
from src.features import build_feature_frame
from src.modeling import build_pipeline


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate booking-fallout models.")
    parser.add_argument("--config", default="config/default.yaml", help="YAML configuration path")
    parser.add_argument("--db-path", help="Override data.db_path (also useful for local smoke tests)")
    parser.add_argument(
        "--model", choices=["logistic_regression", "random_forest"], help="Override model.name"
    )
    parser.add_argument("--output-dir", help="Override output.dir")
    return parser.parse_args()


def _year_mask(metadata: pd.DataFrame, years: list[int]) -> pd.Series:
    return metadata["arrival_date"].dt.year.isin([int(year) for year in years])


def _split(
    features: pd.DataFrame,
    target: pd.Series,
    metadata: pd.DataFrame,
    split_config: dict[str, list[int]],
) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
    result: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]] = {}
    seen: set[int] = set()
    for name, key in [
        ("train", "train_years"),
        ("validation", "validation_years"),
        ("test", "test_years"),
    ]:
        years = {int(year) for year in split_config[key]}
        overlap = seen & years
        if overlap:
            raise ValueError(f"Split years overlap for {name}: {sorted(overlap)}")
        seen |= years
        mask = _year_mask(metadata, sorted(years))
        if not mask.any():
            raise ValueError(f"No rows found for {name} years {sorted(years)}")
        result[name] = (
            features.loc[mask].reset_index(drop=True),
            target.loc[mask].reset_index(drop=True),
            metadata.loc[mask].reset_index(drop=True),
        )
    return result


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = with_overrides(
        load_config(args.config),
        db_path=args.db_path,
        model_name=args.model,
        output_dir=args.output_dir,
    )

    LOGGER.info("Loading bookings from %s", config["data"]["db_path"])
    raw = load_bookings(config["data"]["db_path"], config["data"]["table"])
    clean, audit = clean_bookings(raw)
    features, target, metadata = build_feature_frame(clean)
    split = _split(features, target, metadata, config["split"])
    x_train, y_train, _ = split["train"]
    x_validation, y_validation, _ = split["validation"]
    x_test, y_test, test_metadata = split["test"]

    LOGGER.info(
        "Rows after cleaning: train=%d validation=%d test=%d; features=%d",
        len(x_train), len(x_validation), len(x_test), x_train.shape[1],
    )
    pipeline = build_pipeline(
        x_train,
        model_name=config["model"]["name"],
        model_params=config["model"].get("params", {}),
        random_state=int(config["model"]["random_state"]),
        min_category_frequency=int(config["preprocessing"]["min_category_frequency"]),
    )
    pipeline.fit(x_train, y_train)
    validation_probability = pipeline.predict_proba(x_validation)[:, 1]
    threshold = choose_threshold(
        y_validation.to_numpy(),
        validation_probability,
        config["evaluation"]["threshold_metric"],
    )
    test_probability = pipeline.predict_proba(x_test)[:, 1]

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    dummy_probability = dummy.predict_proba(np.zeros((len(y_test), 1)))[:, 1]
    top_k = float(config["evaluation"]["top_k_fraction"])
    metrics = {
        "model": config["model"]["name"],
        "data_audit": audit,
        "split": {
            name: {"rows": len(values[1]), "positive_rate": float(values[1].mean())}
            for name, values in split.items()
        },
        "validation": classification_metrics(
            y_validation.to_numpy(),
            validation_probability,
            threshold=threshold,
            top_k_fraction=top_k,
        ),
        "test": classification_metrics(
            y_test.to_numpy(), test_probability, threshold=threshold, top_k_fraction=top_k
        ),
        "test_dummy_prior": classification_metrics(
            y_test.to_numpy(), dummy_probability, threshold=threshold, top_k_fraction=top_k
        ),
    }

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    predictions = test_metadata.copy()
    predictions["target"] = y_test
    predictions["fallout_probability"] = test_probability
    predictions["prediction"] = (test_probability >= threshold).astype(int)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    LOGGER.info("Saved model and evaluation artefacts to %s", output_dir)
    LOGGER.info("Test ROC-AUC: %.4f", metrics["test"]["roc_auc"])
    LOGGER.info("Test average precision: %.4f", metrics["test"]["average_precision"])
    LOGGER.info("Test recall@top %.0f%%: %.4f", top_k * 100, metrics["test"]["recall_at_k"])


if __name__ == "__main__":
    main()

