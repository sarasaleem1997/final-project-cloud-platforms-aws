"""
Inference service for predicting total piece travel time.

Two backends are supported:
- Predictor: loads the trained XGBoost model from disk (local use).
- SageMakerPredictor: calls a live SageMaker real-time endpoint via boto3.

The Streamlit app selects the backend based on the SAGEMAKER_ENDPOINT_NAME
environment variable: if set, it uses SageMakerPredictor; otherwise Predictor.

Usage as CLI:
    uv run python -m vaultech_analysis.inference --die-matrix 5052 --strike2 18.3 --oee 13.5

Usage as module (for Streamlit):
    from vaultech_analysis.inference import Predictor
    predictor = Predictor()
    result = predictor.predict(die_matrix=5052, lifetime_2nd_strike_s=18.3, oee_cycle_time_s=13.5)
"""

import argparse
import json
import time
from pathlib import Path

import boto3
import pandas as pd
from xgboost import XGBRegressor


MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
GOLD_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "gold" / "pieces.parquet"


class Predictor:
    """Loads the trained model and provides predictions."""

    def __init__(self, model_dir: Path = MODEL_DIR, gold_file: Path = GOLD_FILE):
        # Load the XGBoost model
        self.model = XGBRegressor()
        self.model.load_model(model_dir / "xgboost_bath_predictor.json")

        # Load model metadata (features, metrics, die_matrices, oee_median)
        with open(model_dir / "model_metadata.json") as f:
            self.metadata = json.load(f)

        self.features = self.metadata["features"]
        self.metrics = self.metadata["metrics"]
        self.valid_matrices = set(self.metadata["die_matrices"])
        self.oee_median = self.metadata["oee_median_default"]

    def predict(
        self,
        die_matrix: int,
        lifetime_2nd_strike_s: float,
        oee_cycle_time_s: float | None = None,
    ) -> dict:
        """Predict total bath time from early-stage features.

        Returns a dict with predicted_bath_time_s, input values, and model_metrics.
        Returns {"error": "..."} for unknown die_matrix values.
        Missing oee_cycle_time_s should default to the median (~13.8s).
        """
        if die_matrix not in self.valid_matrices:
            return {"error": f"Unknown die_matrix: {die_matrix}. Valid values: {sorted(self.valid_matrices)}"}

        oee_value = oee_cycle_time_s if oee_cycle_time_s is not None else self.oee_median

        X = pd.DataFrame([{
            "die_matrix": die_matrix,
            "lifetime_2nd_strike_s": lifetime_2nd_strike_s,
            "oee_cycle_time_s": oee_value,
        }])[self.features]

        predicted = float(self.model.predict(X)[0])

        return {
            "predicted_bath_time_s": round(predicted, 3),
            "die_matrix": die_matrix,
            "lifetime_2nd_strike_s": lifetime_2nd_strike_s,
            "oee_cycle_time_s": oee_cycle_time_s,
            "model_metrics": self.metrics,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.Series:
        """Predict bath time for a DataFrame of pieces.

        Handle missing oee_cycle_time_s by filling with the median.
        """
        X = df[self.features].copy()
        X["oee_cycle_time_s"] = X["oee_cycle_time_s"].fillna(self.oee_median)
        return pd.Series(self.model.predict(X), index=df.index)


class SageMakerPredictor:
    """Calls a live SageMaker real-time endpoint for bath time prediction.

    The endpoint accepts CSV rows (die_matrix,lifetime_2nd_strike_s,oee_cycle_time_s)
    and returns one floating-point prediction per line.

    predict() returns a debug dict with the payload, raw response, and latency —
    used by the Streamlit inference debug panel.
    predict_batch() sends all rows in one CSV request (chunked to stay under 5 MB).
    """

    # Default OEE median to fill nulls (matches training-time imputation)
    _OEE_MEDIAN = 13.81
    # Feature order must match model training
    _FEATURES = ["die_matrix", "lifetime_2nd_strike_s", "oee_cycle_time_s"]
    # Max rows per SageMaker invocation (keep payload < ~4 MB)
    _CHUNK_SIZE = 5000

    def __init__(self, endpoint_name: str, region: str = "eu-west-1"):
        self.endpoint_name = endpoint_name
        self.region = region
        self._runtime = boto3.client("sagemaker-runtime", region_name=region)

    def predict(
        self,
        die_matrix: int,
        lifetime_2nd_strike_s: float,
        oee_cycle_time_s: float | None = None,
    ) -> dict:
        """Predict bath time via SageMaker endpoint.

        Returns a dict with predicted_bath_time_s, inputs, and a _debug key
        containing the raw payload, response, and round-trip latency.
        """
        oee_value = oee_cycle_time_s if oee_cycle_time_s is not None else self._OEE_MEDIAN
        payload = f"{die_matrix},{lifetime_2nd_strike_s},{oee_value}"

        t0 = time.perf_counter()
        response = self._runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="text/csv",
            Body=payload,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        raw_response = response["Body"].read().decode("utf-8").strip()
        prediction = float(raw_response)

        return {
            "predicted_bath_time_s": round(prediction, 3),
            "die_matrix": die_matrix,
            "lifetime_2nd_strike_s": lifetime_2nd_strike_s,
            "oee_cycle_time_s": oee_cycle_time_s,
            "_debug": {
                "endpoint": self.endpoint_name,
                "payload": payload,
                "raw_response": raw_response,
                "latency_ms": latency_ms,
            },
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.Series:
        """Predict bath time for a DataFrame by calling the SageMaker endpoint.

        Sends rows in chunks of up to _CHUNK_SIZE to stay within SageMaker's
        5 MB synchronous invocation limit.
        """
        X = df[self._FEATURES].copy()
        X["oee_cycle_time_s"] = X["oee_cycle_time_s"].fillna(self._OEE_MEDIAN)

        all_predictions = []
        for start in range(0, len(X), self._CHUNK_SIZE):
            chunk = X.iloc[start : start + self._CHUNK_SIZE]
            csv_payload = chunk.to_csv(index=False, header=False)
            response = self._runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="text/csv",
                Body=csv_payload,
            )
            raw = response["Body"].read().decode("utf-8").strip()
            chunk_preds = [float(v) for v in raw.splitlines()]
            all_predictions.extend(chunk_preds)

        return pd.Series(all_predictions, index=df.index)


def main():
    parser = argparse.ArgumentParser(description="Predict bath time for a forging line piece")
    parser.add_argument("--die-matrix", type=int, required=True, help="Die matrix ID (4974, 5052, 5090, 5091)")
    parser.add_argument("--strike2", type=float, required=True, help="Cumulative time at 2nd strike (seconds)")
    parser.add_argument("--oee", type=float, default=None, help="OEE cycle time (seconds, optional)")
    args = parser.parse_args()

    predictor = Predictor()
    result = predictor.predict(
        die_matrix=args.die_matrix,
        lifetime_2nd_strike_s=args.strike2,
        oee_cycle_time_s=args.oee,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
