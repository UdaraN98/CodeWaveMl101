import argparse
import os
import sqlite3
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from urllib.parse import unquote, urlparse


@dataclass(frozen=True)
class BestRun:
    experiment_id: str
    experiment_name: str
    run_id: str
    metric_name: str
    metric_value: float
    artifact_uri: str


def _project_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_tracking_uri() -> str:
    # Keep behavior compatible with existing env overrides
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        return tracking_uri
    return f"file://{os.path.join(_project_dir(), 'mlruns')}"


def _default_db_path() -> str:
    return os.path.join(_project_dir(), "mlflow.db")


def _sanitize_folder_name(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in name).strip("_")


def _artifact_uri_to_path(artifact_uri: str) -> str:
    if artifact_uri.startswith("file://"):
        parsed = urlparse(artifact_uri)
        return unquote(parsed.path)

    # MLflow server-style artifact URIs (often backed by local ./mlartifacts)
    if artifact_uri.startswith("mlflow-artifacts:"):
        parsed = urlparse(artifact_uri)
        rel = unquote(parsed.path).lstrip("/")
        return os.path.join(_project_dir(), "mlartifacts", rel)

    # Allow relative local paths (common when MLflow is run from project root)
    if artifact_uri.startswith("./"):
        return os.path.join(_project_dir(), artifact_uri[2:])
    if artifact_uri.startswith("mlruns/"):
        return os.path.join(_project_dir(), artifact_uri)

    # Fallback: treat as path-like
    return artifact_uri


def _resolve_logged_model_artifact_locations_for_run(db_path: str, run_id: str) -> list[dict[str, str]]:
    """Return logged model artifact locations for a given run_id via mlflow.db."""
    results: list[dict[str, str]] = []
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        cur.execute(
            "SELECT name, version, current_stage, source, run_id FROM model_versions WHERE run_id = ?",
            (run_id,),
        )
        for mv in cur.fetchall():
            source = str(mv["source"])
            if not source.startswith("models:/"):
                continue

            model_id = source[len("models:/"):]
            cur.execute(
                "SELECT model_id, name, artifact_location FROM logged_models WHERE model_id = ?",
                (model_id,),
            )
            lm = cur.fetchone()
            if lm is None:
                continue

            results.append(
                {
                    "registry_name": str(mv["name"]),
                    "version": str(mv["version"]),
                    "stage": str(mv["current_stage"]),
                    "logged_model_name": str(lm["name"]),
                    "model_id": str(lm["model_id"]),
                    "artifact_location": str(lm["artifact_location"]),
                }
            )

    return results


def _find_best_run_via_sqlite(
    db_path: str,
    experiment_name: str,
    metric: str,
    maximize: bool,
    min_metric: Optional[float] = None,
) -> BestRun:
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)

    direction = "DESC" if maximize else "ASC"
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        sql = f"""
        SELECT
            e.experiment_id AS experiment_id,
            e.name          AS experiment_name,
            r.run_uuid      AS run_id,
            r.artifact_uri  AS artifact_uri,
            lm.value        AS metric_value
        FROM experiments e
        JOIN runs r
            ON r.experiment_id = e.experiment_id
        JOIN latest_metrics lm
            ON lm.run_uuid = r.run_uuid
           AND lm.key = ?
        WHERE e.lifecycle_stage = 'active'
          AND r.lifecycle_stage = 'active'
          AND e.name = ?
          AND r.status = 'FINISHED'
        """

        params = [metric, experiment_name]
        if min_metric is not None:
            sql += " AND lm.value >= ?"
            params.append(float(min_metric))

        sql += f" ORDER BY lm.value {direction} LIMIT 1"

        cur.execute(sql, params)
        row = cur.fetchone()
        if row is None:
            raise SystemExit(
                f"No FINISHED runs found in mlflow.db for experiment {experiment_name!r} with metric {metric!r}."
            )

        return BestRun(
            experiment_id=str(row["experiment_id"]),
            experiment_name=str(row["experiment_name"]),
            run_id=str(row["run_id"]),
            metric_name=metric,
            metric_value=float(row["metric_value"]),
            artifact_uri=str(row["artifact_uri"]),
        )


def find_best_run(
    experiment_name: str,
    metric: str,
    maximize: bool,
    min_metric: Optional[float] = None,
) -> BestRun:
    # Prefer SQLite lookup when mlflow.db exists (works without installing mlflow).
    db_path = _default_db_path()
    if os.path.exists(db_path):
        return _find_best_run_via_sqlite(
            db_path=db_path,
            experiment_name=experiment_name,
            metric=metric,
            maximize=maximize,
            min_metric=min_metric,
        )

    # Fallback to MLflow Python client if available.
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as e:
        raise SystemExit(
            "Neither mlflow.db was found nor the 'mlflow' package is importable. "
            "Install mlflow or ensure the tracking database exists. "
            f"Details: {e}"
        )

    mlflow.set_tracking_uri(_default_tracking_uri())
    client = MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise SystemExit(f"Experiment not found: {experiment_name!r}")

    direction = "DESC" if maximize else "ASC"
    order_by = [f"metrics.{metric} {direction}"]

    filter_string = None
    if min_metric is not None:
        filter_string = f"metrics.{metric} >= {min_metric}"

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_string,
        order_by=order_by,
        max_results=1,
    )

    if not runs:
        raise SystemExit(
            f"No runs found for experiment {experiment_name!r} with metric {metric!r}."
        )

    run = runs[0]
    metric_value = run.data.metrics.get(metric)
    if metric_value is None:
        raise SystemExit(
            f"Best run {run.info.run_id} does not have metric {metric!r} logged."
        )

    return BestRun(
        experiment_id=exp.experiment_id,
        experiment_name=exp.name,
        run_id=run.info.run_id,
        metric_name=metric,
        metric_value=float(metric_value),
        artifact_uri=run.info.artifact_uri,
    )


def export_best_artifacts(
    best: BestRun,
    destination_root: str,
    artifact_subpath: Optional[str] = None,
    overwrite: bool = True,
) -> str:
    source_artifacts_dir = _artifact_uri_to_path(best.artifact_uri)
    if artifact_subpath:
        source_artifacts_dir = os.path.join(source_artifacts_dir, artifact_subpath)

    exported_from = f"run_artifacts: {best.artifact_uri}"

    # If run-level artifact URI doesn't exist locally (common with sqlite backend + mlflow-artifacts),
    # fall back to the *model artifacts* stored under mlartifacts via the registry.
    if not os.path.exists(source_artifacts_dir):
        db_path = _default_db_path()
        if os.path.exists(db_path):
            candidates = _resolve_logged_model_artifact_locations_for_run(db_path, best.run_id)
            if candidates:
                chosen = candidates[0]
                source_artifacts_dir = _artifact_uri_to_path(chosen["artifact_location"])
                if artifact_subpath:
                    source_artifacts_dir = os.path.join(source_artifacts_dir, artifact_subpath)
                exported_from = (
                    "registered_model: "
                    f"{chosen['registry_name']} v{chosen['version']} ({chosen['stage']}) "
                    f"artifact_location={chosen['artifact_location']}"
                )

    if not os.path.exists(source_artifacts_dir):
        raise SystemExit(
            "Could not resolve a local artifacts directory to copy. "
            f"Tried: {source_artifacts_dir}"
        )

    exp_folder = _sanitize_folder_name(best.experiment_name)
    run_folder = f"{best.metric_name}_{best.metric_value:.6f}_{best.run_id}"
    dest_dir = os.path.join(destination_root, exp_folder, run_folder)

    os.makedirs(os.path.dirname(dest_dir), exist_ok=True)

    if os.path.exists(dest_dir):
        if overwrite:
            shutil.rmtree(dest_dir)
        else:
            raise SystemExit(f"Destination exists (use --overwrite): {dest_dir}")

    shutil.copytree(source_artifacts_dir, dest_dir)

    metadata_path = os.path.join(dest_dir, "best_run_metadata.txt")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(f"exported_at: {datetime.now().isoformat()}\n")
        f.write(f"experiment_name: {best.experiment_name}\n")
        f.write(f"experiment_id: {best.experiment_id}\n")
        f.write(f"run_id: {best.run_id}\n")
        f.write(f"metric_name: {best.metric_name}\n")
        f.write(f"metric_value: {best.metric_value}\n")
        f.write(f"artifact_uri: {best.artifact_uri}\n")
        f.write(f"export_source: {exported_from}\n")
        if artifact_subpath:
            f.write(f"artifact_subpath: {artifact_subpath}\n")

    return dest_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the best-performing MLflow run artifacts into a dedicated folder under Project/mlruns/. "
            "Defaults to selecting the run with the highest 'f1' metric."
        )
    )
    parser.add_argument(
        "--experiment-name",
        default="customer_churn_optimization",
        help="MLflow experiment name to search",
    )
    parser.add_argument(
        "--metric",
        default="f1",
        help="Metric key to rank runs by (must be logged as an MLflow metric)",
    )
    parser.add_argument(
        "--maximize",
        action="store_true",
        default=True,
        help="Maximize the metric (default)",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Minimize the metric (useful for loss/RMSE)",
    )
    parser.add_argument(
        "--min-metric",
        type=float,
        default=None,
        help="Optional minimum threshold for the metric",
    )
    parser.add_argument(
        "--artifact-subpath",
        default=None,
        help=(
            "Optional subfolder within the run artifacts to copy. "
            "Example: 'lr_optuna' or 'dt_optuna' to copy only the logged model folder."
        ),
    )
    parser.add_argument(
        "--dest",
        default=os.path.join(_project_dir(), "mlruns", "best_model_artifacts"),
        help="Destination root folder",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite destination if it exists (default)",
    )
    args = parser.parse_args()

    maximize = True
    if args.minimize:
        maximize = False

    best = find_best_run(
        experiment_name=args.experiment_name,
        metric=args.metric,
        maximize=maximize,
        min_metric=args.min_metric,
    )

    dest = export_best_artifacts(
        best=best,
        destination_root=args.dest,
        artifact_subpath=args.artifact_subpath,
        overwrite=args.overwrite,
    )

    print("✓ Exported best run artifacts")
    print(f"  experiment: {best.experiment_name} (id={best.experiment_id})")
    print(f"  run_id:     {best.run_id}")
    print(f"  metric:     {best.metric_name}={best.metric_value:.6f}")
    print(f"  from:       {best.artifact_uri}")
    print(f"  to:         {dest}")


if __name__ == "__main__":
    main()
