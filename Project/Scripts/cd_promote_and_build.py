import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def get_metric(client: MlflowClient, run_id: str, metric: str):
    run = client.get_run(run_id)
    return run.data.metrics.get(metric)


def pick_latest_version(versions):
    return max(versions, key=lambda v: int(v.version))


def download_model_artifacts(version, dst_root: Path) -> Path:
    dst_root.mkdir(parents=True, exist_ok=True)
    downloaded_path = mlflow.artifacts.download_artifacts(
        artifact_uri=version.source,
        dst_path=str(dst_root),
    )
    return Path(downloaded_path)


def copy_model_to_inference_app(downloaded_model: Path, inference_app_dir: Path) -> Path:
    target_dir = inference_app_dir / "model"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(downloaded_model, target_dir)
    return target_dir


def build_docker_image(inference_app_dir: Path, image_name: str, image_tag: str):
    cmd = [
        "docker",
        "build",
        "-t",
        f"{image_name}:{image_tag}",
        "-f",
        str(inference_app_dir / "Dockerfile"),
        str(inference_app_dir),
    ]
    subprocess.run(cmd, check=True)


def tag_image(source: str, extra_tag: str):
    subprocess.run(["docker", "tag", source, extra_tag], check=True)


def run_container(image_ref: str, container_name: str, port: int):
    subprocess.run(["docker", "rm", "-f", container_name], check=False)
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "-p",
            f"{port}:8000",
            image_ref,
        ],
        check=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="CD pipeline for MLflow model -> FastAPI image")
    parser.add_argument("--model-name", default="churn_logistic_regression")
    parser.add_argument("--candidate-stage", default="Staging")
    parser.add_argument("--prod-stage", default="Production")
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--image-name", default="churn-inference")
    parser.add_argument("--image-tag", default="latest")
    parser.add_argument("--container-name", default="churn-inference")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--promote", action="store_true", help="Promote candidate to Production if better")
    parser.add_argument("--skip-run", action="store_true", help="Skip running the container after build")
    parser.add_argument("--keep-model", action="store_true", help="Do not delete the copied model directory after build")
    return parser.parse_args()


def main():
    args = parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    client = MlflowClient(tracking_uri=args.mlflow_uri)

    candidate_versions = client.get_latest_versions(args.model_name, stages=[args.candidate_stage])
    if not candidate_versions:
        print(f"No versions found in stage {args.candidate_stage} for model {args.model_name}")
        return 0

    candidate = pick_latest_version(candidate_versions)
    prod_versions = client.get_latest_versions(args.model_name, stages=[args.prod_stage])
    prod = pick_latest_version(prod_versions) if prod_versions else None

    candidate_metric = get_metric(client, candidate.run_id, args.metric)
    prod_metric = get_metric(client, prod.run_id, args.metric) if prod else None

    print(f"Candidate v{candidate.version} {args.metric}={candidate_metric}")
    if prod:
        print(f"Production v{prod.version} {args.metric}={prod_metric}")
    else:
        print("No production model found")

    if candidate_metric is None:
        print(f"Candidate missing metric {args.metric}; aborting")
        return 1

    prod_baseline = prod_metric if prod_metric is not None else float("-inf")
    if candidate_metric < prod_baseline:
        print("Candidate is worse than production; skipping build")
        return 0

    project_root = Path(__file__).resolve().parents[1]
    inference_app_dir = project_root / "inference_app"
    temp_dir = Path(tempfile.mkdtemp(prefix="cd_artifacts_", dir=project_root))

    downloaded_model = download_model_artifacts(candidate, temp_dir)
    copy_model_to_inference_app(downloaded_model, inference_app_dir)

    image_ref = f"{args.image_name}:{args.image_tag}"
    version_tag = f"{args.image_name}:v{candidate.version}"

    build_docker_image(inference_app_dir, args.image_name, args.image_tag)
    tag_image(image_ref, version_tag)

    if not args.skip_run:
        run_container(image_ref, args.container_name, args.port)

    if args.promote:
        client.transition_model_version_stage(
            name=args.model_name,
            version=candidate.version,
            stage=args.prod_stage,
        )
        print(f"Promoted {args.model_name} v{candidate.version} to {args.prod_stage}")

    if not args.keep_model:
        model_dir = inference_app_dir / "model"
        if model_dir.exists():
            shutil.rmtree(model_dir)

    shutil.rmtree(temp_dir, ignore_errors=True)
    print("CD pipeline completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
