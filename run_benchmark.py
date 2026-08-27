import argparse
import json
import pathlib
import subprocess
import sys


def run_benchmark_one_model(model_type: str):
    config_path = (
        pathlib.Path(__file__).parent.resolve()
        / "data"
        / "shared_models"
        / (model_type + ".json")
    )
    with open(config_path, "r") as f:
        config = json.load(f)
    lrs = str([config["learning_rate"]])
    batch_size = str(config["batch_size"])

    try:
        subprocess.run(
            [
                "sbatch",
                f"--export=ALL,MODEL_TYPE={model_type},BATCH_SIZE={batch_size},LRS={lrs},NUM_WORKERS=1",
                "run_benchmark.sh",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(
            f"Error occurred while running the benchmark for model {model_type}: {e.stderr}"
        )


if __name__ == "__main__":
    MODELS = [
        ("single-task-individual", "single-task-individual", ""),
        ("single-task-species", "single-task-species", ""),
        ("single-task-vox-type", "single-task-vox-type", ""),
        ("multi-task-equal", "multi-task-equal", ""),
        ("multi-task-static", "multi-task-static", ""),
        ("multi-task-gradnorm", "multi-task-gradnorm", ""),
        ("ast-frozen-individual", "ast-frozen-individual", ""),
        ("ast-frozen-species", "ast-frozen-species", ""),
        ("ast-frozen-vox-type", "ast-frozen-vox-type", ""),
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=[m[0] for m in MODELS], help="Model to run"
    )
    args = parser.parse_args()
    model_name, model_type, model_params = next(
        (m for m in MODELS if m[0] == args.model), (None, None, None)
    )
    if model_name is None:
        print(f"Model {args.model} not found in MODELS list.")
        sys.exit(1)
    run_benchmark_one_model(model_type)
