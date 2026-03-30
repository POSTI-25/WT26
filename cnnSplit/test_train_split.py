import argparse
from pathlib import Path

try:
    from cnnSplit.model_parallel_rpc import split_model_and_deploy
    from cnnSplit.train import CNN
except ImportError:
    from model_parallel_rpc import split_model_and_deploy
    from train import CNN


def resolve_gpu_json_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    repo_root = Path(__file__).resolve().parents[1]
    fallback = repo_root / "data" / "gpu" / candidate.name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "GPU JSON not found. Tried: "
        f"{candidate} and {fallback}. "
        "Use --gpu-json data/gpu/<file>.json or pass an absolute path."
    )


def assert_contiguous_complete(mapping: dict[str, list[int]], total_layers: int) -> None:
    all_indices: list[int] = []
    for worker, indices in mapping.items():
        if not indices:
            continue
        expected = list(range(indices[0], indices[-1] + 1))
        if indices != expected:
            raise AssertionError(
                f"Worker {worker} has non-contiguous assignment: {indices}"
            )
        all_indices.extend(indices)

    sorted_all = sorted(all_indices)
    expected_all = list(range(total_layers))
    if sorted_all != expected_all:
        raise AssertionError(
            f"Combined assignment does not cover all layers exactly once. "
            f"got={sorted_all}, expected={expected_all}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test CNN split assignment from train.py with GPU JSON input."
    )
    parser.add_argument(
        "--gpu-json",
        default="data/gpu/test_gpu_store_2workers.json",
        help="Path to GPU JSON with gpu_cards entries.",
    )
    parser.add_argument(
        "--worker-names",
        default="",
        help="Optional comma-separated worker names. Example: worker0,worker1",
    )
    args = parser.parse_args()

    worker_names = [w.strip() for w in args.worker_names.split(",") if w.strip()]
    gpu_json_path = resolve_gpu_json_path(args.gpu_json)

    model = CNN().eval()
    plan = split_model_and_deploy(
        model=model,
        gpu_json=gpu_json_path,
        worker_names=worker_names if worker_names else None,
        deploy_rpc=False,
    )

    layer_index_mapping = plan["layer_index_mapping"]
    total_layers = len(list(model.model.children()))

    assert_contiguous_complete(layer_index_mapping, total_layers)
    print("\nSplit validation passed.")
    print(f"GPU JSON: {gpu_json_path}")
    print(f"Total layers: {total_layers}")
    print(f"Mapping: {layer_index_mapping}")


if __name__ == "__main__":
    main()
