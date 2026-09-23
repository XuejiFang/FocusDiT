"""Keep only FLAN-T5 encoder weights in a local sharded checkpoint index.

Run this after downloading config/tokenizer files, the original safetensors
index, and every shard that contains encoder or shared weights. The original
index is preserved as model.safetensors.full.index.json.
"""

import argparse
import json
from pathlib import Path


def prepare(directory: Path) -> tuple[int, int]:
    index_path = directory / "model.safetensors.index.json"
    backup_path = directory / "model.safetensors.full.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")

    source = backup_path if backup_path.exists() else index_path
    original = json.loads(source.read_text())
    weight_map = original["weight_map"]
    selected = {
        key: shard
        for key, shard in weight_map.items()
        if key == "shared.weight" or key.startswith("encoder.")
    }
    if "shared.weight" not in selected or not any(key.startswith("encoder.") for key in selected):
        raise ValueError("Index does not contain both shared and encoder weights")

    shards = sorted(set(selected.values()))
    missing = [shard for shard in shards if not (directory / shard).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing encoder weight shards: {', '.join(missing)}")

    if not backup_path.exists():
        backup_path.write_text(json.dumps(original, indent=2) + "\n")

    reduced = dict(original)
    reduced["weight_map"] = selected
    reduced["metadata"] = {
        **original.get("metadata", {}),
        "total_size": sum((directory / shard).stat().st_size for shard in shards),
    }
    index_path.write_text(json.dumps(reduced, indent=2) + "\n")
    return len(selected), len(shards)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Local FLAN-T5-XXL model directory")
    args = parser.parse_args()
    weights, shards = prepare(args.directory)
    print(f"Prepared encoder-only index: {weights} tensors in {shards} shards")


if __name__ == "__main__":
    main()
