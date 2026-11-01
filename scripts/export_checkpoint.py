"""Local-only template-free export; does not authorize weight publication."""
import argparse
import hashlib
import json
from pathlib import Path
from mashi_ml.model import Model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.checkpoint.resolve() == args.output.resolve():
        raise ValueError("Output must differ from original checkpoint")
    source_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    digest = Model.load(args.checkpoint).export_inference(args.output)
    model = Model.load(args.output)
    print(json.dumps({"source_checkpoint_sha256": source_hash, "export_sha256": digest,
                      "format_version": 3, "profile": "pca-inference-v1", "capabilities": model.capabilities,
                      "distribution": "local only; learned-array licensing remains unresolved"}, indent=2))


if __name__ == "__main__":
    main()
