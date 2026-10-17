from __future__ import annotations
import argparse
import json
from pathlib import Path

from .config import ExperimentConfig


def main():
    parser = argparse.ArgumentParser(description="SFL-Lab scientific workbench")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Start the dashboard and API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)
    run = commands.add_parser("run", help="Run a reproducible experiment from JSON")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--action", choices=["compare", "full", "baseline", "candidate"], default="compare")
    run.add_argument("--output", type=Path, default=Path("artifacts/cli-result.json"))
    commands.add_parser("defaults", help="Print default experiment configuration")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        uvicorn.run("sfl_lab.api:create_app", factory=True, host=args.host, port=args.port, workers=1)
    elif args.command == "defaults":
        print(ExperimentConfig().model_dump_json(indent=2))
    else:
        from .controller import execute_experiment
        config = ExperimentConfig.model_validate_json(args.config.read_text())
        result = execute_experiment(config, args.action, progress=lambda event: print(json.dumps(event), flush=True))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"config": config.model_dump(), "result": result}, indent=2, allow_nan=False))
        print(f"Evidence saved to {args.output}")


if __name__ == "__main__":
    main()
