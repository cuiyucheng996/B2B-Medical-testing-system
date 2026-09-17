"""启动 B2B FastAPI（共用主项目 uv 环境）。

用法（在项目根目录）：
    uv run python "B2B business/run.py"
    uv run python "B2B business/run.py" --port 8001
    uv run python "B2B business/run.py" --reload
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

B2B_ROOT = Path(__file__).resolve().parent
UVGRAPH_ROOT = B2B_ROOT.parent
B2B_SRC = B2B_ROOT / "src"

for path in (str(UVGRAPH_ROOT), str(B2B_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 B2B 咨询 API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--reload", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    target = "api.server:app"
    url = f"http://{args.host}:{args.port}"
    print(f"Starting B2B -> {target} at {url}")
    extra: dict = {}
    if args.reload:
        extra = {
            "reload_dirs": [str(B2B_SRC)],
            "reload_includes": ["*.py", "*.html", "*.js", "*.css"],
        }
    uvicorn.run(
        target,
        host=args.host,
        port=args.port,
        reload=args.reload,
        app_dir=str(B2B_SRC),
        **extra,
    )


if __name__ == "__main__":
    main()
