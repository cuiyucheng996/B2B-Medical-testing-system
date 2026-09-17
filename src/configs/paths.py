"""B2B 项目路径：本地 src + 主项目 uvgraph 根目录。"""

from __future__ import annotations

import sys
from pathlib import Path

B2B_ROOT = Path(__file__).resolve().parents[2]
B2B_SRC = B2B_ROOT / "src"
UVGRAPH_ROOT = B2B_ROOT.parent
B2C_SRC = UVGRAPH_ROOT / "B2C business" / "src"


def ensure_b2b_src() -> Path:
    src = str(B2B_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    return B2B_SRC


def ensure_uvgraph_root() -> Path:
    ensure_b2b_src()
    root = str(UVGRAPH_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return UVGRAPH_ROOT


def ensure_b2c_src() -> Path:
    ensure_uvgraph_root()
    src = str(B2C_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    return B2C_SRC
