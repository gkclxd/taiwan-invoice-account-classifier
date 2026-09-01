# -*- coding: utf-8 -*-
"""
pytest 共用設定：將 src/ 加入 sys.path，讓所有測試可直接
`import invoice_classifier` 或 `from invoice_classifier.xxx import yyy`。
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
