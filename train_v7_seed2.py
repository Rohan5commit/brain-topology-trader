#!/usr/bin/env python3
"""NCP v7 seed 2 — sets SEED=2 (→ physical GPU 3) and delegates to train_v7_walkforward.py."""
import os, subprocess, sys
os.environ["SEED"] = "2"
script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_v7_walkforward.py")
sys.exit(subprocess.call([sys.executable, script] + sys.argv[1:]))
