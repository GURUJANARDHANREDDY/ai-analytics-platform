"""AI Analytics Platform — launcher.

Run with: streamlit run app.py
This runs the Tableau-style dashboard from frontend/app.py.

For the Enterprise Platform (Medallion, Governance, Observability):
    streamlit run enterprise_app.py --server.port 8502
"""
import sys
from pathlib import Path

root = str(Path(__file__).resolve().parent)
if root not in sys.path:
    sys.path.insert(0, root)

exec(open(Path(root) / "frontend" / "app.py", encoding="utf-8").read())
