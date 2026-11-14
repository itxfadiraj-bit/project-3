import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    "generate_data.py",
    "prepare_comments.py",
    "cluster_comments.py",
    "refine_clusters.py",
    "analyze_campaigns.py",
    "validate_comment_volume.py",
]

for script in SCRIPTS:
    print(f"\nRunning {script}", flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / "src" / script)],
        cwd=ROOT,
        check=True,
    )

print("\nALL PIPELINE STEPS COMPLETED")
print("Launch the dashboard with:")
print("python -m streamlit run dashboard.py --server.port 8504")
