import json
from pathlib import Path

summary = json.loads(Path("results/summary.json").read_text())

report = (
    "CNR DATA ANALYSIS REPORT\n"
    f"Samples: {summary['count']}\n"
    f"Total: {summary['total']}\n"
    f"Average: {summary['average']}\n"
)

Path("results/report.txt").write_text(report)

print("Step 3 complete: Generated results/report.txt")
