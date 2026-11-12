import csv
import json
from pathlib import Path

with open("results/data.csv", newline="") as file:
    rows = list(csv.DictReader(file))

values = [float(row["value"]) for row in rows]

summary = {
    "count": len(values),
    "total": sum(values),
    "average": sum(values) / len(values)
}

Path("results/summary.json").write_text(json.dumps(summary, indent=2))

print(f"Step 2 complete: Average = {summary['average']}")
