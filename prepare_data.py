import csv
from pathlib import Path

Path("results").mkdir(exist_ok=True)

with open("results/data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["sample", "value"])
    writer.writerows([
        ["A", 10],
        ["B", 20],
        ["C", 30],
        ["D", 40],
        ["E", 50]
    ])

print("Step 1 complete: Created results/data.csv")
