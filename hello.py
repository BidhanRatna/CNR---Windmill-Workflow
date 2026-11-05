from pathlib import Path

Path("results").mkdir(exist_ok=True)
Path("results/hello.txt").write_text("CNR orchestration successful\n")
print("Hello from a repository executed by Windmill!")
