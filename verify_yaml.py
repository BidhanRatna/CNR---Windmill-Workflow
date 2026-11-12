from pathlib import Path

Path("results").mkdir(exist_ok=True)

Path("results/yaml-proof.txt").write_text(
    "VALIDATION PASSED: YAML selected verify_yaml.py\n"
)

print("YAML PROOF: Windmill executed verify_yaml.py instead of hello.py")
