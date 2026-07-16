import os
from pathlib import Path

total = 0
for f in Path(".").rglob("*.py"):
    total += len(f.read_text(encoding="utf-8", errors="ignore").splitlines())
print(f"Total: {total} Zeilen")