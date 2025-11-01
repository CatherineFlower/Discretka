# -*- coding: utf-8 -*-
import sys, subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
PR6 = BASE / "pr6.py"
DM_MAIN = BASE / "dm_main.py"  # ваш скрипт для контрольной по дискретке

MENU = """Выберите задание:
1 — Контрольная (дискретная математика)  [dm_main.py]
6 — Практическая 6 (простые + творческое) [pr6.py]
q — выход
> """

def run_py(path: Path):
    if not path.exists():
        print(f"Нет файла: {path.name}")
        return 1
    return subprocess.run([sys.executable, str(path)]).returncode

def main():
    while True:
        choice = input(MENU).strip().lower()
        if choice == "1":
            run_py(DM_MAIN)
        elif choice == "6":
            run_py(PR6)
        elif choice in ("q","й"):
            break
        else:
            print("Не понял выбор.")

if __name__ == "__main__":
    main()
