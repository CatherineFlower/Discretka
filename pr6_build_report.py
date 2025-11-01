#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pr6_build_report.py — единый отчёт по Практике №6 + раскладка артефактов.

Что делает:
1) запускает pr6.py (оставляет stdout/stderr в консоли);
2) создаёт подпапку out_pr6/out_files;
3) переносит ВСЕ сгенерированные файлы из out_pr6 в out_pr6/out_files
   (кроме конечных отчётов *.md/*.ipynb);
4) формирует/обновляет единый отчёт:
   - out_pr6/pr6_report_all.md (с встраиванием PNG как base64);
   - out_pr6/pr6_report_all.ipynb (одна markdown-ячейка).
"""

import sys, subprocess, base64, csv, shutil
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent
PR6 = BASE / "pr6.py"

OUT = BASE / "out_pr6"
FILES = OUT / "out_files"           # сюда сложим все артефакты практической
REPORT_MD = OUT / "pr6_report_all.md"
REPORT_NB = OUT / "pr6_report_all.ipynb"

MAX_TABLE_ROWS = 30
IMG_EXT = {".png", ".jpg", ".jpeg"}
TEXT_EXT = {".txt", ".md", ".csv"}

# ---------- утилиты ----------
def run_pr6():
    proc = subprocess.run([sys.executable, str(PR6)],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("\n===== RUN pr6.py: stdout =====")
    if proc.stdout: print(proc.stdout.rstrip("\n"))
    if proc.stderr:
        print("\n===== RUN pr6.py: stderr =====", file=sys.stderr)
        print(proc.stderr.rstrip("\n"), file=sys.stderr)
    return proc.returncode == 0

def ensure_dirs():
    OUT.mkdir(parents=True, exist_ok=True)
    FILES.mkdir(parents=True, exist_ok=True)

def move_artifacts():
    """
    Переносим все созданные файлы из out_pr6/ в out_pr6/out_files,
    кроме итоговых отчётов (*.md, *.ipynb), а также самой подпапки out_files.
    """
    moved = []
    for p in OUT.glob("*"):
        if p.is_dir():
            continue
        if p.name == REPORT_MD.name or p.name == REPORT_NB.name:
            continue
        target = FILES / p.name
        try:
            shutil.move(str(p), str(target))
            moved.append(target)
        except Exception:
            # если уже там — просто учтём
            if target.exists():
                moved.append(target)
    return moved

def to_data_uri(path: Path):
    if not path.exists(): return None
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    ext = path.suffix.lower().lstrip(".")
    if ext == "jpg": ext = "jpeg"
    return f"data:image/{ext};base64,{b64}"

def read_text_trunc(path: Path, limit=2000):
    if not path.exists(): return "_нет файла_"
    s = path.read_text(encoding="utf-8", errors="ignore")
    return s if len(s) <= limit else s[:limit] + f"\n\n… [обрезано, всего {len(s)} символов]"

def md_table_from_csv(path: Path, title=None, delimiter=";"):
    if not path.exists():
        return f"### {title}\n_файл не найден: {path.name}_\n" if title else "_файл не найден_\n"
    rows=[]
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.reader(f, delimiter=delimiter):
            rows.append(r)
    head = rows[0] if rows else []
    data = rows[1:1+MAX_TABLE_ROWS]
    out=[]
    if title: out.append(f"### {title}\n")
    if not head: return "\n".join(out)+ "_пусто_\n"
    out.append("| " + " | ".join(head) + " |")
    out.append("| " + " | ".join("---" for _ in head) + " |")
    for r in data:
        out.append("| " + " | ".join(r) + " |")
    out.append("")
    return "\n".join(out)

def section(h): return f"\n# {h}\n"
def sub(h): return f"\n## {h}\n"
def small(h): return f"\n### {h}\n"

# ---------- построение отчёта ----------
def build_report():
    """
    Собираем единый отчёт из содержимого out_pr6/out_files (всё, что сделала pr6.py).
    Встраиваем картинки как base64, тексты/CSV — как таблицы/параграфы.
    """
    md = []
    md.append(f"# Практика №6 — единый отчёт  \n_Сгенерировано:_ {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    # Блок «Евклид»: таблицы шагов
    md.append(section("Евклид и расширенный Евклид"))
    euclid_csvs = sorted(FILES.glob("euclid_*.csv"))
    if not euclid_csvs:
        md.append("_таблицы шагов не найдены (euclid_*.csv)_\n")
    else:
        md.append("CSV-протоколы шагов (Knuth):\n")
        for p in euclid_csvs:
            title = f"Шаги для пары {p.stem.replace('euclid_', '').replace('_', ', ')}"
            md.append(md_table_from_csv(p, title))

    # Задание 1: простые < 256
    md.append(section("Задание 1. Таблица простых < 256"))
    f = FILES / "pr6_primes_lt256.txt"
    md.append("Файл: `out_pr6/out_files/pr6_primes_lt256.txt`.\n")
    md.append("```text\n" + read_text_trunc(f, limit=3000) + "\n```\n")

    # Задание 2: Ферма
    md.append(section("Задание 2. Метод Ферма для набора чисел"))
    f = FILES / "pr6_fermat_numbers.txt"
    md.append("Файл: `out_pr6/out_files/pr6_fermat_numbers.txt`.\n")
    md.append("```text\n" + read_text_trunc(f) + "\n```\n")

    # Задание 3: большое p
    md.append(section("Задание 3. Проверка большого p (S–S, Леманн, M–R, перебор)"))
    f = FILES / "pr6_bigp_checks.txt"
    md.append("Файл: `out_pr6/out_files/pr6_bigp_checks.txt`.\n")
    md.append("```text\n" + read_text_trunc(f) + "\n```\n")

    # Творческое — картинки + пояснения
    md.append(section("Творческое: визуализации и эмпирика"))

    def add_img(name, title, note=None):
        uri = to_data_uri(FILES / name)
        if uri:
            md.append(small(title))
            if note: md.append(note + "\n")
            md.append(f"![]({uri})\n")
        else:
            md.append(small(title) + f"\n_нет файла {name}_\n")

    # 1) плотность простых
    add_img(
        "pi_vs_xlogx.png",
        "Плотность простых: π(x) и аппроксимация x/ln x",
        "Аппроксимация x/ln x слегка занижает π(x), но уже при x ≤ 2·10^5 идёт близко."
    )

    # 2) разрывы
    add_img(
        "prime_gaps_hist.png",
        "Разрывы между простыми: гистограмма",
        "Чаще всего встречаются малые разрывы; доля больших быстро падает."
    )
    add_img(
        "prime_gaps_running_max.png",
        "Максимальный разрыв ≤ p (running max)",
        "Максимальный наблюдаемый разрыв растёт медленно — крупные разрывы редки."
    )

    # 3) ошибка MR + развёрнутое пояснение
    add_img(
        "mr_false_rate.png",
        "Ошибка Миллера–Рабина (эмпирика)",
        "На нашей выборке из ≈2000 нечётных составных до 2·10^5 доля ложноположительных\n"
        "часто равна 0 уже при t=1…3. Это нормально: 1/4^t — верхняя теоретическая оценка,\n"
        "а на реальной выборке ошибок может не встретиться вовсе. Если нужна «живая» кривая,\n"
        "увеличьте диапазон (до 10^6–10^7), расширьте выборку составных, уберите предварительное\n"
        "деление на малые простые и добавьте «хитрые» составные (Кармайкла, сильные псевдопростые)."
    )

    # 4) Якоби + развёрнутое пояснение
    add_img(
        "jacobi_heatmap.png",
        "Тепловая карта символа Якоби J(a,n)",
        "Ось X — основание a, ось Y — нечётные модули n (3…199). Цвета: +1, 0, −1.\n"
        "Нули (полоса и клин) — там, где gcd(a, n) > 1, по определению J(a,n)=0.\n"
        "Мозаика ±1 визуализирует квадратичную взаимность: при обмене ролями a и n знак\n"
        "меняется, когда оба ≡ 3 (mod 4). Практически это объясняет, почему тест Соловея–Штрассена\n"
        "эффективно отбрасывает множество составных."
    )

    # Кармайкла и псевдопростые Ферма
    add_img(
        "carmichael_hist.png",
        "Числа Кармайкла до заданного лимита",
        "Это составные, проходящие тест Ферма для любого основания a, взаимно простого с n.\n"
        "Встречаются редко, но принципиально важны как контрпримеры к тесту Ферма."
    )
    add_img(
        "fermat_heatmap.png",
        "Псевдопростые Ферма: основание × интервал (тепловая карта)",
        "Количество псевдопростых зависит и от основания, и от диапазона; карта подсвечивает эти зоны."
    )

    # Профили времени
    add_img(
        "tests_timing.png",
        "Профилирование времени тестов",
        "MR обычно даёт лучший баланс «скорость/надёжность»; S–S медленнее из-за символа Якоби;\n"
        "Ферма самый быстрый, но даёт больше ложноположительных."
    )

    # Итоговые выводы (расширённые)
    md.append(section("Короткие выводы по всем блокам"))
    md.append(
        "- Евклид и расширенный Евклид: протокол шагов даёт gcd(a,b) и коэффициенты Безу; НОК = a/gcd·b.\n"
        "- Решето / простые <256: база малых простых и быстрых делимостей.\n"
        "- Ферма: быстро отсеивает составные; метод разности квадратов иногда даёт факторизацию.\n"
        "- Большое p: Миллер–Рабин надёжнее при тех же t; Соловей–Штрассена близок; Ферма — «мягкий».\n"
        "- π(x) против x/ln x: на масштабе до 2·10^5 приближение уже хорошее, но немного занижает.\n"
        "- Разрывы между простыми: малые доминируют; максимум растёт медленно.\n"
        "- Ошибка M–R: на практических диапазонах часто 0 уже при малом t; чтобы увидеть убывание ≈ 4^{-t},\n"
        "  нужна «тяжёлая» выборка (больше чисел, больший диапазон, без препроверок, c Кармайкла).\n"
        "- Символ Якоби: нули при gcd(a,n)>1 и «рисунок» взаимности объясняют силу теста Соловея–Штрассена.\n"
        "- Кармайкла и псевдопростые Ферма: показывают границы применимости теста Ферма.\n"
        "- Профили времени: MR — лучший компромисс между скоростью и надёжностью."
    )

    # запись MD
    REPORT_MD.write_text("\n".join(md), encoding="utf-8")

    # дублируем в .ipynb одной markdown-ячейкой
    try:
        import nbformat as nbf
        nb = nbf.v4.new_notebook()
        nb["cells"] = [nbf.v4.new_markdown_cell(REPORT_MD.read_text(encoding="utf-8"))]
        with REPORT_NB.open("w", encoding="utf-8") as f:
            nbf.write(nb, f)
    except Exception:
        pass

# ---------- main ----------
def main():
    ensure_dirs()
    ok = run_pr6()
    if not ok:
        print("Внимание: pr6.py завершился с ошибкой — продолжаю сбор отчёта из имеющегося.")
    move_artifacts()
    build_report()
    print("\n=== ГОТОВО ===")
    print(f"Отчёт: {REPORT_MD.relative_to(BASE)}")
    if REPORT_NB.exists():
        print(f"Ноутбук: {REPORT_NB.relative_to(BASE)}")
    print(f"Все файлы практической: {FILES.relative_to(BASE)}")

if __name__ == "__main__":
    main()
