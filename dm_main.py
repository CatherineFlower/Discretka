#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, csv, io, base64, subprocess
from pathlib import Path
from datetime import datetime

KEEP_FILES = False # если нужны промежуточные CSV - поставить True
DATA_BITS = "1001"      # информационная комбинация для Хэмминга
MAX_TABLE_ROWS = 30

# --- вверху файла dm_main.py ---
BASE = Path(__file__).resolve().parent
DM_OUT = BASE / "out_dm"
DM_OUT.mkdir(parents=True, exist_ok=True)

FILE_TASK1 = BASE / "task1_text_coding.py"
FILE_TASK2 = BASE / "task2_hamming.py"
FILE_INPUT = BASE / "input_text.txt"   # абсолютный путь передаём как есть

# артефакты теперь в out_dm/
A1 = DM_OUT / "alphabet_1gram.csv"
A2 = DM_OUT / "alphabet_2gram.csv"
C1_SF = DM_OUT / "code_1gram_shannon_fano.csv"
C1_HF = DM_OUT / "code_1gram_huffman.csv"
C2_SF = DM_OUT / "code_2gram_shannon_fano.csv"
C2_HF = DM_OUT / "code_2gram_huffman.csv"
ENC_SF = DM_OUT / "encoded_1gram_shannon_fano.txt"
DEC_SF = DM_OUT / "decoded_1gram_shannon_fano.txt"
ENC_HF = DM_OUT / "encoded_1gram_huffman.txt"
DEC_HF = DM_OUT / "decoded_1gram_huffman.txt"
HAMMING = DM_OUT / "hamming_codeword.txt"

REPORT_MD = DM_OUT / "report_discrete_dm.md"
REPORT_IPYNB = DM_OUT / "report_discrete_dm.ipynb"

def ensure_exists(p: Path, hint: str):
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {p.name}. {hint}")

def read_csv_rows(path: Path):
    rows = []
    if not path.exists(): return rows
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.reader(f, delimiter=';'):
            rows.append(row)
    return rows

def md_table(rows, title=None, max_rows=MAX_TABLE_ROWS):
    if not rows:
        return (f"### {title}\n_Файл не найден._\n" if title else "_Файл не найден._\n")
    head, data = rows[0], rows[1:1+max_rows]
    out = []
    if title: out.append(f"### {title}\n")
    out.append("| " + " | ".join(head) + " |")
    out.append("| " + " | ".join("---" for _ in head) + " |")
    for r in data: out.append("| " + " | ".join(r) + " |")
    out.append("")
    return "\n".join(out)

def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa
        return True
    except Exception:
        return False

def fig_to_data_uri(make_plot_func):
    if not try_import_matplotlib(): return None
    import matplotlib.pyplot as plt
    plt.figure()
    try:
        make_plot_func(plt)
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        plt.close()
        return None

def parse_prob_table(rows):
    if not rows or len(rows) < 2: return []
    head = [h.strip().lower() for h in rows[0]]
    try:
        si, pi = head.index("symbol"), head.index("probability")
    except ValueError:
        return []
    out = []
    for r in rows[1:]:
        if len(r) <= max(si, pi): continue
        sym = r[si]
        try:
            prob = float(r[pi].replace(",", "."))
        except Exception:
            continue
        out.append((sym, prob))
    out.sort(key=lambda x: -x[1])
    return out

def parse_length_hist(rows):
    if not rows or len(rows) < 2: return {}
    head = [h.strip().lower() for h in rows[0]]
    try:
        li = head.index("length")
    except ValueError:
        return {}
    hist = {}
    for r in rows[1:]:
        if len(r) <= li: continue
        try:
            L = int(r[li])
        except Exception:
            continue
        hist[L] = hist.get(L, 0) + 1
    return dict(sorted(hist.items()))

def truncate_text(path: Path, limit=800):
    if not path.exists(): return "_нет файла_"
    s = path.read_text(encoding='utf-8', errors='ignore')
    return s if len(s) <= limit else s[:limit] + f"\n\n… [обрезано, всего {len(s)} символов]"

def run_task1():
    ensure_exists(FILE_TASK1, "Положи task1_text_coding.py рядом с dm_main.py")
    ensure_exists(FILE_INPUT, "Положи input_text.txt рядом с dm_main.py")
    proc = subprocess.run(
        [sys.executable, str(FILE_TASK1), str(FILE_INPUT)],  # передаём абсолютный путь файла ввода
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=str(DM_OUT)
    )
    print("\n=== Задание 1: stdout ===")
    if proc.stdout: print(proc.stdout.rstrip("\n"))
    if proc.stderr:
        print("\n=== Задание 1: stderr ===", file=sys.stderr)
        print(proc.stderr.rstrip("\n"), file=sys.stderr)
    return proc.stdout, proc.stderr

def run_task2(data_bits: str):
    ensure_exists(FILE_TASK2, "Положи task2_hamming.py рядом с dm_main.py")
    proc = subprocess.Popen(
        [sys.executable, str(FILE_TASK2)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=str(DM_OUT)
    )
    out, err = proc.communicate(data_bits + "\n")
    print("\n=== Задание 2: stdout ===")
    if out: print(out.rstrip("\n"))
    if err:
        print("\n=== Задание 2: stderr ===", file=sys.stderr)
        print(err.rstrip("\n"), file=sys.stderr)
    return out, err

def build_unigram_top_plot_data_uri():
    rows = read_csv_rows(A1)
    pairs = parse_prob_table(rows)[:30]
    if not pairs: return None
    def _plot(plt):
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        plt.bar(xs, ys)
        plt.xticks(rotation=90)
        plt.title("Top-30 символов по вероятности (1-граммы)")
        plt.xlabel("Символ")
        plt.ylabel("Вероятность")
    return fig_to_data_uri(_plot)

def build_len_hist_data_uri(code_rows, title):
    hist = parse_length_hist(code_rows)
    if not hist: return None
    def _plot(plt):
        xs, ys = list(hist.keys()), list(hist.values())
        plt.bar(xs, ys)
        plt.title(title)
        plt.xlabel("Длина кода")
        plt.ylabel("Число символов")
    return fig_to_data_uri(_plot)

def build_report(log1_out, log1_err, out2, err2):
    existed_before = REPORT_MD.exists() or REPORT_IPYNB.exists()
    md = []
    md.append(f"# Отчёт по КР (дискретная математика)\n\n_Сгенерировано:_ {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    md.append("## Задание 1 — консольный вывод\n")
    md.append("```text\n" + (log1_out or "").strip() + ("\n" + (log1_err or "").strip() if log1_err else "") + "\n```\n")
    md.append(md_table(read_csv_rows(A1), "Алфавит и вероятности (1-граммы)"))
    uri = build_unigram_top_plot_data_uri()
    if uri: md.append(f"![Top-30 символов]({uri})\n")
    md.append("### Схемы кодирования (1-граммы)\n")
    md.append(md_table(read_csv_rows(C1_SF), "Шеннон–Фано"))
    md.append(md_table(read_csv_rows(C1_HF), "Хаффман"))
    uri_sf = build_len_hist_data_uri(read_csv_rows(C1_SF), "Распределение длин кодов (Шеннон–Фано, 1-граммы)")
    if uri_sf: md.append(f"![Длины Шеннон–Фано]({uri_sf})\n")
    uri_hf = build_len_hist_data_uri(read_csv_rows(C1_HF), "Распределение длин кодов (Хаффман, 1-граммы)")
    if uri_hf: md.append(f"![Длины Хаффмана]({uri_hf})\n")
    md.append(md_table(read_csv_rows(A2), "Алфавит (2-граммы)"))
    md.append(md_table(read_csv_rows(C2_SF), "Шеннон–Фано (2-граммы)"))
    md.append(md_table(read_csv_rows(C2_HF), "Хаффман (2-граммы)"))
    md.append("> Примечание: для 2-грамм показаны таблицы и метрики; декодирование исходного текста не выполняется из-за перекрытия.\n")
    md.append("\n## Задание 2 — консольный вывод\n")
    md.append("```text\n" + (out2 or "").strip() + ("\n" + (err2 or "").strip() if err2 else "") + "\n```\n")
    md.append("**hamming_codeword.txt**\n\n```text\n" + truncate_text(HAMMING) + "\n```\n")
    REPORT_MD.write_text("\n".join(md), encoding="utf-8")
    try:
        import nbformat as nbf
        nb = nbf.v4.new_notebook()
        nb["cells"] = [nbf.v4.new_markdown_cell(REPORT_MD.read_text(encoding="utf-8"))]
        with REPORT_IPYNB.open("w", encoding="utf-8") as f:
            nbf.write(nb, f)
    except Exception:
        pass
    return existed_before

def cleanup():
    if KEEP_FILES: return
    for p in [A1, A2, C1_SF, C1_HF, C2_SF, C2_HF, ENC_SF, DEC_SF, ENC_HF, DEC_HF, HAMMING]:
        try:
            if p.exists(): p.unlink()
        except Exception:
            pass

def main():
    ensure_exists(FILE_TASK1, "Файл обязателен.")
    ensure_exists(FILE_TASK2, "Файл обязателен.")
    ensure_exists(FILE_INPUT, "Файл обязателен.")
    log1_out, log1_err = run_task1()
    out2, err2 = run_task2(DATA_BITS)
    existed_before = build_report(log1_out, log1_err, out2, err2)
    cleanup()
    print("\n=== Итог ===")
    if existed_before:
        print(f"Обновлено: {REPORT_MD.name} и {REPORT_IPYNB.name}")
    else:
        print(f"Создано: {REPORT_MD.name}" + (f", {REPORT_IPYNB.name}" if REPORT_IPYNB.exists() else ""))

if __name__ == "__main__":
    main()
