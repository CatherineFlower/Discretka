#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Практические работы 7 и 8 (ФР, 2 семестр) — в одном файле.

П7:
  - Простой метод: аффинный шифр (параметры a, b)
  - Сложный метод: шифр Гронсфельда (цифровой ключ)

П8:
  - Построение частотного профиля по большому корпусу (динамический алфавит)
  - Шифрование/расшифровка Цезаря
  - Взлом Цезаря по частотам (χ²)
  - Генерация отчёта по двум практикам

Все выходные файлы пишутся в каталог pr7_8/.
"""

import argparse
import csv
import json
import math
import os
from collections import Counter
from typing import Dict, List, Tuple
# -----------------------------
# MD -> Jupyter Notebook (.ipynb)
# -----------------------------
import time

def md_to_ipynb(markdown_text: str) -> dict:
    """
    Простейший конвертер: разбивает markdown на текстовые и кодовые блоки.
    Блоки формата ```...``` становятся code-cell, остальное — markdown-cell.
    Возвращает готовый JSON словарь в формате nbformat v4.
    """
    lines = markdown_text.splitlines()
    cells = []
    in_code = False
    code_lang = ""
    buf = []

    def flush_md():
        nonlocal buf
        if buf:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": "\n".join(buf)
            })
            buf = []

    def flush_code():
        nonlocal buf
        if buf:
            cells.append({
                "cell_type": "code",
                "metadata": {"language": code_lang or "python"},
                "source": "\n".join(buf),
                "outputs": [],
                "execution_count": None
            })
            buf = []

    for line in lines:
        if line.startswith("```") and not in_code:
            # открытие кода
            in_code = True
            code_lang = line.strip().lstrip("`").strip()
            code_lang = code_lang.replace("python", "python")  # на будущее
            flush_md()
            buf = []
            continue
        if line.startswith("```") and in_code:
            # закрытие кода
            in_code = False
            flush_code()
            buf = []
            code_lang = ""
            continue
        buf.append(line)

    # хвост
    if in_code:
        flush_code()
    else:
        flush_md()

    # Минимальный nbformat v4
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.x"
            },
            "generated": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    return nb

def save_ipynb(nb_dict: dict, out_path: str):
    out_path = to_outpath(out_path)
    ensure_outdir()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, ensure_ascii=False, indent=2)


# -----------------------------
# Конфигурация каталога вывода
# -----------------------------

OUT_DIR = "pr7_8"

def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def to_outpath(path: str) -> str:
    if not path:
        return path
    has_dir = (os.path.dirname(path) not in ("", "."))
    if has_dir or os.path.isabs(path):
        return path
    ensure_outdir()
    return os.path.join(OUT_DIR, path)

# -----------------------------
# Динамический алфавит
# -----------------------------

ALPH: str = ""
M: int = 0
INDEX: Dict[str, int] = {}

def set_alphabet(alph: str):
    """Задать алфавит и индексы."""
    global ALPH, M, INDEX
    ALPH = alph
    M = len(ALPH)
    INDEX = {ch: i for i, ch in enumerate(ALPH)}

def derive_alphabet_from_texts(texts: List[str], lower: bool = True) -> str:
    """Построить алфавит в порядке первого появления символов в данных текстах."""
    seen = set()
    order = []
    for t in texts:
        if lower:
            t = t.lower()
        for ch in t:
            if ch not in seen:
                seen.add(ch)
                order.append(ch)
    if not order:
        order = list(" ")
    return "".join(order)

def save_alphabet(path: str, alphabet: str):
    path = to_outpath(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"alphabet": alphabet}, f, ensure_ascii=False, indent=2)

def load_alphabet(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["alphabet"]

# -----------------------------
# I/O утилиты
# -----------------------------

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_text(path: str, text: str):
    ensure_outdir()
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# -----------------------------
# Математика для аффинного шифра
# -----------------------------

def egcd(a: int, b: int) -> Tuple[int, int, int]:
    if b == 0:
        return a, 1, 0
    g, x, y = egcd(b, a % b)
    return g, y, x - (a // b) * y

def modinv(a: int, m: int) -> int:
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError(f"Не существует обратного к a={a} по модулю M={m} (gcd={g})")
    return x % m

# -----------------------------
# П7 — Аффинный (простой) и Гронсфельд (сложный)
# -----------------------------

def p7_affine_encrypt(text: str, a: int, b: int) -> str:
    if M == 0:
        raise ValueError("Алфавит пуст.")
    if math.gcd(a, M) != 1:
        raise ValueError(f"a={a} не взаимно просто с |ALPH|={M}")
    out = []
    for ch in text:
        if ch in INDEX:
            x = INDEX[ch]
            y = (a * x + b) % M
            out.append(ALPH[y])
        else:
            out.append(ch)
    return "".join(out)

def p7_affine_decrypt(text: str, a: int, b: int) -> str:
    if M == 0:
        raise ValueError("Алфавит пуст.")
    if math.gcd(a, M) != 1:
        raise ValueError(f"a={a} не взаимно просто с |ALPH|={M}")
    a_inv = modinv(a, M)
    out = []
    for ch in text:
        if ch in INDEX:
            y = INDEX[ch]
            x = (a_inv * (y - b)) % M
            out.append(ALPH[x])
        else:
            out.append(ch)
    return "".join(out)

DIGITS = "0123456789"

def _digits_from_key(key: str) -> List[int]:
    if not key or any(c not in DIGITS for c in key):
        raise ValueError("Ключ Гронсфельда — только цифры, напр. '31415'")
    return [int(c) for c in key]

def p7_gronsfeld_encrypt(text: str, key: str) -> str:
    if M == 0:
        raise ValueError("Алфавит пуст.")
    ks = _digits_from_key(key)
    klen = len(ks)
    out, j = [], 0
    for ch in text:
        if ch in INDEX:
            shift = ks[j % klen]
            x = INDEX[ch]
            y = (x + shift) % M
            out.append(ALPH[y])
            j += 1
        else:
            out.append(ch)
    return "".join(out)

def p7_gronsfeld_decrypt(text: str, key: str) -> str:
    if M == 0:
        raise ValueError("Алфавит пуст.")
    ks = _digits_from_key(key)
    klen = len(ks)
    out, j = [], 0
    for ch in text:
        if ch in INDEX:
            shift = ks[j % klen]
            y = INDEX[ch]
            x = (y - shift) % M
            out.append(ALPH[x])
            j += 1
        else:
            out.append(ch)
    return "".join(out)

def affine_steps_table(plain: str, a: int, b: int, limit: int = 40) -> str:
    rows = []
    for i, ch in enumerate(plain[:limit]):
        if ch in INDEX:
            x = INDEX[ch]
            y = (a * x + b) % M
            ce = ALPH[y]
            rows.append([i, repr(ch)[1:-1], x, f"(a*x+b)%M=({a}*{x}+{b})%{M}={y}", repr(ce)[1:-1]])
        else:
            rows.append([i, repr(ch)[1:-1], "-", "-", repr(ch)[1:-1]])
    return md_table(["i", "ch", "x", "формула → y", "enc"], rows)

def affine_back_steps_table(cipher: str, a: int, b: int, limit: int = 40) -> str:
    a_inv = modinv(a, M)
    rows = []
    for i, ch in enumerate(cipher[:limit]):
        if ch in INDEX:
            y = INDEX[ch]
            x = (a_inv * (y - b)) % M
            pl = ALPH[x]
            rows.append([i, repr(ch)[1:-1], y, f"x=a^(-1)*(y-b)%M={a_inv}*({y}-{b})%{M}={x}", repr(pl)[1:-1]])
        else:
            rows.append([i, repr(ch)[1:-1], "-", "-", repr(ch)[1:-1]])
    return md_table(["i", "ch", "y", "формула → x", "dec"], rows)

def gronsfeld_steps_table(plain: str, key: str, limit: int = 40) -> str:
    ks = _digits_from_key(key); klen = len(ks)
    rows = []
    j = 0
    for i, ch in enumerate(plain[:limit]):
        if ch in INDEX:
            x = INDEX[ch]; s = ks[j % klen]; y = (x + s) % M; ce = ALPH[y]
            rows.append([i, repr(ch)[1:-1], x, s, f"(x+s)%M=({x}+{s})%{M}={y}", repr(ce)[1:-1]])
            j += 1
        else:
            rows.append([i, repr(ch)[1:-1], "-", "-", "-", repr(ch)[1:-1]])
    return md_table(["i", "ch", "x", "key[i]", "формула → y", "enc"], rows)

def gronsfeld_back_steps_table(cipher: str, key: str, limit: int = 40) -> str:
    ks = _digits_from_key(key); klen = len(ks)
    rows = []
    j = 0
    for i, ch in enumerate(cipher[:limit]):
        if ch in INDEX:
            y = INDEX[ch]; s = ks[j % klen]; x = (y - s) % M; pl = ALPH[x]
            rows.append([i, repr(ch)[1:-1], y, s, f"(y-s)%M=({y}-{s})%{M}={x}", repr(pl)[1:-1]])
            j += 1
        else:
            rows.append([i, repr(ch)[1:-1], "-", "-", "-", repr(ch)[1:-1]])
    return md_table(["i", "ch", "y", "key[i]", "формула → x", "dec"], rows)


# -----------------------------
# П8 — Профиль, Цезарь, χ²
# -----------------------------

def count_freq(text: str) -> Tuple[Dict[str, int], int]:
    cnt = Counter(ch for ch in text if ch in INDEX)
    total = sum(cnt.values())
    return dict(cnt), total

def normalize_counts(counts: Dict[str, int], total: int) -> Dict[str, float]:
    if total == 0:
        return {ch: 0.0 for ch in ALPH}
    return {ch: counts.get(ch, 0) / total for ch in ALPH}

def save_profile(profile: Dict[str, float], path: str):
    path = to_outpath(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"alphabet": ALPH, "profile": profile}, f, ensure_ascii=False, indent=2)

def load_profile_and_set_alphabet(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    alph = data.get("alphabet", "")
    if not alph:
        raise ValueError("В профиле отсутствует alphabet.")
    set_alphabet(alph)
    return data["profile"]

def _rotate_counts(counts: Dict[str, int], shift: int) -> Dict[str, int]:
    """Повернуть частоты на shift (для оценки соответствия при предполагаемом сдвиге)."""
    rot = {ch: 0 for ch in ALPH}
    for ch, c in counts.items():
        if ch in INDEX:
            idx = INDEX[ch]
            idx2 = (idx - shift) % M  # «сдвигаем обратно»
            rot[ALPH[idx2]] = rot.get(ALPH[idx2], 0) + c
    return rot

def _chisquare(observed: Dict[str, int], profile: Dict[str, float]) -> float:
    """Классическая метрика χ²: меньше — лучше (ближе к профилю)."""
    n = sum(observed.values())
    if n == 0:
        return float("inf")
    s = 0.0
    for ch in ALPH:
        exp = profile.get(ch, 0.0) * n
        obs = observed.get(ch, 0)
        if exp > 0:
            s += (obs - exp) ** 2 / exp
    return s

def caesar_encrypt(text: str, shift: int) -> str:
    if M == 0:
        raise ValueError("Алфавит пуст.")
    k = shift % M
    out = []
    for ch in text:
        if ch in INDEX:
            x = INDEX[ch]
            y = (x + k) % M
            out.append(ALPH[y])
        else:
            out.append(ch)
    return "".join(out)

def caesar_attack_full(cipher_text: str, profile: Dict[str, float]) -> List[Tuple[int, float]]:
    """Полная таблица χ² по всем сдвигам."""
    counts, _ = count_freq(cipher_text)
    scores = []
    for k in range(M):
        rot = _rotate_counts(counts, k)
        chi = _chisquare(rot, profile)
        scores.append((k, chi))
    scores.sort(key=lambda x: x[1])
    return scores

# -----------------------------
# Вспомогательные сохранения (CSV)
# -----------------------------

def save_profile_csv(profile: Dict[str, float], path_csv: str):
    """Сохраняем таблицу профиля: символ, ord, freq, cumfreq (по убыванию freq)."""
    path_csv = to_outpath(path_csv)
    rows = []
    cum = 0.0
    for ch, freq in sorted(profile.items(), key=lambda kv: kv[1], reverse=True):
        cum += freq
        rows.append([ch, ord(ch), f"{freq:.8f}", f"{cum:.8f}"])
    ensure_outdir()
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["char", "ord", "freq", "cumfreq"])
        w.writerows(rows)

def save_chi2_table_csv(scores: List[Tuple[int, float]], path_csv: str):
    """Сохраняем χ² по всем сдвигам."""
    path_csv = to_outpath(path_csv)
    ensure_outdir()
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["shift", "chi2"])
        for k, chi in scores:
            w.writerow([k, f"{chi:.6f}"])

# -----------------------------
# Команды CLI
# -----------------------------

def cmd_p8_build_profile(args):
    """Построение профиля и (при необходимости) нового алфавита по корпусу+сообщению."""
    texts_for_alph = []
    corpus_raw = read_text(args.corpus) if os.path.exists(args.corpus) else ""
    texts_for_alph.append(corpus_raw)
    msg_raw = read_text(args.message) if (args.message and os.path.exists(args.message)) else ""
    if msg_raw:
        texts_for_alph.append(msg_raw)

    alph = derive_alphabet_from_texts(texts_for_alph, lower=not args.no_lower)
    set_alphabet(alph)

    corpus_for_profile = corpus_raw if args.no_lower else corpus_raw.lower()
    counts = Counter(ch for ch in corpus_for_profile if ch in INDEX)
    total = sum(counts.values())
    profile = normalize_counts(counts, total)

    save_profile(profile, args.profile)
    save_profile_csv(profile, args.profile_csv)
    if args.alphabet_json:
        save_alphabet(args.alphabet_json, ALPH)

    print(f"[P8] Профиль: {to_outpath(args.profile)} | таблица: {to_outpath(args.profile_csv)}")
    print(f"     Алфавит: |ALPH|={M}. Пример: {repr(ALPH[:80])}{'…' if M>80 else ''}")
    if args.message and msg_raw:
        print(f"     В алфавит включены символы из '{args.message}'.")

def cmd_p8_caesar_encrypt(args):
    load_profile_and_set_alphabet(args.profile)
    plain = read_text(args.input)
    if not args.no_lower:
        plain = plain.lower()
    cipher = caesar_encrypt(plain, args.shift)
    outp = to_outpath(args.output)
    write_text(outp, cipher)
    meta = {"method": "caesar", "shift": args.shift, "alphabet_len": M}
    write_text(to_outpath(args.output + ".meta.json"), json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[P8][Caesar] encrypt shift={args.shift % M} -> {outp}")

def cmd_p8_caesar_decrypt(args):
    """Явная расшифровка Цезаря по известному сдвигу (применяется -shift)."""
    load_profile_and_set_alphabet(args.profile)
    cipher = read_text(args.input)
    if not args.no_lower:
        cipher = cipher.lower()
    plain = caesar_encrypt(cipher, -args.shift)
    outp = to_outpath(args.output)
    write_text(outp, plain)
    meta = {"method": "caesar-decode", "shift": args.shift, "alphabet_len": M}
    write_text(to_outpath(args.output + ".meta.json"), json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[P8][Caesar] decrypt shift={args.shift % M} -> {outp}")

def cmd_p8_caesar_attack(args):
    profile = load_profile_and_set_alphabet(args.profile)
    cipher = read_text(args.cipher)
    if not args.no_lower:
        cipher = cipher.lower()
    scores = caesar_attack_full(cipher, profile)
    if args.debug:
        save_chi2_table_csv(scores, args.chi2_csv)
        print(f"[P8][Attack] χ² по всем сдвигам сохранён в {to_outpath(args.chi2_csv)}")
    top = scores[:args.topk]
    print("[P8][Attack] Top candidates (lower χ² is better):")
    for k, chi in top:
        print(f"  shift={k:2d}  chi2={chi:.4f}")
    best_k = top[0][0]
    best_plain = caesar_encrypt(cipher, -best_k)
    outp = to_outpath(args.output)
    write_text(outp, best_plain)
    meta = {"attack": "caesar-chi2", "best_shift": best_k, "alphabet_len": M}
    write_text(to_outpath(args.output + ".meta.json"), json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[P8] Best shift={best_k}. Decryption saved to {outp}")

def cmd_p7_affine_encrypt(args):
    load_profile_and_set_alphabet(args.profile)
    text = read_text(args.input)
    if not args.no_lower:
        text = text.lower()
    enc = p7_affine_encrypt(text, args.a, args.b)
    outp = to_outpath(args.output)
    write_text(outp, enc)
    meta = {"method": "affine", "a": args.a, "b": args.b, "alphabet_len": M}
    write_text(to_outpath(args.output + ".meta.json"), json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[P7][Affine] encrypt a={args.a} b={args.b} -> {outp}")

def cmd_p7_affine_decrypt(args):
    load_profile_and_set_alphabet(args.profile)
    text = read_text(args.input)
    if not args.no_lower:
        text = text.lower()
    dec = p7_affine_decrypt(text, args.a, args.b)
    outp = to_outpath(args.output)
    write_text(outp, dec)
    meta = {"method": "affine-decode", "a": args.a, "b": args.b, "alphabet_len": M}
    write_text(to_outpath(args.output + ".meta.json"), json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[P7][Affine] decrypt a={args.a} b={args.b} -> {outp}")

def cmd_p7_gronsfeld_encrypt(args):
    load_profile_and_set_alphabet(args.profile)
    text = read_text(args.input)
    if not args.no_lower:
        text = text.lower()
    enc = p7_gronsfeld_encrypt(text, args.key)
    outp = to_outpath(args.output)
    write_text(outp, enc)
    meta = {"method": "gronsfeld", "key": args.key, "alphabet_len": M}
    write_text(to_outpath(args.output + ".meta.json"), json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[P7][Gronsfeld] encrypt key={args.key} -> {outp}")

def cmd_p7_gronsfeld_decrypt(args):
    load_profile_and_set_alphabet(args.profile)
    text = read_text(args.input)
    if not args.no_lower:
        text = text.lower()
    dec = p7_gronsfeld_decrypt(text, args.key)
    outp = to_outpath(args.output)
    write_text(outp, dec)
    meta = {"method": "gronsfeld-decode", "key": args.key, "alphabet_len": M}
    write_text(to_outpath(args.output + ".meta.json"), json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[P7][Gronsfeld] decrypt key={args.key} -> {outp}")

# -----------------------------
# Отчёт: безопасные помощники
# -----------------------------

def md_table(headers: List[str], rows: List[List[str]]) -> str:
    """
    Простейшая Markdown-таблица:
    headers = ["колонка1", "колонка2"]
    rows = [["a","1"], ["b","2"]]
    """
    line_head = "| " + " | ".join(headers) + " |"
    line_sep  = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [line_head, line_sep]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def profile_top_md_table(profile: Dict[str, float], limit: int = 50) -> str:
    """ТОП-символы профиля как Markdown-таблица (символ, ord, freq, cumfreq)."""
    rows = []
    cum = 0.0
    for ch, freq in sorted(profile.items(), key=lambda kv: kv[1], reverse=True)[:limit]:
        cum += freq
        shown = ch if ch not in ("\n", "\t", "\r") else repr(ch)
        rows.append([shown, ord(ch), f"{freq:.6f}", f"{cum:.6f}"])
    return md_table(["символ", "ord", "freq", "cumfreq"], rows)

def attack_top_md_table(cipher: str, profile: Dict[str, float], topk: int = 10) -> Tuple[str, List[Tuple[int,float]]]:
    """
    Возвращает:
      - markdown-таблицу (shift, chi2) для TOP-k
      - список TOP-k (для последующего best_k)
    """
    scores = caesar_attack_full(cipher, profile)
    top = scores[:topk]
    rows = [[k, f"{chi:.6f}"] for k, chi in top]
    table = md_table(["shift", "chi2"], rows)
    return table, top


def fence(code: str) -> str:
    # Без f-строк, чтобы не ловить конфликтов с { } и ```
    return "```\n" + str(code) + "\n```"

def _profile_top_lines(profile: Dict[str, float], limit: int = 50) -> str:
    rows = sorted(profile.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    out = "символ | ord | freq\n" + "-" * 26 + "\n"
    for ch, freq in rows:
        shown = ch if ch not in ("\n", "\t", "\r") else repr(ch)
        out += "{} | {:>4d} | {:.6f}\n".format(shown, ord(ch), freq)
    return out

def _attack_log(cipher: str, profile: Dict[str, float], topk: int = 10) -> Tuple[str, List[Tuple[int, float]]]:
    scores = caesar_attack_full(cipher, profile)
    top = scores[:topk]
    log = "shift | chi2\n" + "--------------\n"
    for k, chi in top:
        log += "{:>5d} | {:.6f}\n".format(k, chi)

    # Пояснение по лучшему сдвигу
    best_k = top[0][0]
    counts, _ = count_freq(cipher)
    rot = _rotate_counts(counts, best_k)
    n_better = sum(1 for ch in ALPH if (rot.get(ch, 0) > 0) == (profile.get(ch, 0.0) > 0.0))
    log += "\nЛучший сдвиг k={}: совпадений ненулевых частот {} из {}.\n".format(best_k, n_better, M)
    return log, top

# -----------------------------
# Отчёт (П7 + П8)
# -----------------------------

def cmd_report(args):
    # Загрузка профиля (он же задаёт алфавит)
    try:
        profile = load_profile_and_set_alphabet(args.profile)
        profile_ok, profile_err = True, ""
    except Exception as e:
        profile_ok, profile_err, profile = False, str(e), {}

    def maybe_lower(s: str) -> str:
        return s if getattr(args, "no_lower", False) else s.lower()

    # Полные тексты
    corpus  = maybe_lower(read_text(args.corpus))  if os.path.exists(args.corpus)  else ""
    message = maybe_lower(read_text(args.message)) if os.path.exists(args.message) else ""

    def read_from_out_or_root(name: str) -> str:
        p1 = to_outpath(name)
        if os.path.exists(p1): return read_text(p1)
        if os.path.exists(name): return read_text(name)
        return ""

    caesar_cipher  = read_from_out_or_root(args.cipher)
    caesar_decoded = read_from_out_or_root(args.decoded)

    # === П7: аффинный / Гронсфельд (полные тексты + промежуточные шаги) ===
    aff_enc = p7_affine_encrypt(message, args.a, args.b) if (message and profile_ok) else ""
    aff_dec = p7_affine_decrypt(aff_enc, args.a, args.b) if aff_enc else ""
    gr_enc  = p7_gronsfeld_encrypt(message, args.key)     if (message and profile_ok) else ""
    gr_dec  = p7_gronsfeld_decrypt(gr_enc, args.key)      if gr_enc else ""

    # Табличные «шаги» (по первым N символам)
    N = 40
    aff_steps_enc = affine_steps_table(message, args.a, args.b, limit=N) if aff_enc else ""
    aff_steps_dec = affine_back_steps_table(aff_enc, args.a, args.b, limit=N) if aff_enc else ""
    gr_steps_enc  = gronsfeld_steps_table(message, args.key, limit=N) if gr_enc else ""
    gr_steps_dec  = gronsfeld_back_steps_table(gr_enc, args.key, limit=N) if gr_enc else ""

    # === П8: χ² и таблица TOP-k ===
    attack_table_md, top_scores = ("", [])
    if profile_ok and caesar_cipher:
        cipher_for_metric = caesar_cipher if getattr(args, "no_lower", False) else caesar_cipher.lower()
        attack_table_md, top_scores = attack_top_md_table(cipher_for_metric, profile, topk=10)

    # Сборка Markdown
    md = []
    md.append("# Отчёт по практическим работам 7 и 8\n")
    md.append(f"**Длина алфавита:** {M}\n\n")
    md.append("## Алфавит (первые 120 знаков в порядке появления)\n")
    preview = ALPH[:120] + ("…" if len(ALPH) > 120 else "")
    md.append("```\n" + preview + "\n```\n")
    md.append("\n---\n\n")

    md.append("## Профиль частот (срез)\n")
    if profile_ok:
        md.append("Полная таблица в файле: `profile_table.csv`.\n\n")
        md.append(profile_top_md_table(profile, 50) + "\n")
    else:
        md.append(f"Не удалось загрузить профиль: {profile_err}\n")

    md.append("\n---\n\n")
    md.append("## Практическая 7 — параметры\n")
    md.append(f"- Аффинный: `a = {args.a}`, `b = {args.b}`\n")
    md.append(f"- Гронсфельд: `key = {args.key}`\n")

    if message:
        md.append("\n**Исходное сообщение (mes.txt, полностью):**\n")
        md.append("```\n" + message + "\n```\n")

    if aff_enc:
        md.append("\n### Аффинный — шифрование (полный текст)\n")
        md.append("```\n" + aff_enc + "\n```\n")
        md.append("\n**Промежуточные шаги (первые 40 символов):**\n")
        md.append(aff_steps_enc + "\n")
    if aff_dec:
        md.append("\n### Аффинный — расшифрование (полный текст)\n")
        md.append("```\n" + aff_dec + "\n```\n")
        md.append("\n**Обратные шаги (первые 40 символов):**\n")
        md.append(aff_steps_dec + "\n")

    if gr_enc:
        md.append("\n### Гронсфельд — шифрование (полный текст)\n")
        md.append("```\n" + gr_enc + "\n```\n")
        md.append("\n**Промежуточные шаги (первые 40 символов):**\n")
        md.append(gr_steps_enc + "\n")
    if gr_dec:
        md.append("\n### Гронсфельд — расшифрование (полный текст)\n")
        md.append("```\n" + gr_dec + "\n```\n")
        md.append("\n**Обратные шаги (первые 40 символов):**\n")
        md.append(gr_steps_dec + "\n")

    md.append("\n---\n\n")
    md.append("## Практическая 8 — частотный анализ Цезаря\n")
    if corpus:
        md.append("**Корпус (полностью):**\n```\n" + corpus + "\n```\n")
    if caesar_cipher:
        md.append("\n**Цезарь — шифртекст (полностью):**\n```\n" + caesar_cipher + "\n```\n")
    if caesar_decoded:
        md.append("\n**Цезарь — расшифровка (лучший сдвиг, полностью):**\n```\n" + caesar_decoded + "\n```\n")
    if attack_table_md:
        md.append("\n**Подбор сдвига (χ², ТОП-10):**\n")
        md.append(attack_table_md + "\n")
        md.append(f"\nЛучший сдвиг по χ²: **k = {top_scores[0][0]}**.\n")

    md.append("\n---\n\n")
    md.append("## Выводы\n")
    md.append("- Полные тексты для всех стадий добавлены в отчёт.\n")
    md.append("- Для аффинного и Гронсфельда показаны пошаговые вычисления по первым 40 символам (индексы, формулы, ключ).\n")
    md.append("- χ²-таблица иллюстрирует выбор корректного сдвига Цезаря.\n")

    outp = to_outpath(args.out)
    write_text(outp, "".join(md))
    print(f"[REPORT] Markdown-отчёт сохранён в {outp}")


def quickstart():
    ensure_outdir()

    corpus_path  = to_outpath("text.txt")
    mes_path     = to_outpath("mes.txt")
    profile_path = to_outpath("profile.json")
    profile_csv  = to_outpath("profile_table.csv")
    cipher_path  = to_outpath("cipher.txt")
    chi2_csv     = to_outpath("chi2_by_shift.csv")
    decoded_path = to_outpath("decoded.txt")
    report_path  = to_outpath("report_prac7_8.md")

    if not os.path.exists(corpus_path):
        write_text(corpus_path, "пример большого текста для корпуса (добавь сюда свой текст).")
    if not os.path.exists(mes_path):
        write_text(mes_path, "пример сообщения для шифрования. добавь сюда свой текст.")

    class P:
        corpus = corpus_path
        message = mes_path
        profile = "profile.json"
        profile_csv = "profile_table.csv"
        alphabet_json = "alphabet.json"
        no_lower = False
    cmd_p8_build_profile(P)

    class C1:
        profile = profile_path
        input = mes_path
        shift = 9
        output = "cipher.txt"
        no_lower = False
    cmd_p8_caesar_encrypt(C1)

    class C2:
        profile = profile_path
        cipher = cipher_path
        output = "decoded.txt"
        topk = 10
        no_lower = False
        debug = True
        chi2_csv = "chi2_by_shift.csv"
    cmd_p8_caesar_attack(C2)

    # явная расшифровка тем же сдвигом (пример)
    dec_byshift_path = to_outpath("decoded.byshift.txt")
    write_text(dec_byshift_path, caesar_encrypt(read_text(cipher_path), -7))
    print(f"[Quickstart] Caesar decrypt by shift=7 -> {dec_byshift_path}")

    # П7: аффинный и Гронсфельд — enc + dec
    plain = read_text(mes_path).lower()
    aff = p7_affine_encrypt(plain, a=5, b=8)
    aff_enc_path = to_outpath("affine.enc.txt")
    write_text(aff_enc_path, aff)
    aff_dec = p7_affine_decrypt(aff, a=5, b=8)
    write_text(to_outpath("affine.dec.txt"), aff_dec)

    gr = p7_gronsfeld_encrypt(plain, key="31415")
    gr_enc_path = to_outpath("gr.enc.txt")
    write_text(gr_enc_path, gr)
    gr_dec = p7_gronsfeld_decrypt(gr, key="31415")
    write_text(to_outpath("gr.dec.txt"), gr_dec)

    class R:
        out = "report_prac7_8.md"
        ipynb_out = "report_prac7_8.ipynb"
        profile = profile_path
        corpus = corpus_path
        message = mes_path
        cipher = "cipher.txt"
        decoded = "decoded.txt"
        a = 5
        b = 8
        key = "31415"
        no_lower = False
    cmd_report(R)

    print(f"[Quickstart] Готово. Отчёт: {report_path}")
    print(f"[Quickstart] Профиль CSV: {profile_csv}")
    print(f"[Quickstart] χ² по сдвигам: {chi2_csv}")

# -----------------------------
# Аргументы
# -----------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Практики 7 и 8: динамический алфавит, профиль, Цезарь, аффинный, Гронсфельд, отчёт",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=False)

    # Профиль
    s = sub.add_parser("p8-build-profile", help="П8: Построение профиля и алфавита")
    s.add_argument("--corpus", default="text.txt")
    s.add_argument("--message", default="mes.txt")
    s.add_argument("--profile", default="profile.json")
    s.add_argument("--profile-csv", default="profile_table.csv")
    s.add_argument("--alphabet-json", default="alphabet.json")
    s.add_argument("--no-lower", action="store_true")
    s.set_defaults(func=cmd_p8_build_profile)

    # Цезарь: шифрование
    s = sub.add_parser("p8-caesar-encrypt", help="П8: Цезарь — шифрование")
    s.add_argument("--profile", default="profile.json")
    s.add_argument("--input", default="mes.txt")
    s.add_argument("--shift", type=int, default=7)
    s.add_argument("--output", default="cipher.txt")
    s.add_argument("--no-lower", action="store_true")
    s.set_defaults(func=cmd_p8_caesar_encrypt)

    # Цезарь: явная расшифровка
    s = sub.add_parser("p8-caesar-decrypt", help="П8: Цезарь — расшифрование по заданному сдвигу")
    s.add_argument("--profile", default="profile.json")
    s.add_argument("--input", default="cipher.txt")
    s.add_argument("--shift", type=int, default=7, help="Сдвиг, которым шифровали (применится со знаком минус)")
    s.add_argument("--output", default="decoded.byshift.txt")
    s.add_argument("--no-lower", action="store_true")
    s.set_defaults(func=cmd_p8_caesar_decrypt)

    # Цезарь: взлом
    s = sub.add_parser("p8-caesar-attack", help="П8: Цезарь — частотный взлом (χ²)")
    s.add_argument("--profile", default="profile.json")
    s.add_argument("--cipher", default="cipher.txt")
    s.add_argument("--output", default="decoded.txt")
    s.add_argument("--topk", type=int, default=10)
    s.add_argument("--no-lower", action="store_true")
    s.add_argument("--debug", action="store_true", help="Сохранить chi2_by_shift.csv")
    s.add_argument("--chi2-csv", default="chi2_by_shift.csv")
    s.set_defaults(func=cmd_p8_caesar_attack)

    # П7 — Аффинный
    s = sub.add_parser("p7-affine-encrypt", help="П7: аффинный — шифрование")
    s.add_argument("--profile", default="profile.json")
    s.add_argument("--input", default="mes.txt")
    s.add_argument("--a", type=int, default=5)
    s.add_argument("--b", type=int, default=8)
    s.add_argument("--output", default="affine.enc.txt")
    s.add_argument("--no-lower", action="store_true")
    s.set_defaults(func=cmd_p7_affine_encrypt)

    s = sub.add_parser("p7-affine-decrypt", help="П7: аффинный — расшифрование")
    s.add_argument("--profile", default="profile.json")
    s.add_argument("--input", default="affine.enc.txt")
    s.add_argument("--a", type=int, default=5)
    s.add_argument("--b", type=int, default=8)
    s.add_argument("--output", default="affine.dec.txt")
    s.add_argument("--no-lower", action="store_true")
    s.set_defaults(func=cmd_p7_affine_decrypt)

    # П7 — Гронсфельд
    s = sub.add_parser("p7-gronsfeld-encrypt", help="П7: Гронсфельд — шифрование")
    s.add_argument("--profile", default="profile.json")
    s.add_argument("--input", default="mes.txt")
    s.add_argument("--key", default="31415")
    s.add_argument("--output", default="gr.enc.txt")
    s.add_argument("--no-lower", action="store_true")
    s.set_defaults(func=cmd_p7_gronsfeld_encrypt)

    s = sub.add_parser("p7-gronsfeld-decrypt", help="П7: Гронсфельд — расшифрование")
    s.add_argument("--profile", default="profile.json")
    s.add_argument("--input", default="gr.enc.txt")
    s.add_argument("--key", default="31415")
    s.add_argument("--output", default="gr.dec.txt")
    s.add_argument("--no-lower", action="store_true")
    s.set_defaults(func=cmd_p7_gronsfeld_decrypt)

    # Отчёт
    s = sub.add_parser("report", help="Сформировать Markdown-отчёт")
    s.add_argument("--out", default="report_prac7_8.md")
    s.add_argument("--ipynb-out", default="report_prac7_8.ipynb")
    s.add_argument("--profile", default="profile.json")
    s.add_argument("--corpus", default="text.txt")
    s.add_argument("--message", default="mes.txt")
    s.add_argument("--cipher", default="cipher.txt")
    s.add_argument("--decoded", default="decoded.txt")
    s.add_argument("--a", type=int, default=5)
    s.add_argument("--b", type=int, default=8)
    s.add_argument("--key", default="31415")
    s.add_argument("--no-lower", action="store_true")
    s.set_defaults(func=cmd_report)

    return p

# -----------------------------
# main
# -----------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Без подкоманды — быстрый сценарий
    if not getattr(args, "cmd", None):
        print("[Info] Команда не указана — выполняю сценарий по умолчанию (quickstart).")
        quickstart()
        return

    args.func(args)

if __name__ == "__main__":
    main()
