#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 2: Hamming code (table method)
"""

import sys
import random

def hamming_generate_tables(r):
    n = 2**r - 1
    k = n - r
    positions = list(range(1, n+1))
    parity_positions = {2**i for i in range(r)}
    data_positions = [p for p in positions if p not in parity_positions]
    return n, k, positions, parity_positions, data_positions

def hamming_encode(data_bits, r):
    n, k, positions, parity_positions, data_positions = hamming_generate_tables(r)
    if len(data_bits) != k:
        raise ValueError(f"Ожидается {k} информационных бит для r={r}.")
    code = [0]*(n+1)  # 1-indexed
    for idx, pos in enumerate(data_positions):
        code[pos] = data_bits[idx]
    for i in range(r):
        ppos = 2**i
        s = 0
        for pos in positions:
            if pos & ppos and pos != ppos:
                s ^= code[pos]
        code[ppos] = s
    return code[1:]

def hamming_syndrome(code_bits, r):
    n = len(code_bits)
    positions = list(range(1, n+1))
    syn = 0
    for i in range(r):
        ppos = 2**i
        s = 0
        for pos in positions:
            if pos & ppos:
                s ^= code_bits[pos-1]
        if s:
            syn |= ppos
    return syn

def main():
    try:
        r = int(input("Введите r (число проверочных битов): ").strip())
    except Exception:
        print("Некорректное r"); sys.exit(1)
    n, k, positions, parity_positions, data_positions = hamming_generate_tables(r)
    try:
        k_in = int(input(f"Введите k (число информационных битов, должно быть {k}): ").strip())
    except Exception:
        print("Некорректное k"); sys.exit(1)
    if k_in != k:
        print(f"Внимание: для r={r} корректное k={k}. Будет использовано k={k}.")
    data = [random.randint(0,1) for _ in range(k)]
    print("Информационная комбинация:", ''.join(map(str,data)))
    code = hamming_encode(data, r)
    code_str = ''.join(map(str, code))
    with open("hamming_codeword.txt", "w", encoding="utf-8") as f:
        f.write(code_str)
    print("Кодовый вектор:", code_str)
    err_pos = random.randint(1, len(code))
    code_err = code[:]
    code_err[err_pos-1] ^= 1
    code_err_str = ''.join(map(str, code_err))
    print(f"Внесена ошибка в позиции {err_pos}: {code_err_str}")
    syn = hamming_syndrome(code_err, r)
    print(f"Синдром: {syn} (bin: {bin(syn)[2:] or '0'})")
    if syn != 0 and 1 <= syn <= len(code_err):
        code_fix = code_err[:]
        code_fix[syn-1] ^= 1
        print("Исправлено:", ''.join(map(str, code_fix)))
        ok = "совпадает" if code_fix == code else "НЕ совпадает"
        print("Сравнение с исходным кодовым вектором:", ok)
    else:
        print("Ошибок не обнаружено или позиция вне диапазона.")

if __name__ == "__main__":
    main()
