#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, random

'''
Задание 2
'''

def minimal_r_for_k(k: int) -> int:
    r = 1
    while (1 << r) < (k + r + 1):
        r += 1
    return r

def positions(n, r):
    parity = {1 << i for i in range(r)}
    data = [p for p in range(1, n + 1) if p not in parity]
    return parity, data

def hamming_encode_bits(data_bits: str):
    k = len(data_bits)
    r = minimal_r_for_k(k)
    n = (1 << r) - 1
    parity_pos, data_pos = positions(n, r)
    code = [0] * n
    for i, p in enumerate(data_pos[:k]):
        code[p - 1] = 1 if data_bits[i] == "1" else 0
    for j in range(r):
        ppos = 1 << j
        s = 0
        for pos in range(1, n + 1):
            if pos & ppos:
                s ^= code[pos - 1]
        code[ppos - 1] = s
    return "".join(str(b) for b in code), r, n, k

def syndrome(codeword: str, r: int) -> int:
    n = len(codeword)
    s = 0
    for j in range(r):
        ppos = 1 << j
        t = 0
        for pos in range(1, n + 1):
            if pos & ppos:
                t ^= int(codeword[pos - 1])
        if t:
            s |= ppos
    return s

def main():
    data = input("Введите информационную комбинацию (0/1): ").strip().replace(" ", "")
    if not data or any(c not in "01" for c in data):
        print("Ожидалась двоичная строка из 0/1."); sys.exit(1)

    code, r, n, k = hamming_encode_bits(data)
    print(f"k={k}, r={r}, n={n}")
    print(f"Скорость k/n={k/n:.3f}, избыточность r/n={r/n:.3f}")
    print("Кодовый вектор:", code)

    with open("hamming_codeword.txt", "w", encoding="utf-8") as f:
        f.write(code)

    pos_err = random.randint(1, n)
    corrupted = list(code)
    corrupted[pos_err - 1] = "1" if corrupted[pos_err - 1] == "0" else "0"
    corrupted = "".join(corrupted)
    print(f"Внесена ошибка в позиции {pos_err}: {corrupted}")

    s = syndrome(corrupted, r)
    print(f"Синдром: {s} (bin: {s:b})")

    corrected = list(corrupted)
    if 1 <= s <= n:
        corrected[s - 1] = "1" if corrected[s - 1] == "0" else "0"
    corrected = "".join(corrected)
    print("Исправлено:", corrected)
    print("Сравнение с исходным кодовым вектором:", "совпадает" if corrected == code else "НЕ совпадает")

if __name__ == "__main__":
    main()
