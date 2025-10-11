#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 1: Text coding & compression (Shannon–Fano, Huffman).
"""

import math
import csv
import sys
from collections import Counter

def write_csv(path, header, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(header)
        for row in rows:
            w.writerow(row)

def entropy(prob_map):
    return -sum(p * math.log2(p) for p in prob_map.values() if p > 0)

def uniform_code_length(alphabet_size):
    if alphabet_size <= 1:
        return 0
    return math.ceil(math.log2(alphabet_size))

def average_code_length(prob_map, code_map):
    return sum(prob_map[s] * len(code_map[s]) for s in prob_map)

def build_freq(text, ngram=1):
    if ngram == 1:
        return dict(Counter(text))
    tokens = [text[i:i+ngram] for i in range(len(text)-ngram+1)]
    return dict(Counter(tokens))

def probs_from_freq(freq):
    total = sum(freq.values())
    return {k: v/total for k, v in freq.items()} if total>0 else {}

def shannon_fano(prob_map):
    items = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)
    codes = {sym: "" for sym, _ in items}
    def split(sub):
        if len(sub) <= 1:
            return
        total = sum(p for _, p in sub)
        cum = 0.0
        split_idx = 0
        best = float('inf')
        for i in range(len(sub)-1):
            cum += sub[i][1]
            diff = abs(cum - total/2)
            if diff < best:
                best = diff
                split_idx = i
        left = sub[:split_idx+1]
        right = sub[split_idx+1:]
        for s, _ in left: codes[s] += '0'
        for s, _ in right: codes[s] += '1'
        split(left)
        split(right)
    split(items)
    return codes

def huffman(prob_map):
    import heapq
    heap = []
    for s, p in prob_map.items():
        heapq.heappush(heap, (p, len(heap), s))
    parent = {}
    nodes_count = len(heap)
    while len(heap) > 1:
        p1, _, n1 = heapq.heappop(heap)
        p2, _, n2 = heapq.heappop(heap)
        merged = f"@{nodes_count}"
        nodes_count += 1
        parent[n1] = (merged, '0')
        parent[n2] = (merged, '1')
        heapq.heappush(heap, (p1+p2, len(heap), merged))
    root = heap[0][2]
    codes = {}
    for s in prob_map:
        bits = []
        node = s
        while node != root:
            par, bit = parent[node]
            bits.append(bit)
            node = par
        codes[s] = ''.join(reversed(bits)) if bits else '0'
    return codes

def encode(text, code_map):
    return ''.join(code_map[ch] for ch in text)

def decode(bits, code_map):
    trie = {}
    for s, c in code_map.items():
        node = trie
        for b in c:
            node = node.setdefault(b, {})
        node['$'] = s
    out = []
    node = trie
    for b in bits:
        if b not in node:
            raise ValueError("Invalid bit sequence")
        node = node[b]
        if '$' in node:
            out.append(node['$'])
            node = trie
    if node is not trie:
        raise ValueError("Trailing bits don't resolve")
    return ''.join(out)

def save_alphabet_csv(path, freq, prob):
    rows = []
    for s in sorted(freq.keys(), key=lambda k: (-prob[k], k)):
        rows.append([repr(s), freq[s], f"{prob[s]:.6f}"])
    write_csv(path, ["Symbol", "Count", "Probability"], rows)

def save_code_csv(path, code_map, prob):
    rows = []
    for s in sorted(code_map.keys(), key=lambda k: (-prob[k], k)):
        rows.append([repr(s), f"{prob[s]:.6f}", code_map[s], len(code_map[s])])
    write_csv(path, ["Symbol", "Probability", "Code", "Length"], rows)

def pipeline(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    # Unigrams
    freq1 = build_freq(text, 1)
    prob1 = probs_from_freq(freq1)
    save_alphabet_csv("alphabet_1gram.csv", freq1, prob1)

    H = entropy(prob1)
    Lu = uniform_code_length(len(freq1))
    R = Lu - H
    print(f"[1-граммы] H = {H:.6f} ; Lu = {Lu} ; R = {R:.6f}")

    code_sf = shannon_fano(prob1)
    save_code_csv("code_1gram_shannon_fano.csv", code_sf, prob1)
    Lavg = average_code_length(prob1, code_sf)
    eta = H / Lavg if Lavg>0 else 0.0
    print(f"[1-граммы, Шеннон–Фано] Lavg = {Lavg:.6f} ; η = {eta:.6f}")

    enc = encode(text, code_sf)
    with open("encoded_1gram_shannon_fano.txt", "w", encoding="utf-8") as f:
        f.write(enc)
    dec = decode(enc, code_sf)
    with open("decoded_1gram_shannon_fano.txt", "w", encoding="utf-8") as f:
        f.write(dec)

    code_hf = huffman(prob1)
    save_code_csv("code_1gram_huffman.csv", code_hf, prob1)
    Lavg_hf = average_code_length(prob1, code_hf)
    eta_hf = H / Lavg_hf if Lavg_hf>0 else 0.0
    print(f"[1-граммы, Хаффман] Lavg = {Lavg_hf:.6f} ; η = {eta_hf:.6f}")

    enc_hf = encode(text, code_hf)
    with open("encoded_1gram_huffman.txt", "w", encoding="utf-8") as f:
        f.write(enc_hf)
    dec_hf = decode(enc_hf, code_hf)
    with open("decoded_1gram_huffman.txt", "w", encoding="utf-8") as f:
        f.write(dec_hf)

    # Bigrams (no encode/decode due to overlap ambiguity)
    freq2 = build_freq(text, 2)
    prob2 = probs_from_freq(freq2)
    save_alphabet_csv("alphabet_2gram.csv", freq2, prob2)
    code2_sf = shannon_fano(prob2)
    save_code_csv("code_2gram_shannon_fano.csv", code2_sf, prob2)
    H2 = entropy(prob2)
    Lavg2 = average_code_length(prob2, code2_sf)
    print(f"[2-граммы, Шеннон–Фано] H2 = {H2:.6f} ; Lavg2 = {Lavg2:.6f}")

    code2_hf = huffman(prob2)
    save_code_csv("code_2gram_huffman.csv", code2_hf, prob2)
    Lavg2_hf = average_code_length(prob2, code2_hf)
    print(f"[2-граммы, Хаффман] Lavg2 = {Lavg2_hf:.6f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python task1_text_coding.py <input_text_file>")
        sys.exit(1)
    pipeline(sys.argv[1])
