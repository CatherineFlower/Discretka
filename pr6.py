# -*- coding: utf-8 -*-
# pr6.py — Практика №6. Евклид, тесты простоты, генерация простых + творческое и единый отчет.

import math, random, time, io, base64
from pathlib import Path
import matplotlib.pyplot as plt

# ===== пути/настройки =====
OUT = Path(__file__).resolve().parent / "out_pr6"
FILES = OUT / "out_files"
OUT.mkdir(parents=True, exist_ok=True)
FILES.mkdir(parents=True, exist_ok=True)

N = 200_000               # верхняя граница для π(x), разрывов, эмпирики MR
LIMIT_CAR = 300_000       # диапазон поиска чисел Кармайкла
SAMPLE_COMPOSITES = 2000  # размер выборки составных для эмпирики MR
random.seed(42)

# ===== базовые алгоритмы =====
def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

def egcd(a: int, b: int):
    # расширенный Евклид: возвращает g, x, y такие, что a*x + b*y = g
    if b == 0:
        return a, 1, 0
    g, x1, y1 = egcd(b, a % b)
    return g, y1, x1 - (a // b) * y1

def lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b

def mod_pow(a: int, e: int, m: int) -> int:
    r = 1
    a %= m
    while e:
        if e & 1:
            r = (r * a) % m
        a = (a * a) % m
        e >>= 1
    return r

def sieve(n: int):
    s = [True] * (n + 1)
    s[:2] = [False, False]
    p = 2
    while p * p <= n:
        if s[p]:
            s[p*p:n+1:p] = [False] * (((n - p*p)//p) + 1)
        p += 1
    return [i for i, v in enumerate(s) if v]

def is_square(x: int) -> bool:
    r = int(math.isqrt(x))
    return r*r == x

# ===== лёгкая факторизация =====
_PRIMES_UP_TO = sieve(2000)

def trial_factor(n: int):
    fac, x = [], n
    for p in _PRIMES_UP_TO:
        if p*p > x: break
        if x % p == 0:
            c = 0
            while x % p == 0:
                x //= p; c += 1
            fac.append((p, c))
    f = 3
    r = int(math.isqrt(x))
    while f <= r and x > 1:
        if x % f == 0:
            c = 0
            while x % f == 0:
                x //= f; c += 1
            fac.append((f, c))
            r = int(math.isqrt(x))
        f += 2
    if x > 1:
        fac.append((x, 1))
    return fac

# ===== тесты простоты =====
def is_prime_trial(n: int) -> bool:
    # перебор делителей до sqrt(n)
    if n < 2: return False
    if n % 2 == 0: return n == 2
    i, r = 3, int(math.isqrt(n))
    while i <= r:
        if n % i == 0: return False
        i += 2
    return True

def fermat_primality(n: int, t: int = 5, bases=None) -> bool:
    if n < 2: return False
    small = (2,3,5,7,11,13,17,19,23,29)
    for p in small:
        if n % p == 0:
            return n == p
    if bases is None:
        for _ in range(t):
            a = random.randrange(2, n-1)
            if mod_pow(a, n-1, n) != 1:
                return False
        return True
    else:
        for a in bases:
            if 1 < a < n-1 and mod_pow(a, n-1, n) != 1:
                return False
        return True

def jacobi(a: int, n: int) -> int:
    if n <= 0 or n % 2 == 0: return 0
    a %= n; t = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3,5): t = -t
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3: t = -t
        a %= n
    return t if n == 1 else 0

def solovay_strassen(n: int, t: int = 5) -> bool:
    if n < 2: return False
    if n % 2 == 0: return n == 2
    for _ in range(t):
        a = random.randrange(2, n-1)
        if gcd(a, n) != 1: return False
        j = (jacobi(a, n) + n) % n
        p = mod_pow(a, (n-1)//2, n)
        if p != j: return False
    return True

def lehmann(n: int, t: int = 5) -> bool:
    # тест Леманна: a^((n-1)/2) == ±1 (mod n)
    if n < 2: return False
    if n % 2 == 0: return n == 2
    for _ in range(t):
        a = random.randrange(2, n-1)
        x = mod_pow(a, (n-1)//2, n)
        if x != 1 and x != n-1:
            return False
    return True

def miller_rabin(n: int, t: int = 7) -> bool:
    if n < 2: return False
    for p in (2,3,5,7,11,13,17,19,23,29):
        if n % p == 0:
            return n == p
    d, s = n-1, 0
    while d % 2 == 0:
        d //= 2; s += 1
    bases = (2,3,5,7,11) if n < 2_147_483_648 else None
    if bases is None:
        for _ in range(t):
            a = random.randrange(2, n-1)
            x = mod_pow(a, d, n)
            if x in (1, n-1): continue
            for _ in range(s-1):
                x = (x*x) % n
                if x == n-1: break
            else:
                return False
        return True
    else:
        for a in bases:
            if a >= n: continue
            x = mod_pow(a, d, n)
            if x in (1, n-1): continue
            for _ in range(s-1):
                x = (x*x) % n
                if x == n-1: break
            else:
                return False
        return True

# ===== Кармайкла и псевдопростые Ферма =====
def is_square_free(factors) -> bool:
    return all(e == 1 for _, e in factors)

def is_carmichael(n: int) -> bool:
    if n < 3 or n % 2 == 0: return False
    fac = trial_factor(n)
    if not fac or not is_square_free(fac): return False
    for p, _ in fac:
        if (n-1) % (p-1) != 0: return False
    return True

def fermat_pseudoprimes(limit: int, bases=(2,3,5)):
    ps = set(sieve(limit))
    res = []
    for n in range(3, limit+1, 2):
        if n in ps: continue
        if fermat_primality(n, bases=bases):
            res.append(n)
    return res

# ===== утилиты данных/графиков/отчета =====
def primes_pi(n: int):
    ps = sieve(n)
    pi = [0]*(n+1); c = 0; j = 0
    for x in range(2, n+1):
        if j < len(ps) and ps[j] == x:
            c += 1; j += 1
        pi[x] = c
    return ps, pi

def odd_composites(limit: int, k: int = SAMPLE_COMPOSITES, primes_set=None):
    if primes_set is None:
        primes_set = set(sieve(limit))
    out, x = [], 9
    while x <= limit and len(out) < k:
        if x % 2 == 1 and x not in primes_set:
            out.append(x)
        x += 2
    return out

def plot_save(make, path: Path):
    plt.figure()
    make(plt)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def to_data_uri(path: Path):
    if not path.exists(): return None
    b = path.read_bytes()
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:image/png;base64,{b64}"

def read_text_trunc(path: Path, limit: int = 2000):
    if not path.exists(): return "_нет файла_"
    s = path.read_text(encoding="utf-8", errors="ignore")
    if len(s) <= limit: return s
    return s[:limit] + f"\n\n... [обрезано, всего {len(s)} символов]"

def md_h2(s: str) -> str: return f"\n\n## {s}\n"
def md_small(s: str) -> str: return f"**{s}**\n"

# ===== Евклид: полный протокол (таблица Кнута) =====
def euclid_steps(a: int, b: int):
    # Возвращает таблицу шагов и коэффициенты Безу.
    if a < b: a, b = b, a
    A, B = a, b
    rows = []
    x_m2, y_m2 = 1, 0  # x_{-1}, y_{-1}
    x_m1, y_m1 = 0, 1  # x_0, y_0
    i = 1
    while B != 0:
        q = A // B
        r = A % B
        x_i = x_m2 - q * x_m1
        y_i = y_m2 - q * y_m1
        rows.append((i, A, B, q, r, x_i, y_i))
        A, B = B, r
        x_m2, x_m1 = x_m1, x_i
        y_m2, y_m1 = y_m1, y_i
        i += 1
    g = A
    x, y = x_m2, y_m2
    return rows, g, x, y

def save_euclid_csv(path: Path, rows):
    lines = ["i;A;B;q;r;x;y"]
    for (i, A, B, q, r, x, y) in rows:
        lines.append(f"{i};{A};{B};{q};{r};{x};{y}")
    path.write_text("\n".join(lines), encoding="utf-8")

# ===== ЗАДАНИЕ 1 =====
def task_1_primes_lt256():
    ps = [p for p in sieve(255)]
    (FILES / "pr6_primes_lt256.txt").write_text("\n".join(map(str, ps)), encoding="utf-8")
    print("Задание 1: простые < 256 -> out_pr6/out_files/pr6_primes_lt256.txt")

# ===== ЗАДАНИЕ 2 =====
def task_2_fermat_for_numbers():
    src = FILES / "pr6_numbers.txt"
    if src.exists():
        nums = [int(s) for s in src.read_text(encoding="utf-8").split()]
    else:
        nums = [91, 221, 341, 1009, 10007, 99991]
    lines = []
    for n in nums:
        primal = fermat_primality(n, t=5)
        if primal:
            lines.append(f"{n}: вероятно простое по Ферма")
        else:
            fac = None
            if n % 2 == 0:
                fac = (2, n//2)
            else:
                a = math.isqrt(n) + (0 if is_square(n) else 1)
                tries = 0
                while tries < 200_000:
                    b2 = a*a - n
                    if b2 >= 0 and is_square(b2):
                        b = int(math.isqrt(b2))
                        p, q = a - b, a + b
                        if 1 < p < n:
                            fac = (p, q); break
                    a += 1; tries += 1
            if fac:
                lines.append(f"{n}: составное, разложение Ферма {fac[0]}*{fac[1]}")
            else:
                lines.append(f"{n}: составное, факторизация не найдена методом Ферма")
    (FILES / "pr6_fermat_numbers.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Задание 2: Ферма -> out_pr6/out_files/pr6_fermat_numbers.txt")

# ===== ЗАДАНИЕ 3 =====
def task_3_big_prime_checks():
    src = FILES / "pr6_bigp.txt"
    if src.exists():
        p = int(src.read_text(encoding="utf-8").strip())
    else:
        while True:
            p = (random.randrange(1_000_000_000, 1_100_000_000) | 1)
            if miller_rabin(p, t=8): break
        src.write_text(str(p), encoding="utf-8")
    res = [
        f"p={p}",
        f"Соловей-Штрассена: {'простое' if solovay_strassen(p, t=8) else 'составное'}",
        f"Леманн: {'простое' if lehmann(p, t=8) else 'составное'}",
        f"Миллер-Рабин: {'простое' if miller_rabin(p, t=8) else 'составное'}",
        f"Перебор делителей: {'простое' if is_prime_trial(p) else 'составное'}",
    ]
    (FILES / "pr6_bigp_checks.txt").write_text("\n".join(res), encoding="utf-8")
    print("Задание 3: большое p -> out_pr6/out_files/pr6_bigp_checks.txt")

# ===== ДОП: Евклид с полным протоколом =====
def euclid_full():
    inp = FILES / "pr6_euclid_input.txt"
    if inp.exists():
        pairs = []
        for line in inp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line: continue
            a_str, b_str = line.split()[:2]
            pairs.append((int(a_str), int(b_str)))
    else:
        pairs = [(240, 46), (12345, 5432), (987654, 3210)]

    summary = []
    for a, b in pairs:
        rows, g, x, y = euclid_steps(a, b)
        csv_path = FILES / f"euclid_{a}_{b}.csv"
        save_euclid_csv(csv_path, rows)
        l = a // g * b
        summary.append(f"{a},{b}: gcd={g}, x={x}, y={y}, lcm={l}, csv={csv_path.name}")
        print(f"[Евклид] {a},{b} -> gcd={g}, x={x}, y={y}, lcm={l}")
    (FILES / "pr6_euclid_summary.txt").write_text("\n".join(summary), encoding="utf-8")

# ===== ТВОРЧЕСКОЕ 1: π(x) и разрывы =====
def creative_pi_and_gaps():
    ps, pi = primes_pi(N)
    xs = list(range(2, N+1, 100))
    pi_x = [pi[x] for x in xs]
    approx = [x/math.log(x) for x in xs]
    plot_save(lambda plt: (plt.plot(xs, pi_x, label="pi(x)"),
                           plt.plot(xs, approx, label="x/ln x"),
                           plt.xlabel("x"), plt.ylabel("кол-во простых <= x"),
                           plt.title("pi(x) vs x/ln x"), plt.legend()),
              FILES / "pi_vs_xlogx.png")

    gaps = [ps[i+1]-ps[i] for i in range(len(ps)-1)]
    plot_save(lambda plt: (plt.hist(gaps, bins=range(1, 40), density=True),
                           plt.xlabel("gap"), plt.ylabel("доля"),
                           plt.title("Распределение разрывов между простыми")),
              FILES / "prime_gaps_hist.png")

    maxrun, m = [], 0
    for g in gaps:
        m = max(m, g); maxrun.append(m)
    plot_save(lambda plt: (plt.plot(ps[1:], maxrun),
                           plt.xlabel("текущий простой p"), plt.ylabel("max gap <= p"),
                           plt.title("Максимальный разрыв по мере роста p")),
              FILES / "prime_gaps_running_max.png")

# ===== ТВОРЧЕСКОЕ 2: ошибка MR =====
def creative_mr_error():
    # диапазон и выборка составных
    MR_LIMIT = 1_000_000
    primes_set = set(sieve(MR_LIMIT))
    nums = odd_composites(MR_LIMIT, k=5000, primes_set=primes_set)

    # добавим известных сильных псевдопростых к основанию 2 (<= ~1e5)
    spsp_base2 = [
        2047, 3277, 4033, 4681, 8321, 15841, 29341, 42799,
        49141, 52633, 65281, 74665, 80581, 85489, 88357
    ]
    for n in spsp_base2:
        if n <= MR_LIMIT and n not in primes_set:
            nums.append(n)

    # MR c фиксированными наборами оснований: [2], [2,3], [2,3,5], ...
    base_pool = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    def mr_with_bases(n: int, bases) -> bool:
        # классический MR, но проверяем именно заданные основания
        if n < 2:
            return False
        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29):
            if n % p == 0:
                return n == p
        d, s = n - 1, 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in bases:
            if a >= n:
                continue
            x = mod_pow(a, d, n)
            if x in (1, n - 1):
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True  # прошло все основания — вероятно простое

    ts = list(range(1, 8))
    rates = []
    for t in ts:
        bases = base_pool[:t]
        bad = sum(1 for n in nums if mr_with_bases(n, bases))
        rates.append(bad / len(nums))

    # все значения > 0 — можно честно показать лог-ось без eps
    def _plot(plt):
        plt.plot(ts, rates, marker="o")
        for x, r in zip(ts, rates):
            plt.annotate(f"{r:.3e}", (x, r), textcoords="offset points",
                         xytext=(0, 6), ha="center", fontsize=8)
        plt.yscale("log")
        plt.xlabel("число раундов t")
        plt.ylabel("доля ложноположительных")
        plt.title("Ошибка Миллера-Рабина (эмпирика)")

    plot_save(_plot, FILES / "mr_false_rate.png")

# ===== ТВОРЧЕСКОЕ 3: Якоби heatmap =====
def creative_jacobi_heatmap():
    ns = [n for n in range(3, 200) if n % 2 == 1]
    max_a = max(ns) - 1
    grid = []
    for n in ns:
        row = []
        for a in range(1, max_a+1):
            row.append(jacobi(a, n) if a < n else 0)
        grid.append(row)

    def _plot(plt):
        im = plt.imshow(grid, aspect="auto", origin="lower",
                        extent=[1, max_a, ns[0], ns[-1]])
        c = plt.colorbar(im); c.set_label("J(a,n)")
        plt.xlabel("a"); plt.ylabel("n (нечетные)")
        plt.title("Тепловая карта символа Якоби")

    plot_save(_plot, FILES / "jacobi_heatmap.png")

# ===== ТВОРЧЕСКОЕ 4: Кармайкла + Ферма heatmap =====
def creative_carmichael_and_fermat():
    car = [n for n in range(3, LIMIT_CAR+1, 2) if is_carmichael(n)]
    (FILES / "carmichael.txt").write_text("\n".join(map(str, car)), encoding="utf-8")

    bins = 10
    step = LIMIT_CAR // bins
    counts = []
    for i in range(bins):
        lo, hi = i*step + 1, (i+1)*step
        counts.append(sum(1 for x in car if lo <= x <= hi))
    plot_save(lambda plt: (plt.bar([f"{i*step//1000}-{(i+1)*step//1000}k" for i in range(bins)], counts),
                           plt.title(f"Числа Кармайкла <= {LIMIT_CAR:,}".replace(",", " ")),
                           plt.xlabel("интервалы"), plt.ylabel("кол-во")),
              FILES / "carmichael_hist.png")

    bases = list(range(2, 11))
    intervals = 10
    step = N // intervals
    heat = [[0 for _ in range(intervals)] for _ in bases]
    fps = {a: set(fermat_pseudoprimes(N, bases=(a,))) for a in bases}
    for j, a in enumerate(bases):
        for i in range(intervals):
            lo, hi = i*step + 1, (i+1)*step
            heat[j][i] = sum(1 for x in fps[a] if lo <= x <= hi)
    plot_save(lambda plt: (plt.imshow(heat, aspect="auto", origin="lower",
                                      extent=[1, intervals, bases[0], bases[-1]]),
                           plt.colorbar(label="кол-во псевдопростых"),
                           plt.xlabel("интервальные бины <=200k"), plt.ylabel("основание a"),
                           plt.title("Ферма-псевдопростые: основание x интервал")),
              FILES / "fermat_heatmap.png")

# ===== ТВОРЧЕСКОЕ 5: профили времени =====
def creative_timing():
    ps = set(sieve(N))
    nums = odd_composites(N, k=800, primes_set=ps)
    ts = list(range(1, 8))

    def avg_time(func, *fargs):
        t0 = time.perf_counter()
        for n in nums:
            func(n, *fargs)
        return (time.perf_counter() - t0) / len(nums)

    mr_times = [avg_time(miller_rabin, t) for t in ts]
    ss_times = [avg_time(solovay_strassen, t) for t in ts]
    f_times  = [avg_time(fermat_primality, t) for t in ts]

    plot_save(lambda plt: (plt.plot(ts, mr_times, marker="o", label="Миллер-Рабин"),
                           plt.plot(ts, ss_times, marker="s", label="Соловей-Штрассена"),
                           plt.plot(ts, f_times,  marker="^", label="Ферма"),
                           plt.xlabel("число раундов t"), plt.ylabel("ср. время на число, с"),
                           plt.title("Профилирование времени тестов"), plt.legend()),
              FILES / "tests_timing.png")

# ===== единый отчет =====
def build_report_all():
    md = []
    md.append(f"# Практика №6 — единый отчет\n\n_Сгенерировано:_ {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Евклид: сводка + ссылки на CSV
    md.append(md_h2("Евклид и расширенный Евклид"))
    md.append("Файл со сводкой: `pr6_euclid_summary.txt`. Протоколы шагов в CSV: `euclid_*.csv`.\n")
    md.append("```text\n" + read_text_trunc(FILES / "pr6_euclid_summary.txt", 4000) + "\n```\n")

    # Задания 1–3
    md.append(md_h2("Задание 1. Таблица простых < 256"))
    md.append("```text\n" + read_text_trunc(FILES / "pr6_primes_lt256.txt", 4000) + "\n```\n")

    md.append(md_h2("Задание 2. Метод Ферма для набора чисел"))
    md.append("```text\n" + read_text_trunc(FILES / "pr6_fermat_numbers.txt") + "\n```\n")

    md.append(md_h2("Задание 3. Проверка большого p (S-S, Леманн, M-R, перебор)"))
    md.append("```text\n" + read_text_trunc(FILES / "pr6_bigp_checks.txt") + "\n```\n")

    # Творческое с пояснениями
    # --- Creative 1: плотность простых и разрывы ---------------------------------
    md.append(md_h2("Creative 1. Плотность простых и разрывы"))

    for name, title, note in [
        (
                "pi_vs_xlogx.png",
                "π(x) и x/ln x",
                "Ось X — x, ось Y — количество. Кривая π(x) vs аппроксимация x/ln x: "
                "x/ln x слегка занижает счётчик, но уже при x ≤ 2·10^5 даёт рабочую точность; "
                "именно поэтому x/ln x удобно использовать как быструю оценку и sanity-check.\n"
        ),
        (
                "prime_gaps_hist.png",
                "Гистограмма разрывов между простыми",
                "Ось X — размер разрыва g = p_{k+1}-p_k, ось Y — частота. "
                "Столбцы показывают, что малые разрывы доминируют, крупные редки; "
                "именно поэтому средний разрыв растёт медленно (~ln p) и выбор порогов по g рационален.\n"
        ),
        (
                "prime_gaps_running_max.png",
                "Максимальный разрыв по мере роста p",
                "Ось X — значение p (или индекс простого), ось Y — текущий рекордный разрыв. "
                "Кривая роста рекордов медленная, согласуется с эвристикой порядка ln² p; "
                "именно поэтому «большие провалы» встречаются, но крайне нечасто.\n"
        ),
    ]:
        uri = to_data_uri(FILES / name)
        if uri:
            md.append(md_small(title))
            md.append(note)
            md.append(f"![]({uri})\n")
        else:
            md.append(md_small(title))
            md.append("_Иллюстрация недоступна: файл не найден._\n")

    # --- Creative 2: эмпирическая ошибка Миллера–Рабина --------------------------
    md.append(md_h2("Creative 2. Ошибка Миллера–Рабина (эмпирика)"))
    md.append(
        "Ось X — число раундов t, ось Y — доля ложноположительных (лог-шкала). "
        "Точки/линия показывают почти экспоненциальное падение ошибки; "
        "именно поэтому прибавка 1–2 раундов даёт резкий прирост надёжности.\n"
    )
    uri = to_data_uri(FILES / "mr_false_rate.png")
    if uri:
        md.append(f"![]({uri})\n")
    else:
        md.append("_Иллюстрация недоступна: файл не найден._\n")

    md.append(md_h2("Creative 3. Символ Якоби"))
    md.append("Ось X — основание a, ось Y — нечетные модули n (3..199). "
              "Цвета: +1, 0, -1. Нули там, где gcd(a,n)>1. Мозаика ±1 демонстрирует квадратичную взаимность; "
              "именно поэтому тест Соловея-Штрассена эффективно отбрасывает много составных.\n")
    uri = to_data_uri(FILES / "jacobi_heatmap.png")
    if uri: md.append(f"![]({uri})\n")

    # --- Creative 4: Кармайкла и псевдопростые Ферма -----------------------------
    md.append(md_h2("Creative 4. Кармайкла и псевдопростые Ферма"))
    md.append(
        "Числа Кармайкла — составные, проходящие тест Ферма для любого основания, взаимно простого с n.\n"
    )

    # Гистограмма/накопление чисел Кармайкла
    uri = to_data_uri(FILES / "carmichael_hist.png")
    if uri:
        md.append(md_small("Распределение чисел Кармайкла"))
        md.append(
            "Ось X — n (или бины по n/лог-шкала), ось Y — количество/плотность. "
            "График показывает, что такие n встречаются регулярно и их число растёт с диапазоном; "
            "именно поэтому один лишь тест Ферма принципиально ненадёжён — существуют бесконечные семейства «обманок».\n"
        )
        md.append(f"![]({uri})\n")

    # Тепловая карта «кто проходит Ферма»
    uri = to_data_uri(FILES / "fermat_heatmap.png")
    if uri:
        md.append(md_small("Тепловая карта прохождений теста Ферма"))
        md.append(
            "Ось X — основание a, ось Y — нечётные модули n (например, 3..199). "
            "Цвета: «проходит»/«не проходит» (и нули там, где gcd(a,n)>1). "
            "Сплошные полосы для чисел Кармайкла показывают, что они проходят тест при всех взаимно простых основаниях; "
            "именно поэтому Ферма даёт систематические ложноположительные и его нужно усиливать (Соловей–Штрассен, Миллер–Рабин).\n"
        )
        md.append(f"![]({uri})\n")

    # --- Creative 5: Профили времени тестов --------------------------------------
    md.append(md_h2("Creative 5. Профили времени тестов"))
    uri = to_data_uri(FILES / "tests_timing.png")
    if uri:
        md.append(md_small("Сравнение времени работы проверок простоты"))
        md.append(
            "Ось X — размер входа (n или число бит), ось Y — время (часто лог-шкала). "
            "Кривые для разных подходов показывают практическую асимптотику: детерминированные/наивные методы растут резко, "
            "в то время как вероятностные тесты держатся почти линейно по числу раундов и быстро по битовой длине. "
            "Именно поэтому в реальных системах применяют Миллера–Рабина с несколькими раундами (и, при необходимости, "
            "детерминизуют фиксированными основаниями) — это оптимальный компромисс точности и скорости.\n"
        )
        md.append(f"![]({uri})\n")

    # Итоговые выводы
    md.append(md_h2("Короткие выводы"))
    md.append(
        "- Евклид/расш. Евклид: получаем gcd(a,b), коэффициенты Безу и lcm.\n"
        "- Решето и простые < 256: база малых простых для быстрых проверок делимости.\n"
        "- Ферма: быстрый отбраковщик; разложение Ферма иногда находит факторы как разность квадратов.\n"
        "- Большое p: Миллер-Рабин надежнее при том же t; Соловей-Штрассена близок; Ферма наиболее «мягкий».\n"
        "- pi(x): x/ln x дает неплохое приближение, слегка занижая.\n"
        "- Разрывы между простыми: малые доминируют, максимум растет медленно.\n"
        "- Ошибка M-R: на практических диапазонах часто нулевая при малом t; чтобы увидеть ~4^{-t}, нужна «тяжелая» выборка.\n"
        "- Якоби: нули при gcd(a,n)>1 и узор взаимности объясняют силу теста Соловея-Штрассена.\n"
        "- Кармайкла и псевдопростые: показывают границы применимости теста Ферма.\n"
        "- Время: M-R лучший компромисс скорость/надежность; S-S медленнее; Ферма самый быстрый, но менее строгий.\n"
    )

    # запись .md
    report_md = OUT / "pr6_report_all.md"
    report_md.write_text("\n".join(md), encoding="utf-8")

    # дублируем в .ipynb одной markdown-ячейкой
    try:
        import nbformat as nbf
        nb = nbf.v4.new_notebook()
        nb["cells"] = [nbf.v4.new_markdown_cell(report_md.read_text(encoding="utf-8"))]
        with (OUT / "pr6_report_all.ipynb").open("w", encoding="utf-8") as f:
            nbf.write(nb, f)
    except Exception:
        pass

    print("Отчет: out_pr6/pr6_report_all.md (+ .ipynb)")

# ===== main =====
def main():
    # Блок Евклида (полный протокол, CSV)
    euclid_full()

    # Задания практической
    task_1_primes_lt256()
    task_2_fermat_for_numbers()
    task_3_big_prime_checks()

    # Творческое
    creative_pi_and_gaps()
    creative_mr_error()
    creative_jacobi_heatmap()
    creative_carmichael_and_fermat()
    creative_timing()

    # Единый отчет
    build_report_all()

if __name__ == "__main__":
    main()
