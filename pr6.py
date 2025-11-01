# -*- coding: utf-8 -*-
"""
Практика 6: алгоритм Евклида, расширенный Евклид, тесты простоты, генерация простых.
Плюс творческое исследование по простым числам и вероятностным тестам.

Все выходные файлы складываются в каталог out_pr6/.
"""

import math
import random
import time
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------- пути и настройки ----------------------
OUT = Path(__file__).resolve().parent / "out_pr6"
OUT.mkdir(parents=True, exist_ok=True)

# верхние границы для графиков и эмпирики
N = 200_000           # для pi(x), разрывов между простыми, эмпирики Миллера-Рабина
LIMIT_CAR = 300_000   # верхняя граница для поиска чисел Кармайкла
SAMPLE_COMPOSITES = 2000  # размер выборки составных для эмпирики
random.seed(42)       # воспроизводимость

# ---------------------- базовые алгоритмы ----------------------
def gcd(a: int, b: int) -> int:
    """НОД (классический Евклид)."""
    while b:
        a, b = b, a % b
    return a

def egcd(a: int, b: int):
    """Расширенный Евклид: возвращает (g, x, y), где g=НОД(a,b), ax+by=g."""
    if b == 0:
        return a, 1, 0
    g, x1, y1 = egcd(b, a % b)
    return g, y1, x1 - (a // b) * y1

def lcm(a: int, b: int) -> int:
    """НОК через НОД."""
    return a // gcd(a, b) * b

def mod_pow(a: int, e: int, m: int) -> int:
    """Быстрое возведение в степень по модулю."""
    r = 1
    a %= m
    while e:
        if e & 1:
            r = (r * a) % m
        a = (a * a) % m
        e >>= 1
    return r

def sieve(n: int):
    """Решето Эратосфена: список простых <= n."""
    s = [True] * (n + 1)
    s[:2] = [False, False]
    p = 2
    while p * p <= n:
        if s[p]:
            s[p * p:n + 1:p] = [False] * (((n - p * p) // p) + 1)
        p += 1
    return [i for i, v in enumerate(s) if v]

def is_square(x: int) -> bool:
    """Проверка: x является точным квадратом."""
    r = int(math.isqrt(x))
    return r * r == x

# ---------------------- легкая факторизация ----------------------
_PRIMES_UP_TO = sieve(2000)

def trial_factor(n: int):
    """Грубая факторизация делителями до sqrt(n). Возвращает список (p, e)."""
    fac = []
    x = n
    for p in _PRIMES_UP_TO:
        if p * p > x:
            break
        if x % p == 0:
            c = 0
            while x % p == 0:
                x //= p
                c += 1
            fac.append((p, c))
    f = 3
    r = int(math.isqrt(x))
    while f <= r and x > 1:
        if x % f == 0:
            c = 0
            while x % f == 0:
                x //= f
                c += 1
            fac.append((f, c))
            r = int(math.isqrt(x))
        f += 2
    if x > 1:
        fac.append((x, 1))
    return fac

# ---------------------- тесты простоты ----------------------
def is_prime_trial(n: int) -> bool:
    """Непосредственная проверка простоты: делители до sqrt(n)."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    r = int(math.isqrt(n))
    while i <= r:
        if n % i == 0:
            return False
        i += 2
    return True

def fermat_primality(n: int, t: int = 5, bases=None) -> bool:
    """Вероятностный тест Ферма (t случайных оснований или фиксированные bases)."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29):
        if n % p == 0:
            return n == p
    if bases is None:
        for _ in range(t):
            a = random.randrange(2, n - 1)
            if mod_pow(a, n - 1, n) != 1:
                return False
        return True
    else:
        for a in bases:
            if not (1 < a < n - 1):
                continue
            if mod_pow(a, n - 1, n) != 1:
                return False
        return True

def jacobi(a: int, n: int) -> int:
    """Символ Якоби J(a,n) для нечетного n > 0. Возвращает -1, 0 или 1."""
    if n <= 0 or n % 2 == 0:
        return 0
    a %= n
    t = 1
    while a:
        while a % 2 == 0:
            a //= 2
            r = n % 8
            if r in (3, 5):
                t = -t
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            t = -t
        a %= n
    return t if n == 1 else 0

def solovay_strassen(n: int, t: int = 5) -> bool:
    """Тест Соловея-Штрассена."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for _ in range(t):
        a = random.randrange(2, n - 1)
        if gcd(a, n) != 1:
            return False
        j = (jacobi(a, n) + n) % n
        p = mod_pow(a, (n - 1) // 2, n)
        if p != j:
            return False
    return True

def lehmann(n: int, t: int = 5) -> bool:
    """Тест Леманна: a^((n-1)/2) == +-1 (mod n)."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for _ in range(t):
        a = random.randrange(2, n - 1)
        x = mod_pow(a, (n - 1) // 2, n)
        if x != 1 and x != n - 1:
            return False
    return True

def miller_rabin(n: int, t: int = 7) -> bool:
    """Тест Миллера-Рабина (случайные основания, либо детерминированные для 32-бит)."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29):
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # небольшой хак: для 32-бит можно взять фиксированные базы
    bases = (2, 3, 5, 7, 11) if n < 2_147_483_648 else None
    if bases is None:
        for _ in range(t):
            a = random.randrange(2, n - 1)
            x = mod_pow(a, d, n)
            if x in (1, n - 1):
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    else:
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
        return True

# ---------------------- Кармайкла и псевдопростые Ферма ----------------------
def is_square_free(factors) -> bool:
    """Проверка на свободность от квадратов по разложению factors."""
    return all(e == 1 for _, e in factors)

def is_carmichael(n: int) -> bool:
    """Проверка критерия Кормика: n составное, без квадратов в разложении, и (p-1)|(n-1) для всех простых p|n."""
    if n < 3 or n % 2 == 0:
        return False
    fac = trial_factor(n)
    if not fac or not is_square_free(fac):
        return False
    for p, _ in fac:
        if (n - 1) % (p - 1) != 0:
            return False
    return True

def fermat_pseudoprimes(limit: int, bases=(2, 3, 5)):
    """Список ферма-псевдопростых до limit для указанных оснований."""
    ps = set(sieve(limit))
    res = []
    for n in range(3, limit + 1, 2):
        if n in ps:
            continue
        if fermat_primality(n, bases=bases):
            res.append(n)
    return res

# ---------------------- вспомогательные данные и графики ----------------------
def primes_pi(n: int):
    """Возвращает (простые до n, массив pi[x] = кол-во простых <= x)."""
    ps = sieve(n)
    pi = [0] * (n + 1)
    c = 0
    j = 0
    for x in range(2, n + 1):
        if j < len(ps) and ps[j] == x:
            c += 1
            j += 1
        pi[x] = c
    return ps, pi

def odd_composites(limit: int, k: int = SAMPLE_COMPOSITES, primes_set=None):
    """k нечетных составных чисел до limit (для эмпирики)."""
    if primes_set is None:
        primes_set = set(sieve(limit))
    out = []
    x = 9
    while x <= limit and len(out) < k:
        if x % 2 == 1 and x not in primes_set:
            out.append(x)
        x += 2
    return out

def plot_save(make, path: Path):
    """Сохранить график, созданный функцией make(plt)."""
    plt.figure()
    make(plt)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

# ---------------------- расширенный Евклид: полный протокол ----------------------
def euclid_steps(a: int, b: int):
    """
    Полная таблица шагов (по Кнуту).
    Возвращает:
      rows: список кортежей (i, A, B, q, r, x, y)
      g, x, y: НОД и коэффициенты Безу: a*x + b*y = g
    Рекурренты:
      x_i = x_{i-2} - q_i * x_{i-1}
      y_i = y_{i-2} - q_i * y_{i-1}
    Старт: (x_{-1}, y_{-1})=(1,0), (x_0, y_0)=(0,1)
    """
    if a < b:
        a, b = b, a
    A, B = a, b
    rows = []
    x_prev2, y_prev2 = 1, 0
    x_prev1, y_prev1 = 0, 1
    i = 1
    while B != 0:
        q = A // B
        r = A % B
        x_i = x_prev2 - q * x_prev1
        y_i = y_prev2 - q * y_prev1
        rows.append((i, A, B, q, r, x_i, y_i))
        A, B = B, r
        x_prev2, x_prev1 = x_prev1, x_i
        y_prev2, y_prev1 = y_prev1, y_i
        i += 1
    g = A
    x, y = x_prev2, y_prev2
    return rows, g, x, y

def save_euclid_csv(path: Path, rows):
    """Сохранить шаги в CSV с разделителем ';'."""
    lines = ["i;A;B;q;r;x;y"]
    for (i, A, B, q, r, x, y) in rows:
        lines.append(f"{i};{A};{B};{q};{r};{x};{y}")
    path.write_text("\n".join(lines), encoding="utf-8")

# ====================== ЗАДАНИЯ ПРАКТИЧЕСКОЙ ======================

def task1_euclid_block():
    """
    Задание 1. Евклид и расширенный Евклид.
    Берем пары из out_pr6/pr6_euclid_input.txt (если нет — используем дефолт).
    Для каждой пары формируем CSV с таблицей шагов и сводку pr6_euclid.md.
    """
    inp = OUT / "pr6_euclid_input.txt"
    if inp.exists():
        pairs = []
        for line in inp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            a_str, b_str = line.split()[:2]
            pairs.append((int(a_str), int(b_str)))
    else:
        pairs = [(240, 46), (12345, 5432), (987654, 3210)]

    summary = [
        "# Задание 1: Евклид и расширенный Евклид (полный протокол)",
        "",
        "Входные пары: out_pr6/pr6_euclid_input.txt (если файла нет, используются пары по умолчанию).",
        "Для каждой пары сохраняется CSV: колонки i;A;B;q;r;x;y.",
        "",
    ]

    for a, b in pairs:
        rows, g, x, y = euclid_steps(a, b)
        csv_path = OUT / f"euclid_{a}_{b}.csv"
        save_euclid_csv(csv_path, rows)
        l = a // g * b
        summary += [
            f"## Пара a={a}, b={b}",
            f"- НОД(a,b) = {g}",
            f"- Безу: {a}*({x}) + {b}*({y}) = {a*x + b*y}",
            f"- НОК(a,b) = {l}",
            f"- Таблица: {csv_path.name}",
            ""
        ]
        print(f"[Task1] a={a}, b={b} -> gcd={g}, x={x}, y={y}, lcm={l} -> {csv_path.name}")

    (OUT / "pr6_euclid.md").write_text("\n".join(summary), encoding="utf-8")
    print("Task1 report: out_pr6/pr6_euclid.md")

def task2_primes_lt_256():
    """
    Задание 2. Таблица простых < 256 с помощью решета Эратосфена.
    """
    ps = [p for p in sieve(255)]
    (OUT / "pr6_primes_lt256.txt").write_text("\n".join(map(str, ps)), encoding="utf-8")
    print("Task2: таблица простых <256 -> out_pr6/pr6_primes_lt256.txt")

def task3_fermat_and_big_prime_checks():
    """
    Задание 3. Метод Ферма для набора чисел + проверка большого p разными тестами.
    1) Читаем числа из out_pr6/pr6_numbers.txt (если нет, используем дефолт).
       Для каждого числа выполняем тест Ферма; для составных пытаемся факторизовать
       (простая факторизация + попытка Ферма).
       Результат -> out_pr6/pr6_fermat_numbers.txt
    2) Генерируем или читаем большое вероятно простое p из out_pr6/pr6_bigp.txt.
       Проверяем p тестами Соловея-Штрассена, Леманна, Миллера-Рабина и перебором.
       Результат -> out_pr6/pr6_bigp_checks.txt
    """
    # 3.1 Метод Ферма для набора чисел
    src = OUT / "pr6_numbers.txt"
    if src.exists():
        nums = [int(s) for s in src.read_text(encoding="utf-8").split()]
    else:
        nums = [91, 221, 341, 1009, 10007, 99991]  # примеры: составные и простые
    lines = []
    for n in nums:
        probable_prime = fermat_primality(n, t=5)
        if probable_prime:
            lines.append(f"{n}: вероятно простое по Ферма")
        else:
            fac = None
            # быстрая попытка простого разложения
            tf = trial_factor(n)
            if tf:
                # восстановим разложение вида p*q при двух простых сомножителях
                primes_only = [p for p, e in tf for _ in range(e)]
                if len(primes_only) == 2:
                    fac = (primes_only[0], primes_only[1])
            # попытка факторизации Ферма (подходит для не слишком далеких факторов)
            if fac is None and n % 2 == 1:
                a = math.isqrt(n)
                if a * a < n:
                    a += 1
                tries = 0
                while tries < 200_000:
                    b2 = a * a - n
                    if b2 >= 0 and is_square(b2):
                        b = int(math.isqrt(b2))
                        p = a - b
                        q = a + b
                        if 1 < p < n and p * q == n:
                            fac = (p, q)
                            break
                    a += 1
                    tries += 1
            if fac:
                lines.append(f"{n}: составное, разложение {fac[0]}*{fac[1]}")
            else:
                lines.append(f"{n}: составное, разложение не найдено")
    (OUT / "pr6_fermat_numbers.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Task3: метод Ферма для набора чисел -> out_pr6/pr6_fermat_numbers.txt")

    # 3.2 Проверка большого p
    src_p = OUT / "pr6_bigp.txt"
    if src_p.exists():
        p = int(src_p.read_text(encoding="utf-8").strip())
    else:
        while True:
            p = (random.randrange(1_000_000_000, 1_100_000_000) | 1)
            if miller_rabin(p, t=8):
                break
        src_p.write_text(str(p), encoding="utf-8")

    res = [
        f"p={p}",
        f"Соловей-Штрассен: {'простое' if solovay_strassen(p, t=8) else 'составное'}",
        f"Леманн: {'простое' if lehmann(p, t=8) else 'составное'}",
        f"Миллер-Рабин: {'простое' if miller_rabin(p, t=8) else 'составное'}",
        f"Перебор делителей: {'простое' if is_prime_trial(p) else 'составное'}",
    ]
    (OUT / "pr6_bigp_checks.txt").write_text("\n".join(res), encoding="utf-8")
    print("Task3: проверка большого p -> out_pr6/pr6_bigp_checks.txt")

# ====================== ТВОРЧЕСКОЕ ЗАДАНИЕ ======================

def creative1_pi_vs_xlogx_and_gaps():
    """
    Блок 1. Плотность простых: сравнение pi(x) и x/ln x.
    Плюс разрывы между соседними простыми и рост максимального разрыва.
    """
    ps, pi = primes_pi(N)
    xs = list(range(2, N + 1, 100))
    pi_x = [pi[x] for x in xs]
    approx = [x / math.log(x) for x in xs]

    # pi(x) vs x/ln x
    plot_save(
        lambda plt: (
            plt.plot(xs, pi_x, label="pi(x)"),
            plt.plot(xs, approx, label="x/ln x"),
            plt.xlabel("x"),
            plt.ylabel("кол-во простых <= x"),
            plt.title("Сравнение pi(x) и x/ln x"),
            plt.legend(),
        ),
        OUT / "pi_vs_xlogx.png",
    )

    # распределение разрывов
    gaps = [ps[i + 1] - ps[i] for i in range(len(ps) - 1)]
    plot_save(
        lambda plt: (
            plt.hist(gaps, bins=range(1, 40), density=True),
            plt.xlabel("разрыв"),
            plt.ylabel("доля"),
            plt.title("Распределение разрывов между простыми"),
        ),
        OUT / "prime_gaps_hist.png",
    )

    # бегущий максимум разрыва
    maxrun = []
    m = 0
    for g in gaps:
        m = max(m, g)
        maxrun.append(m)
    plot_save(
        lambda plt: (
            plt.plot(ps[1:], maxrun),
            plt.xlabel("текущий простой p"),
            plt.ylabel("макс. разрыв <= p"),
            plt.title("Рост максимального разрыва между простыми"),
        ),
        OUT / "prime_gaps_running_max.png",
    )
    print("Creative1: pi_vs_xlogx.png, prime_gaps_hist.png, prime_gaps_running_max.png")

def creative2_mr_error():
    """
    Блок 2. Эмпирическая ошибка теста Миллера-Рабина.
    Для ~2000 нечетных составных до N считаем долю ложноположительных при t=1..7.
    """
    ps = set(sieve(N))
    nums = odd_composites(N, k=SAMPLE_COMPOSITES, primes_set=ps)
    ts = range(1, 8)
    rates = []
    for t in ts:
        bad = sum(1 for n in nums if miller_rabin(n, t))
        rates.append(bad / len(nums))
    plot_save(
        lambda plt: (
            plt.plot(list(ts), rates, marker="o"),
            plt.yscale("log"),
            plt.xlabel("число раундов t"),
            plt.ylabel("доля ложноположительных"),
            plt.title("Ошибка Миллера-Рабина (эмпирика)"),
        ),
        OUT / "mr_false_rate.png",
    )
    print("Creative2: mr_false_rate.png")

def creative3_jacobi_heatmap():
    """
    Блок 3. Тепловая карта символа Якоби J(a,n) для нечетных n в [3,199] и a в [1,198].
    """
    ns = [n for n in range(3, 200) if n % 2 == 1]
    max_a = max(ns) - 1
    grid = []
    for n in ns:
        row = []
        for a in range(1, max_a + 1):
            row.append(jacobi(a, n) if a < n else 0)
        grid.append(row)
    def _plot(plt_):
        im = plt_.imshow(grid, aspect="auto", origin="lower",
                         extent=[1, max_a, ns[0], ns[-1]])
        plt_.colorbar(im, label="J(a,n)")
        plt_.xlabel("a")
        plt_.ylabel("n (нечетные)")
        plt_.title("Тепловая карта символа Якоби")
    plot_save(_plot, OUT / "jacobi_heatmap.png")
    print("Creative3: jacobi_heatmap.png")

def creative4_carmichael_and_fermat():
    """
    Блок 4. Числа Кармайкла до LIMIT_CAR и ферма-псевдопростые до N.
    """
    # Кармайкла
    car = [n for n in range(3, LIMIT_CAR + 1, 2) if is_carmichael(n)]
    (OUT / "carmichael.txt").write_text("\n".join(map(str, car)), encoding="utf-8")

    bins = 10
    step = LIMIT_CAR // bins
    counts = []
    labels = []
    for i in range(bins):
        lo, hi = i * step + 1, (i + 1) * step
        labels.append(f"{lo//1000}-{hi//1000}k")
        counts.append(sum(1 for x in car if lo <= x <= hi))
    plot_save(
        lambda plt: (
            plt.bar(labels, counts),
            plt.title(f"Числа Кармайкла до {LIMIT_CAR}"),
            plt.xlabel("интервалы"),
            plt.ylabel("количество"),
        ),
        OUT / "carmichael_hist.png",
    )

    # ферма-псевдопростые: тепловая карта основание vs интервал
    bases = list(range(2, 11))
    intervals = 10
    step = N // intervals
    heat = [[0 for _ in range(intervals)] for _ in bases]
    fps = {a: set(fermat_pseudoprimes(N, bases=(a,))) for a in bases}
    for j, a in enumerate(bases):
        for i in range(intervals):
            lo, hi = i * step + 1, (i + 1) * step
            heat[j][i] = sum(1 for x in fps[a] if lo <= x <= hi)
    def _plot2(plt_):
        im = plt_.imshow(heat, aspect="auto", origin="lower",
                         extent=[1, intervals, bases[0], bases[-1]])
        plt_.colorbar(im, label="кол-во псевдопростых")
        plt_.xlabel("интервалы до 200k")
        plt_.ylabel("основание a")
        plt_.title("Ферма-псевдопростые: основание vs интервал")
    plot_save(_plot2, OUT / "fermat_heatmap.png")
    print("Creative4: carmichael_hist.png, fermat_heatmap.png")

def creative5_timing_profile():
    """
    Блок 5. Профилирование времени тестов: Ферма, Соловей-Штрассена, Миллер-Рабин.
    """
    ps = set(sieve(N))
    nums = odd_composites(N, k=800, primes_set=ps)
    ts = range(1, 8)

    def avg_time(func, *fargs):
        t0 = time.perf_counter()
        for n in nums:
            func(n, *fargs)
        return (time.perf_counter() - t0) / max(1, len(nums))

    mr_times = [avg_time(miller_rabin, t) for t in ts]
    ss_times = [avg_time(solovay_strassen, t) for t in ts]
    f_times  = [avg_time(fermat_primality, t) for t in ts]

    plot_save(
        lambda plt: (
            plt.plot(list(ts), mr_times, marker="o", label="Миллер-Рабин"),
            plt.plot(list(ts), ss_times, marker="s", label="Соловей-Штрассена"),
            plt.plot(list(ts), f_times, marker="^", label="Ферма"),
            plt.xlabel("число раундов t"),
            plt.ylabel("среднее время, с"),
            plt.title("Сравнение времени работы тестов"),
            plt.legend(),
        ),
        OUT / "tests_timing.png",
    )
    print("Creative5: tests_timing.png")

# ---------------------- отчет ----------------------
def write_report():
    """
    Короткий отчет в Markdown с разбиением на задания и творческие блоки.
    """
    md = [
        "# Практика 6: Евклид, простые и вероятностные тесты",
        "",
        "## Задание 1. Евклид и расширенный Евклид",
        "- Сводка: pr6_euclid.md",
        "- CSV с протоколом шагов: euclid_*.csv",
        "",
        "## Задание 2. Таблица простых чисел < 256",
        "- Файл: pr6_primes_lt256.txt",
        "",
        "## Задание 3. Метод Ферма и проверка большого p",
        "- Файл с результатами по набору чисел: pr6_fermat_numbers.txt",
        "- Проверка большого p: pr6_bigp_checks.txt",
        "",
        "## Творческое исследование",
        "### Creative 1. Плотность простых и разрывы",
        "График pi(x) и x/ln x, распределение разрывов и рост максимального разрыва.",
        "![pi(x) vs x/ln x](pi_vs_xlogx.png)",
        "![Распределение разрывов](prime_gaps_hist.png)",
        "![Максимальный разрыв](prime_gaps_running_max.png)",
        "",
        "### Creative 2. Ошибка Миллера-Рабина (эмпирика)",
        "Доля ложноположительных для t=1..7 (ось Y логарифмическая).",
        "![MR error](mr_false_rate.png)",
        "",
        "### Creative 3. Символ Якоби",
        "Тепловая карта J(a,n) для нечетных n в [3,199] и a в [1,198].",
        "![Jacobi heatmap](jacobi_heatmap.png)",
        "",
        "### Creative 4. Кармайкла и ферма-псевдопростые",
        "Гистограмма чисел Кармайкла до 300k и тепловая карта ферма-псевдопростых.",
        "![Carmichael](carmichael_hist.png)",
        "![Fermat heatmap](fermat_heatmap.png)",
        "",
        "### Creative 5. Время работы тестов",
        "Сравнение средней длительности для Ферма, Соловея-Штрассена и Миллера-Рабина.",
        "![Timing](tests_timing.png)",
        "",
        "## Дополнительно",
        "- Перечень чисел Кармайкла: carmichael.txt",
    ]
    (OUT / "pr6_report.md").write_text("\n\n".join(md), encoding="utf-8")
    print("Отчет: out_pr6/pr6_report.md")

# ---------------------- точка входа ----------------------
def main():
    # задания практической
    task1_euclid_block()
    task2_primes_lt_256()
    task3_fermat_and_big_prime_checks()

    # творческая часть (5 независимых блоков)
    creative1_pi_vs_xlogx_and_gaps()
    creative2_mr_error()
    creative3_jacobi_heatmap()
    creative4_carmichael_and_fermat()
    creative5_timing_profile()

    # общий отчет
    write_report()

if __name__ == "__main__":
    main()
