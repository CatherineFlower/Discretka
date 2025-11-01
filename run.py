# -*- coding: utf-8 -*-
import math, random, sys
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE
N = 200_000

def gcd(a,b):
    while b: a,b = b, a%b
    return a

def mod_pow(a,e,m):
    r=1; a%=m
    while e:
        if e&1: r=(r*a)%m
        a=(a*a)%m; e>>=1
    return r

def sieve(n):
    s=[True]*(n+1); s[:2]=[False,False]
    p=2
    while p*p<=n:
        if s[p]:
            s[p*p:n+1:p]=[False]*(((n-p*p)//p)+1)
        p+=1
    return [i for i,v in enumerate(s) if v]

def is_square(x):
    r=int(math.isqrt(x))
    return r*r==x

def fermat_factor(n, max_iter=1_000_000):
    if n%2==0: return 2, n//2
    a=math.isqrt(n)
    if a*a<n: a+=1
    for _ in range(max_iter):
        b2=a*a-n
        if is_square(b2):
            b=int(math.isqrt(b2))
            p,q=a-b,a+b
            if 1<p<n: return p,q
        a+=1
    return None

def fermat_primality(n, t=5):
    if n<2: return False
    for p in (2,3,5,7,11,13,17,19,23,29):
        if n%p==0: return n==p
    for _ in range(t):
        a=random.randrange(2,n-1)
        if mod_pow(a,n-1,n)!=1:
            return False
    return True

def jacobi(a,n):
    if n<=0 or n%2==0: return 0
    a%=n; t=1
    while a:
        while a%2==0:
            a//=2
            r=n%8
            if r in (3,5): t=-t
        a,n = n,a
        if a%4==3 and n%4==3: t=-t
        a%=n
    return t if n==1 else 0

def solovay_strassen(n, t=5):
    if n<2: return False
    if n%2==0: return n==2
    for _ in range(t):
        a=random.randrange(2,n-1)
        if gcd(a,n)!=1: return False
        j=(jacobi(a,n)+n)%n
        p=mod_pow(a,(n-1)//2,n)
        if p!=j: return False
    return True

def lehmann(n, t=5):
    if n<2: return False
    if n%2==0: return n==2
    for _ in range(t):
        a=random.randrange(2,n-1)
        x=mod_pow(a,(n-1)//2,n)
        if x!=1 and x!=n-1:
            return False
    return True

def miller_rabin(n, t=7):
    if n<2: return False
    small=[2,3,5,7,11,13,17,19,23,29]
    for p in small:
        if n%p==0: return n==p
    d=n-1; s=0
    while d%2==0: d//=2; s+=1
    for _ in range(t):
        a=random.randrange(2,n-1)
        x=mod_pow(a,d,n)
        if x in (1,n-1): continue
        for _ in range(s-1):
            x=(x*x)%n
            if x==n-1: break
        else:
            return False
    return True

def is_prime_trial(n):
    if n<2: return False
    if n%2==0: return n==2
    i=3; r=int(math.isqrt(n))
    while i<=r:
        if n%i==0: return False
        i+=2
    return True

def primes_pi(n):
    ps=sieve(n)
    pi=[0]*(n+1); c=0; j=0
    for x in range(2,n+1):
        if j<len(ps) and ps[j]==x: c+=1; j+=1
        pi[x]=c
    return ps,pi

def odd_composites(limit, k=2000, primes_set=None):
    if primes_set is None:
        primes_set=set(sieve(limit))
    out=[]; x=9
    while x<=limit and len(out)<k:
        if x%2==1 and x not in primes_set: out.append(x)
        x+=2
    return out

def plot_save(make, path):
    plt.figure()
    make(plt)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def task_1_table_primes_lt256():
    ps=[p for p in sieve(255)]
    txt="\n".join(map(str,ps))
    (OUT/"pr6_primes_lt256.txt").write_text(txt, encoding="utf-8")
    print("Таблица простых <256: pr6_primes_lt256.txt")

def task_2_fermat_for_n_numbers():
    src=OUT/"pr6_numbers.txt"
    if src.exists():
        nums=[int(s) for s in src.read_text(encoding="utf-8").split()]
    else:
        nums=[91, 221, 341, 1009, 10007, 99991]  # примеры
    lines=[]
    for n in nums:
        primal = fermat_primality(n, t=5)
        if primal:
            lines.append(f"{n}: вероятно простое по Ферма")
        else:
            fac = fermat_factor(n)
            if fac: lines.append(f"{n}: составное, разложение Ферма {fac[0]}*{fac[1]}")
            else:   lines.append(f"{n}: составное, факторизация не найдена методом Ферма")
    (OUT/"pr6_fermat_numbers.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Метод Ферма для n чисел: pr6_fermat_numbers.txt")

def task_3_big_prime_checks():
    src=OUT/"pr6_bigp.txt"
    if src.exists():
        p=int(src.read_text(encoding="utf-8").strip())
    else:
        while True:
            p=random.randrange(1_000_000_000, 1_100_000_000)|1
            if miller_rabin(p, t=8): break
        src.write_text(str(p), encoding="utf-8")
    res=[
        f"p={p}",
        f"Соловей–Штрассен: {'простое' if solovay_strassen(p, t=8) else 'составное'}",
        f"Леманн: {'простое' if lehmann(p, t=8) else 'составное'}",
        f"Миллер–Рабин: {'простое' if miller_rabin(p, t=8) else 'составное'}",
        f"Непосредственная проверка: {'простое' if is_prime_trial(p) else 'составное'}",
    ]
    (OUT/"pr6_bigp_checks.txt").write_text("\n".join(res), encoding="utf-8")
    print("Проверка большого p: pr6_bigp_checks.txt")

def task_4_creative_pi_vs_xlogx_and_gaps():
    ps,pi = primes_pi(N)
    xs=list(range(2,N+1,100))
    pi_x=[pi[x] for x in xs]
    approx=[x/math.log(x) for x in xs]
    plot_save(lambda plt: (plt.plot(xs,pi_x,label="π(x)"),
                           plt.plot(xs,approx,label="x/ln x"),
                           plt.xlabel("x"), plt.ylabel("кол-во простых ≤ x"),
                           plt.title("π(x) vs x/ln x"), plt.legend()),
              OUT/"pi_vs_xlogx.png")
    gaps=[ps[i+1]-ps[i] for i in range(len(ps)-1)]
    plot_save(lambda plt: (plt.hist(gaps, bins=range(1,40), density=True),
                           plt.xlabel("gap"), plt.ylabel("доля"),
                           plt.title("Распределение разрывов между простыми")),
              OUT/"prime_gaps_hist.png")
    maxrun=[]; m=0
    for g in gaps: m=max(m,g); maxrun.append(m)
    plot_save(lambda plt: (plt.plot(ps[1:], maxrun),
                           plt.xlabel("текущий простой p"), plt.ylabel("max gap ≤ p"),
                           plt.title("Максимальный разрыв по мере роста p")),
              OUT/"prime_gaps_running_max.png")
    print("Творческое: pi_vs_xlogx.png, prime_gaps_hist.png, prime_gaps_running_max.png")

def task_5_creative_mr_error():
    limit=200_000
    ps=set(sieve(limit))
    nums=odd_composites(limit, k=2000, primes_set=ps)
    ts=range(1,8)
    rates=[]
    for t in ts:
        bad=sum(1 for n in nums if miller_rabin(n,t))
        rates.append(bad/len(nums))
    plot_save(lambda plt: (plt.plot(list(ts), rates, marker="o"),
                           plt.yscale("log"), plt.xlabel("t"), plt.ylabel("доля ложноположительных"),
                           plt.title("Ошибка Миллера–Рабина (эмпирика)")),
              OUT/"mr_false_rate.png")
    print("Творческое: mr_false_rate.png")

def task_6_creative_jacobi_heatmap():
    ns=[n for n in range(3,200) if n%2==1]; max_a=max(ns)-1
    grid=[]
    for n in ns:
        row=[]
        for a in range(1,max_a+1):
            row.append(jacobi(a,n) if a<n else 0)
        grid.append(row)
    plot_save(lambda plt: (plt.imshow(grid, aspect="auto", origin="lower",
                                      extent=[1,max_a,ns[0],ns[-1]]),
                           plt.xlabel("a"), plt.ylabel("n (нечётные)"),
                           plt.title("Тепловая карта J(a,n)")),
              OUT/"jacobi_heatmap.png")
    print("Творческое: jacobi_heatmap.png")

def write_report():
    md = [
        "# Практика 6: алгоритм Евклида, тесты простоты, генерация простых",
        "## Задание 1 — таблица простых < 256",
        "- Файл: pr6_primes_lt256.txt",
        "## Задание 2 — метод Ферма для n чисел + разложение",
        "- Файл: pr6_fermat_numbers.txt",
        "## Задание 3 — проверка большого простого p",
        "- Файл: pr6_bigp_checks.txt",
        "## Творческое — визуализации",
        "![π(x)](pi_vs_xlogx.png)",
        "![gaps](prime_gaps_hist.png)",
        "![running](prime_gaps_running_max.png)",
        "![mr](mr_false_rate.png)",
        "![jacobi](jacobi_heatmap.png)",
    ]
    (OUT/"pr6_report.md").write_text("\n\n".join(md), encoding="utf-8")
    print("Отчёт: pr6_report.md")

def main():
    task_1_table_primes_lt256()
    task_2_fermat_for_n_numbers()
    task_3_big_prime_checks()
    task_4_creative_pi_vs_xlogx_and_gaps()
    task_5_creative_mr_error()
    task_6_creative_jacobi_heatmap()
    write_report()

if __name__=="__main__":
    main()
