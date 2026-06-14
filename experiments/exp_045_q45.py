"""
UAF v6.0 — EXP 045: Q45 — Количественный закон применимости: ошибка vs цикличность
====================================================================================
ГИПОТЕЗА: точность спектрального закона A0=K0/λmax количественно связана с
          цикличностью сети (Q43/Q44 показали: ацикличные сети ломают закон).
          Если |ошибка| убывает с долей циклов по единому тренду через все
          домены — это закон применимости.

ДАННЫЕ:  53 реальные сети из 6 доменов (MJS20): пищевые (symmetrized),
          торговля, социальные, нейро, метаболизм, язык. N=15–500.
          Мера цикличности = доля 2-циклов (обоюдных связей) в исходной
          направленной матрице. K0=(δ−f)/α_s выведена, не подгоняется.

МЕТОД:   измерить |ошибку| закона и долю 2-циклов для каждой сети.
          Множественная регрессия |err| ~ N + cyc, частные корреляции
          (отделить эффект цикличности от размера сети).

ОТВЕТ:   ГИПОТЕЗА ПОДТВЕРЖДАЕТСЯ ЧАСТИЧНО — цикличность реальна, но не
          доминирует.

          Сырая корреляция cyc<->|err| = −0.51 (больше циклов → меньше ошибка).
          НО размер сети — confound. После множественной регрессии (53 сети):
            размер N:    коэф +1.60, частная корр +0.67 (ГЛАВНЫЙ фактор)
            цикличность: коэф −2.87, частная корр −0.31 (реальный, слабее)
            R² = 0.52

          Примеры:
            торговля (cyc=0.74, N=24): ошибка 0.8% — мало, циклично
            celegans (cyc=0.17, N=297): ошибка 12.7% — много, крупная+ацикл.
            пищевые symmetr. (cyc=0): 4–9% — работают через симметризацию

ВЫВОД:   Цикличность ДЕЙСТВИТЕЛЬНО снижает ошибку закона (частная корр −0.31
          после контроля размера) — это подтверждает механизм Q43/Q44
          (закон требует замкнутости). НО доминирующий фактор — размер сети
          (+0.67): крупные сети дают больший разброс. Единого чистого закона
          "ошибка = f(цикличность)" НЕТ — это многофакторная зависимость.

          Честно: это не "новый закон применимости", а количественное
          подтверждение, что (а) цикличность помогает, (б) размер мешает,
          (в) вместе R²=0.52 — половина дисперсии объяснена, половина нет.

ВЕРДИКТ: ЧАСТИЧНО ПОДТВЕРЖДЕНО. Цикличность — реальный, но не главный
          фактор точности (частная корр −0.31). Размер сети влияет сильнее
          (+0.67). Гипотеза о едином тренде "ошибка~цикличность" не
          подтвердилась в чистом виде — зависимость многофакторная, R²=0.52.
          Это честная граница: связь есть, но простого закона применимости
          из неё не выводится. Согласуется со стилем v5.2 (Q37): реальные
          эффекты часто ограничены и без замкнутой формы.

Запуск:
    PYTHONPATH=. python experiments/exp_045_q45.py
    (требует data/real/mjs_foodwebs/ и data/real/mjs_other/)
"""

import numpy as np
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uaf.core import integrate_network
import scipy.sparse as sp
import scipy.sparse.linalg as sla


AL_S = 0.06
F = 0.002
DELTA = 0.012
K0 = (DELTA - F) / AL_S
FWDIR = 'data/real/mjs_foodwebs'
OTHERDIR = 'data/real/mjs_other'


def load_directed(path):
    E = []
    for line in open(path):
        p = line.split()
        if len(p) >= 2:
            try:
                E.append((int(p[0]), int(p[1])))
            except ValueError:
                pass
    nodes = sorted(set(x for e in E for x in e))
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for a, b in E:
        A[idx[a], idx[b]] = 1
    np.fill_diagonal(A, 0)
    return A


def load_symmetrized(path):
    A = load_directed(path)
    return ((A + A.T) > 0).astype(float)


def cyclicity(A):
    return np.minimum(A, A.T).sum() / max(A.sum(), 1)


def lmax_sym(A):
    S = (A + A.T) / 2
    return float(sla.eigsh(sp.csr_matrix(S.astype(float)), k=1, which='LA',
                           return_eigenvectors=False)[0])


def A0_crit(adj, T=800, dt=0.6, n_bisect=16):
    n = len(adj); lo, hi = 0.001, 0.95
    for _ in range(n_bisect):
        mid = (lo + hi) / 2
        tr = integrate_network(np.full(n, mid), adj, T=T, dt=dt, delta=DELTA)
        if tr[-1].mean() > 0.5:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def giant_component(A):
    n = len(A); seen = np.zeros(n, bool); best = []
    S = ((A + A.T) > 0)
    for s in range(n):
        if seen[s]:
            continue
        st = [s]; c = []
        while st:
            u = st.pop()
            if seen[u]:
                continue
            seen[u] = True; c.append(u); st.extend(np.where(S[u])[0])
        if len(c) > len(best):
            best = c
    b = np.array(best)
    return A[b][:, b]


def partial_corr(x, y, z):
    bx = np.polyfit(z, x, 1); rx = x - np.polyval(bx, z)
    by = np.polyfit(z, y, 1); ry = y - np.polyval(by, z)
    return np.corrcoef(rx, ry)[0, 1]


def collect():
    sets = [(p, 'пища') for p in sorted(glob.glob(f'{FWDIR}/*.dat'))]
    for fn in ['net_trade_basic.dat', 'net_trade_food.dat', 'net_trade_crude.dat',
               'net_social_prison.dat', 'net_moreno_highschool.dat',
               'cat_brain.dat', 'net_celegans_neural.dat', 'rhesus_brain_2.dat',
               'net_CQ.dat', 'net_CT.dat', 'net_green_eggs.dat']:
        p = os.path.join(OTHERDIR, fn)
        if os.path.exists(p):
            sets.append((p, 'др'))
    rows = []
    for path, dom in sets:
        Adir = load_directed(path)
        if len(Adir) < 15 or len(Adir) > 500:
            continue
        c = cyclicity(Adir)
        A = giant_component(load_symmetrized(path))
        a0 = A0_crit(A)
        if a0 > 0.9:
            continue
        err = abs((a0 - K0/lmax_sym(A)) / (K0/lmax_sym(A)) * 100)
        rows.append((len(A), c, err))
    return np.array(rows)


def main():
    print("\n" + "#"*72)
    print("  UAF v6.0 — EXP 045 / Q45: ошибка закона vs цикличность")
    print("  Есть ли количественный закон применимости?")
    print("#"*72)
    np.random.seed(0)
    rows = collect()
    N, c, e = rows.T

    print("\n" + "="*72)
    print(f"EXP 045  Множественная регрессия на {len(rows)} реальных сетях")
    print("="*72)
    X = np.column_stack([np.ones_like(N), N/100, c])
    coef, *_ = np.linalg.lstsq(X, e, rcond=None)
    pred = X @ coef
    r2 = 1 - np.sum((e-pred)**2) / np.sum((e-e.mean())**2)
    print(f"""
  |err| ~ a + b·(N/100) + c·cyc:
    размер N:    коэф {coef[1]:+.2f}, частная корр {partial_corr(N,e,c):+.2f}
    цикличность: коэф {coef[2]:+.2f}, частная корр {partial_corr(c,e,N):+.2f}
    R² = {r2:.2f}

  Сырая корреляция cyc<->|err| = {np.corrcoef(c,e)[0,1]:+.2f}
""")

    print("="*72)
    print("ВЕРДИКТ Q45: ЧАСТИЧНО ПОДТВЕРЖДЕНО")
    print("="*72)
    print(f"""
  ✓ Цикличность снижает ошибку (частная корр {partial_corr(c,e,N):+.2f}) —
    подтверждает механизм Q43/Q44 (закон требует замкнутости)
  ✗ Но размер сети — главный фактор (частная корр {partial_corr(N,e,c):+.2f})
  ~ R²={r2:.2f}: половина дисперсии объяснена, единого чистого закона нет

  Честно: связь "ошибка~цикличность" реальна, но многофакторна. Это не
  новый закон применимости, а количественное подтверждение механизма +
  обнаружение, что размер сети влияет сильнее. Согласуется со стилем v5.2
  (Q37): реальные эффекты часто ограничены, без замкнутой формы.

  СЛЕДУЮЩИЙ ВОПРОС Q46:
    Почему размер увеличивает ошибку? Гипотеза: крупные сети более
    гетерогенны (выше φ=Var(k)/<k²>), а φ-остаток (v5.2 Q37) растёт с
    гетерогенностью. Проверить: |err| ~ φ напрямую. Если φ объясняет
    размерный эффект — это связывает v6.0 (домены) с v5.2 (остаток φ)
    в единую картину.
""")


if __name__ == "__main__":
    main()
