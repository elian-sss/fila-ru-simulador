"""
SIMULADOR DE DESEMPENHO DA FILA DO RU - UFOPA (jantar, 17h20-18h20, dia atipico)
Avaliacao de Desempenho (PC010029)

Entradas (mesma pasta):
    planilha_minuto_a_minuto_ru.csv   minuto;chegaram;sairam;no_sistema;fase
    tempos_atendimentos_ru.csv        Volta;Tempo_Atendimento_s

Modelo: cadeia de Markov de tempo discreto com estado (fase, n, etapa de servico).
  - fase  : C (calma) ou P (pico), modelo de Gilbert (p = C->P, r = P->C, por minuto)
  - n     : alunos no sistema (fila + quem assina), capacidade K
  - etapa : o tempo de assinatura e quase constante (CV ~ 0,2), entao a assinatura
            e dividida em k etapas exponenciais (Erlang-k). Com k = 1 volta ao
            modelo geometrico da aula (servico exponencial).

Uso:
    python simulador_ru_jantar.py            # demo (1 jantar) + verificacao (pi) + validacao + cenarios
    python simulador_ru_jantar.py --rapido   # 100 jantares em vez de 500
    python simulador_ru_jantar.py --sem-navegador   # gera o painel sem abrir o navegador

Ao final, gera painel_ru.html (pagina animada com todos os resultados) e abre no navegador.
Precisa de painel_ru.py e painel_template.html na mesma pasta.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEMENTE = 42
DURACAO_MIN = 60          # 17h20 -> 18h20
N_INICIAL = 28            # alunos na fila as 17h20
K_MAX = 60                # capacidade fisica do corredor
R_JANTARES = 100 if "--rapido" in sys.argv else 500

# ------------------------------------------------------------------ calibracao
plan = pd.read_csv("planilha_minuto_a_minuto_ru.csv", sep=";")
tempos = pd.read_csv("tempos_atendimentos_ru.csv", sep=";")

cheg = plan["chegaram"].to_numpy()
fase = plan["fase"].tolist()
LAM_MEDIA = cheg.mean()
LAM_C = cheg[plan["fase"] == "C"].mean()
LAM_P = cheg[plan["fase"] == "P"].mean()

# estimacao de P (Aula 09): p_ij = n_ij / n_i
pares = list(zip(fase[:-1], fase[1:]))
n_CP, n_CC = pares.count(("C", "P")), pares.count(("C", "C"))
n_PC, n_PP = pares.count(("P", "C")), pares.count(("P", "P"))
P_CP = n_CP / (n_CP + n_CC)         # por minuto
P_PC = n_PC / (n_PC + n_PP)         # por minuto

N_CRONOMETRADOS = 40
medidos = tempos["Tempo_Atendimento_s"].to_numpy()[1:N_CRONOMETRADOS]
S_MEDIO = medidos.mean()                          # ~26,17 s
CV = medidos.std(ddof=1) / S_MEDIO                # ~0,20
MU = 60.0 / S_MEDIO                               # alunos/min por atendente
K_ERLANG = int(round(1 / CV**2))                  # ~25 etapas

# partida fria: a volta 1 (55,31 s) foi ~29 s mais lenta que a media; so o 1o aluno
PARTIDA_FRIA_S = tempos["Tempo_Atendimento_s"].iloc[0] - S_MEDIO

MU_QR = 15.0                                      # HIPOTESE: 4 s por aluno com leitor de QR


# ------------------------------------------------------------------ simulador
def simular(R=1, cenario="atual", com_fases=True, k=K_ERLANG, dt_s=0.25, rng=None,
            mu=MU, duracao=DURACAO_MIN, n0=N_INICIAL, chegadas_medidas=None):
    """Simula R jantares em paralelo. Retorna dict com L, W, D, variancia das saidas, n(t)."""
    rng = rng or np.random.default_rng(SEMENTE)
    dt = dt_s / 60.0                            # passo em minutos
    passos = int(round(duracao / dt))
    por_min = int(round(1 / dt))

    if cenario == "qr":
        mu, k = MU_QR, 5                        # leitor de QR: mais rapido e mais variavel (hipotese)

    q = k * mu * dt                             # chance de avancar uma etapa no passo
    a_max = LAM_P * dt
    assert a_max + 2 * q <= 1, "passo grande demais: diminua dt"

    n = np.full(R, n0)
    pico = np.zeros(R, bool)
    e1 = np.full(R, -1)       # etapa do atendente 1 (-1 = livre); o 1o aluno comeca apos a partida fria
    e2 = np.full(R, -1)       # etapa do atendente 2

    agenda = None
    if chegadas_medidas is not None:            # repete as chegadas medidas, minuto a minuto
        agenda = np.zeros((R, passos), bool)
        for mi in range(duracao):
            sorteio = rng.random((R, por_min))
            ordem = sorteio.argsort(axis=1).argsort(axis=1)
            agenda[:, mi * por_min:(mi + 1) * por_min] = ordem < chegadas_medidas[mi]

    arr = np.zeros((R, duracao))
    sai = np.zeros((R, duracao))
    n_min = np.zeros((R, duracao))

    for t in range(passos):
        m = t // por_min

        # troca de fase (Gilbert)
        if com_fases:
            u = rng.random(R)
            pico = np.where(pico, u >= P_PC * dt, u < P_CP * dt)
            lam = np.where(pico, LAM_P, LAM_C)
        else:
            lam = np.full(R, LAM_MEDIA)

        # chegada
        if agenda is not None:
            cheg_ = agenda[:, t] & (n < K_MAX)
        else:
            cheg_ = (rng.random(R) < lam * dt) & (n < K_MAX)
        n += cheg_
        arr[:, m] += cheg_

        # andamento da assinatura nos atendentes
        for j in (1, 2):
            e = e1 if j == 1 else e2
            ativo = e >= 0
            avanca = ativo & (rng.random(R) < q)
            e += avanca
            fim = avanca & (e == k)
            e[fim] = -1
            n -= fim
            sai[:, m] += fim

        # inicio de atendimento: quem esta na fila ocupa atendente livre
        espera = n - (e1 >= 0) - (e2 >= 0)
        livre1 = (e1 < 0) & (espera > 0) & (t * dt_s >= (PARTIDA_FRIA_S if t * dt_s < 120 else 0))
        e1[livre1] = 0
        espera -= livre1

        if cenario == "2_sempre":
            pode2 = np.ones(R, bool)
        elif cenario == "2_pico":
            pode2 = pico
        else:
            pode2 = np.zeros(R, bool)
        livre2 = (e2 < 0) & (espera > 0) & pode2
        e2[livre2] = 0

        if (t + 1) % por_min == 0:
            n_min[:, m] = n

    L = n_min.mean(axis=1)
    lam_ef = arr.sum(axis=1) / duracao
    W = L / np.where(lam_ef > 0, lam_ef, np.nan)
    D = arr.var(axis=1, ddof=1) / np.where(arr.mean(axis=1) > 0, arr.mean(axis=1), np.nan)
    return dict(L=L, W=W, D=D, var_saidas=sai.var(axis=1, ddof=1),
                saidas=sai.sum(axis=1), n_min=n_min, pico=pico)


def faixa(v):
    return np.nanpercentile(v, 5), np.nanpercentile(v, 95)


# ------------------------------------------------------------------ verificacao (simulador x pi)
def distribuicao_pi(k=K_ERLANG, dt_s=0.25, K=K_MAX):
    """Distribuicao estacionaria pi da cadeia (fase, n, etapa), cenario atual (1 atendente).

    Monta a matriz de transicao de UM passo seguindo a mesma ordem do simular():
    troca de fase -> chegada -> andamento da assinatura -> inicio de atendimento.
    Resolve pi = pi P com soma 1. Devolve P(n) para n = 0..K e a fracao do tempo em pico.
    """
    from scipy.sparse import lil_matrix, identity
    from scipy.sparse.linalg import spsolve
    dt = dt_s / 60.0
    q = k * MU * dt
    bloco = 1 + K * k                       # estados por fase: n = 0 (livre) e n >= 1 com etapa 0..k-1

    def idx(f, n, e):
        return f * bloco + (0 if n == 0 else 1 + (n - 1) * k + e)

    N = 2 * bloco
    P = lil_matrix((N, N))
    for f in (0, 1):
        trocas = ([(1, P_CP * dt), (0, 1 - P_CP * dt)] if f == 0
                  else [(0, P_PC * dt), (1, 1 - P_PC * dt)])
        for n in range(K + 1):
            for e in ([-1] if n == 0 else range(k)):
                i = idx(f, n, e)
                for f2, pf in trocas:
                    a = (LAM_P if f2 else LAM_C) * dt if n < K else 0.0
                    for chega, pa in ((1, a), (0, 1 - a)):
                        if pa == 0:
                            continue
                        for e2, ps in ([(e + 1, q), (e, 1 - q)] if e >= 0 else [(e, 1.0)]):
                            n3, e3 = n + chega, e2
                            if e3 == k:             # terminou a assinatura
                                n3, e3 = n3 - 1, -1
                            if e3 < 0 and n3 > 0:   # proximo da fila comeca
                                e3 = 0
                            P[i, idx(f2, n3, e3)] += pf * pa * ps
    A = (P.T.tocsr() - identity(N, format="csr")).tolil()
    A[0, :] = np.ones(N)                    # troca uma equacao por "soma de pi = 1"
    b = np.zeros(N)
    b[0] = 1.0
    pi = spsolve(A.tocsr(), b)
    pn = np.zeros(K + 1)
    for f in (0, 1):
        pn[0] += pi[idx(f, 0, -1)]
        for n in range(1, K + 1):
            pn[n] += pi[idx(f, n, 0): idx(f, n, 0) + k].sum()
    return pn, pi[bloco:].sum()


def verificar(horas, rng):
    """Roda o simulador por `horas` em regime: rodadas de 5 h, descartando a 1a hora de cada."""
    o = simular(horas // 4, "atual", True, duracao=300, n0=0, rng=rng)
    nm = o["n_min"][:, 60:]
    return dict(L=float(nm.mean()), ocupado=float((nm > 0).mean()))


# ------------------------------------------------------------------ medicao real
L_REAL = plan["no_sistema"].mean()
W_REAL = L_REAL / LAM_MEDIA
D_REAL = cheg.var(ddof=1) / cheg.mean()
VS_REAL = plan["sairam"].var(ddof=1)

if __name__ == "__main__":
    print("=== PARAMETROS CALIBRADOS COM OS DADOS ===")
    print(f"lambda medio = {LAM_MEDIA:.3f}/min | lambda_C = {LAM_C:.2f} | lambda_P = {LAM_P:.2f}")
    print(f"p (C->P) = {n_CP}/{n_CP+n_CC} = {P_CP:.3f} | r (P->C) = {n_PC}/{n_PC+n_PP} = {P_PC:.3f}")
    print(f"servico: media {S_MEDIO:.2f} s (mu = {MU:.3f}/min) | CV = {CV:.2f} | k = {K_ERLANG} etapas")
    print(f"medicao real: L = {L_REAL:.2f} | W = {W_REAL:.2f} min | D = {D_REAL:.2f} | "
          f"var(saidas/min) = {VS_REAL:.2f}\n")

    # ---- DEMO: passo a passo (1 jantar) e comparacao com a medicao
    print("=== DEMO: JANTAR SIMULADO (17h20-18h20) ===")
    r1 = simular(1, "atual", True, rng=np.random.default_rng(SEMENTE))
    rm = simular(100, "atual", True, rng=np.random.default_rng(SEMENTE))
    lo_, hi_ = np.percentile(rm["n_min"], [5, 95], axis=0)
    for m in range(0, DURACAO_MIN, 10):
        hh, mm = divmod(17 * 60 + 20 + m, 60)
        print(f"{hh:02d}:{mm:02d} | 1 jantar: {int(r1['n_min'][0, m]):2d} | media de 100: {rm['n_min'][:, m].mean():4.1f} "
              f"[{lo_[m]:4.1f}; {hi_[m]:4.1f}] | medido: {int(plan['no_sistema'][m]):2d}")
    print()

    # ---- VERIFICACAO: o programa em regime bate com a distribuicao pi da cadeia?
    print("=== VERIFICACAO: SIMULADOR EM REGIME x DISTRIBUICAO pi (1 atendente, com fases) ===")
    pn, p_pico = distribuicao_pi()
    verif = dict(pi=dict(L=float((np.arange(K_MAX + 1) * pn).sum()), ocupado=float(1 - pn[0])),
                 p_pico=float(p_pico))
    for h in (100, 1000):
        verif[f"h{h}"] = verificar(h, np.random.default_rng(SEMENTE))
    print(f"pi exata : L = {verif['pi']['L']:5.2f} | atendente ocupado = {100*verif['pi']['ocupado']:.1f}% "
          f"| tempo em pico = {100*p_pico:.1f}%")
    for h in (100, 1000):
        v = verif[f"h{h}"]
        print(f"{h:4d} h   : L = {v['L']:5.2f} | atendente ocupado = {100*v['ocupado']:.1f}%")
    print()

    # ---- VALIDACAO: tres modelos x 500 jantares
    print(f"=== VALIDACAO ({R_JANTARES} jantares, faixa de 90%) ===")
    modelos = [
        ("A: sem fases, servico exponencial (aula)", dict(com_fases=False, k=1, dt_s=5)),
        ("B: com fases, servico exponencial", dict(com_fases=True, k=1, dt_s=5)),
        ("C: com fases, servico Erlang (calibrado)", dict(com_fases=True, k=K_ERLANG, dt_s=0.25)),
    ]
    res = {}
    for nome, kw in modelos:
        o = simular(R_JANTARES, "atual", rng=np.random.default_rng(SEMENTE), **kw)
        res[nome] = o
        traj_lo, traj_hi = np.percentile(o["n_min"], [5, 95], axis=0)
        cobre = np.mean((plan["no_sistema"].to_numpy() >= traj_lo) & (plan["no_sistema"].to_numpy() <= traj_hi))
        rmse = np.sqrt(np.mean((o["n_min"].mean(axis=0) - plan["no_sistema"].to_numpy()) ** 2))
        print(f"{nome}\n  D = {o['D'].mean():.2f} {np.round(faixa(o['D']), 2)} (real {D_REAL:.2f}) | "
              f"var(saidas/min) = {o['var_saidas'].mean():.2f} (real {VS_REAL:.2f})\n"
              f"  L = {o['L'].mean():.1f} {np.round(faixa(o['L']), 1)} (real {L_REAL:.1f}) | "
              f"W = {np.nanmean(o['W']):.2f} {np.round(faixa(o['W']), 1)} (real {W_REAL:.2f})\n"
              f"  n(t) real dentro da faixa 90%: {100*cobre:.0f}% dos minutos | RMSE da media: {rmse:.1f} alunos")

    # ---- VALIDACAO CONDICIONADA: chegadas = as medidas; so o atendimento e sorteado
    print(f"\n=== VALIDACAO CONDICIONADA ({R_JANTARES} jantares, chegadas repetidas como medidas) ===")
    real_n = plan["no_sistema"].to_numpy()
    cond = {}
    for nome, kw in modelos[1:]:
        kw2 = dict(kw)
        kw2["com_fases"] = True
        o = simular(R_JANTARES, "atual", rng=np.random.default_rng(SEMENTE), chegadas_medidas=cheg, **kw2)
        cond[nome] = o
        lo, hi = np.percentile(o["n_min"], [5, 95], axis=0)
        dentro = (real_n >= lo) & (real_n <= hi)
        rmse = np.sqrt(np.mean((o["n_min"].mean(axis=0) - real_n) ** 2))
        rmse18 = np.sqrt(np.mean((o["n_min"].mean(axis=0)[:18] - real_n[:18]) ** 2))
        print(f"{nome}\n  L = {o['L'].mean():.1f} {np.round(faixa(o['L']), 1)} (real {L_REAL:.1f}) | "
              f"W = {np.nanmean(o['W']):.2f} {np.round(faixa(o['W']), 2)} (real {W_REAL:.2f})\n"
              f"  n(t) real dentro da faixa 90%: {100*dentro.mean():.0f}% (60 min), "
              f"{100*dentro[:18].mean():.0f}% (17h20-17h37, medido) | "
              f"RMSE: {rmse:.1f} (60 min), {rmse18:.1f} (17h20-17h37)")

    # ---- CENARIOS (modelo C)
    print(f"\n=== CENARIOS 'E SE?' (modelo C, {R_JANTARES} jantares) ===")
    cenarios = [("Atual (1 atendente)", "atual"), ("2.o atendente so no pico", "2_pico"),
                ("2 atendentes na hora toda", "2_sempre"), ("QR Code (hipotese: 4 s/aluno)", "qr")]
    nomes, wm, wlo, whi = [], [], [], []
    cen_res = {}
    for nome, cod in cenarios:
        o = simular(R_JANTARES, cod, True, rng=np.random.default_rng(SEMENTE))
        cen_res[cod] = o
        lo, hi = faixa(o["W"])
        p5 = np.mean(o["W"] < 5) * 100
        print(f"{nome:32s} L = {o['L'].mean():5.2f} | W = {np.nanmean(o['W']):5.2f} min [{lo:4.1f}; {hi:4.1f}] | "
              f"P(W<5) = {p5:3.0f}%")
        nomes.append(nome); wm.append(np.nanmean(o["W"])); wlo.append(lo); whi.append(hi)

    # ---- GRAFICOS
    t = np.arange(DURACAO_MIN)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for axx, nome, cor in [(ax[0], modelos[1][0], "#dc2626"), (ax[1], modelos[2][0], "#1d4ed8")]:
        o = cond[nome]
        lo, hi = np.percentile(o["n_min"], [5, 95], axis=0)
        axx.fill_between(t, lo, hi, color=cor, alpha=0.2, label="faixa 90% simulada")
        axx.plot(t, o["n_min"].mean(axis=0), color=cor, label="media simulada")
        axx.plot(t, plan["no_sistema"], "k", lw=2, label="medicao (n real)")
        axx.axvline(17.5, color="gray", ls=":")
        axx.set_title(nome + "\n(chegadas = medidas)", fontsize=10)
        axx.set_xlabel("minutos apos 17h20")
        axx.set_ylabel("alunos no sistema")
        axx.grid(alpha=.3)
        axx.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("fig_validacao.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    err = [np.array(wm) - np.array(wlo), np.array(whi) - np.array(wm)]
    b = ax.bar(range(4), wm, yerr=err, capsize=5, color=["#dc2626", "#f59e0b", "#16a34a", "#0284c7"])
    ax.axhline(5, color="k", ls="--", label="meta: 5 min")
    ax.set_xticks(range(4)); ax.set_xticklabels([n.replace(" (", "\n(") for n in nomes], fontsize=8)
    ax.set_ylabel("tempo medio no sistema W (min)"); ax.legend(); ax.grid(axis="y", alpha=.3)
    for bi, v, h in zip(b, wm, whi):
        ax.text(bi.get_x() + bi.get_width() / 2, h + 0.3, f"{v:.1f}", ha="center", fontweight="bold")
    plt.tight_layout(); plt.savefig("fig_cenarios.png", dpi=150); plt.close()
    print("\nGraficos salvos: fig_validacao.png e fig_cenarios.png")

    # ---- PAINEL WEB (pagina animada com todos os resultados)
    import webbrowser
    import painel_ru
    params = dict(
        lam_c=LAM_C, lam_p=LAM_P, lam_media=LAM_MEDIA, p_cp=P_CP, p_pc=P_PC,
        n_cp=n_CP, n_c=n_CP + n_CC, n_pc=n_PC, n_p=n_PC + n_PP,
        s_medio=S_MEDIO, cv=CV, mu=MU, k=K_ERLANG, partida_fria=PARTIDA_FRIA_S, mu_qr=MU_QR,
        n0=N_INICIAL, kmax=K_MAX, duracao=DURACAO_MIN, n_assinaturas=len(tempos),
        L_real=L_REAL, W_real=W_REAL, D_real=D_REAL, VS_real=VS_REAL,
    )
    params = {c: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for c, v in params.items()}
    dados = painel_ru.montar(params, plan, r1, modelos, res, cond, cenarios, cen_res, R_JANTARES,
                               verif=verif)
    pagina = painel_ru.gerar(dados)
    print(f"Painel salvo: {pagina.name}")
    if "--sem-navegador" not in sys.argv:
        webbrowser.open(pagina.as_uri())
