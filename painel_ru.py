"""
Gera o painel web (painel_ru.html) com os resultados do simulador.

Os numeros vem do simulador_ru_jantar.py (500 jantares, semente 42) e sao gravados
dentro da pagina. A animacao da fila roda no navegador com o mesmo modelo
(fases de Gilbert + assinatura em k etapas), usando os parametros calculados dos CSV.
"""
import json
import datetime
import pathlib
import numpy as np

BASE = pathlib.Path(__file__).resolve().parent


def _r(v, d=2):
    return np.round(np.asarray(v, dtype=float), d).tolist()


def _f(v, d=3):
    return round(float(v), d)


def montar(P, plan, r1, modelos, res, cond, cenarios, cen_res, R, verif=None):
    real = plan["no_sistema"].to_numpy()

    def banda(o):
        lo, hi = np.percentile(o["n_min"], [5, 95], axis=0)
        return dict(media=_r(o["n_min"].mean(axis=0)), lo=_r(lo, 1), hi=_r(hi, 1))

    def resumo(o):
        lo, hi = np.percentile(o["n_min"], [5, 95], axis=0)
        media = o["n_min"].mean(axis=0)
        dentro = (real >= lo) & (real <= hi)
        return dict(
            L=_f(o["L"].mean()), L_faixa=_r(np.nanpercentile(o["L"], [5, 95]), 1),
            W=_f(np.nanmean(o["W"])), W_faixa=_r(np.nanpercentile(o["W"], [5, 95]), 2),
            D=_f(np.nanmean(o["D"])), D_faixa=_r(np.nanpercentile(o["D"], [5, 95]), 2),
            var_saidas=_f(o["var_saidas"].mean()),
            rmse=_f(np.sqrt(np.mean((media - real) ** 2))),
            rmse18=_f(np.sqrt(np.mean((media[:18] - real[:18]) ** 2))),
            cobertura=_f(dentro.mean()),
            **banda(o),
        )

    letras = ["A", "B", "C"]
    nome_C = modelos[2][0]
    oC = res[nome_C]

    return dict(
        gerado=datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
        R=R,
        P=P,
        verificacao=verif,
        medicao=dict(
            minuto=plan["minuto"].tolist(),
            chegaram=plan["chegaram"].astype(int).tolist(),
            sairam=plan["sairam"].astype(int).tolist(),
            no_sistema=plan["no_sistema"].astype(int).tolist(),
            fase=plan["fase"].tolist(),
        ),
        um_vs_500=dict(
            um=r1["n_min"][0].astype(int).tolist(),
            amostras=oC["n_min"][:40].astype(int).tolist(),
            **banda(oC),
        ),
        modelos=[dict(letra=l, nome=n, **resumo(res[n])) for l, (n, _) in zip(letras, modelos)],
        validacao=[dict(letra=l, nome=n, **resumo(cond[n])) for l, (n, _) in zip(letras[1:], modelos[1:])],
        cenarios=[dict(cod=cod, nome=nome,
                       L=_f(cen_res[cod]["L"].mean()), W=_f(np.nanmean(cen_res[cod]["W"])),
                       W_faixa=_r(np.nanpercentile(cen_res[cod]["W"], [5, 95]), 2),
                       p5=_f(np.mean(cen_res[cod]["W"] < 5)),
                       media=_r(cen_res[cod]["n_min"].mean(axis=0)))
                  for nome, cod in cenarios],
    )


def gerar(dados, template="painel_template.html", saida="painel_ru.html"):
    html = (BASE / template).read_text(encoding="utf-8")
    js = json.dumps(dados, ensure_ascii=False).replace("</", "<\\/")
    destino = BASE / saida
    destino.write_text(html.replace("__DADOS_JSON__", js), encoding="utf-8")
    return destino
