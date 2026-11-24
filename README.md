# Fila do RU: simulador do jantar

Trabalho de Avaliação de Desempenho (UFOPA). Simula a fila do jantar do Restaurante
Universitário (17h20 às 18h20, assinatura manual, 28 alunos esperando no início) com uma
cadeia de Markov de tempo discreto, valida o modelo com a medição do dia e testa cenários.

**Pergunta:** um segundo atendente no pico reduz o tempo médio no sistema para menos de 5 minutos?

## Como rodar

```bash
pip install -r requirements.txt
python simulador_ru_jantar.py
```

Roda em uns 20 s e, no fim, abre `painel_ru.html` no navegador. A página também abre sozinha,
com dois cliques, sem rodar nada.

- `--sem-navegador`: gera a página sem abrir o navegador.
- `--rapido`: 100 jantares em vez de 500. Os números mudam um pouco.

## O modelo

- **Estado** `(fase, n, etapa)`: calma ou pico, alunos no sistema (máximo 60) e andamento da assinatura.
- **Passo** de 0,25 s (14.400 passos por jantar).
- **Chegadas** em duas fases (modelo de Gilbert): 1,13/min na calma e 4,50/min no pico. A troca de fase é estimada da planilha (p = 3/45, r = 3/14 por minuto).
- **Assinatura** com média de 26,2 s e CV de 0,20, dividida em 25 etapas (Erlang-25).

## O que o script calcula

| Etapa | Resultado |
|---|---|
| Verificação | Em 100 h de simulação, o atendente fica ocupado 83,4% do tempo, contra 82,9% pela distribuição exata π |
| Chegadas | Índice D medido de 1,56; o modelo sem fases dá 0,84 e não reproduz as rajadas |
| Validação (500 jantares) | Erro médio de 0,7 aluno; W de 9,74 min simulado contra 9,68 min medido; fila real dentro da faixa de 90% em 100% dos minutos |
| Cenários | Hoje: 9,7 min · 2.º atendente só no pico: 4,8 min · 2 atendentes: 2,0 min · QR Code (hipótese): 0,4 min |

## Arquivos

| Arquivo | Para que serve |
|---|---|
| `simulador_ru_jantar.py` | Simulador: calibração, verificação, validação e cenários |
| `painel_ru.py` | Monta os resultados e grava a página |
| `painel_template.html` | Modelo da página (animação e gráficos) |
| `painel_ru.html` | Página gerada, pronta para abrir |
| `planilha_minuto_a_minuto_ru.csv` | Chegadas, saídas e alunos no sistema a cada minuto |
| `tempos_atendimentos_ru.csv` | Tempo de cada assinatura |
| `fig_validacao.png`, `fig_cenarios.png` | Gráficos gerados pelo script |

A animação da página roda no navegador com o mesmo modelo e os parâmetros calculados pelo
Python. Os resultados dos 500 jantares vêm do Python e ficam gravados na página.

## Limites

Um único dia atípico, com só 3 picos observados. O 2.º atendente foi suposto tão rápido quanto
o 1.º, e os 4 s por aluno do QR Code são hipótese, não medição.
