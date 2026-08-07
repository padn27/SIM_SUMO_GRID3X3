"""
compare_neighborhood.py

Descrição:
    Análise comparativa do efeito da coordenação baseada em vizinhança
    sobre o desempenho dos agentes MARL em diferentes níveis de cobertura
    da rede.

Funções:
    - Leitura dos resultados experimentais com e sem vizinhança
    - Cálculo de métricas médias e desvios padrão
    - Normalização da recompensa pelo número de agentes
    - Comparação entre configurações de 2, 8 e 12 agentes
    - Cálculo dos ganhos percentuais associados à vizinhança
    - Geração de tabelas consolidadas dos resultados

Configurações avaliadas:
    - 2 agentes
    - 8 agentes
    - 12 agentes

Entrada:
    results/metrics/

Saída:
    results/statistics/

Arquivos gerados:
    - comparacao_vizinhanca.csv
    - comparacao_vizinhanca_numerica.csv

Autor:
    Priscila A. D. Nicácio
"""


from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# DIRETÓRIOS DO PROJETO
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = BASE_DIR / "results" / "metrics"

OUTPUT_DIR = BASE_DIR / "results" / "statistics"


# ==========================================================
# ARQUIVOS E CONFIGURAÇÕES DOS EXPERIMENTOS
# ==========================================================

casos = [
    {
        "caso": "2 agentes",
        "vizinhanca": "Não",
        "arquivo": (
            INPUT_DIR
            / "parcial_semvizinhanca"
            / "metricas_marl_proc_0.csv"
        ),
        "agentes": ["n11", "n12"],
    },

    {
        "caso": "2 agentes",
        "vizinhanca": "Sim",
        "arquivo": (
            INPUT_DIR
            / "parcial_comvizinhanca"
            / "metricas_marl_proc_0.csv"
        ),
        "agentes": ["n11", "n12"],
    },

    {
        "caso": "8 agentes",
        "vizinhanca": "Não",
        "arquivo": (
            INPUT_DIR
            / "intermediario_semvizinhanca"
            / "metricas_marl_proc_0.csv"
        ),
        "agentes": None,
    },

    {
        "caso": "8 agentes",
        "vizinhanca": "Sim",
        "arquivo": (
            INPUT_DIR
            / "intermediario_comvizinhanca"
            / "metricas_marl_proc_0.csv"
        ),
        "agentes": None,
    },

    {
        "caso": "12 agentes",
        "vizinhanca": "Não",
        "arquivo": (
            INPUT_DIR
            / "total_semvizinhanca"
            / "metricas_marl_proc_0.csv"
        ),
        "agentes": None,
    },

    {
        "caso": "12 agentes",
        "vizinhanca": "Sim",
        "arquivo": (
            INPUT_DIR
            / "total_comvizinhanca"
            / "metricas_marl_proc_0.csv"
        ),
        "agentes": None,
    },
]


# ==========================================================
# CONFIGURAÇÕES DA ANÁLISE
# ==========================================================

# Número de episódios finais utilizados na análise.
JANELA_FINAL = 4050

# Fatores utilizados para conversão das métricas.
FATOR_DELAY = 1e4
FATOR_CO2 = 1e3


# ==========================================================
# CONVERSÃO DE VALORES NUMÉRICOS
# ==========================================================

def para_numero(valor):
    """
    Converte diferentes representações de valores para float.

    Trata:
        - valores numéricos;
        - strings com ponto decimal;
        - strings com vírgula decimal;
        - valores ausentes.

    Retorna:
        float ou NaN.
    """

    if pd.isna(valor):
        return np.nan

    if isinstance(
        valor,
        (int, float, np.number)
    ):
        return float(valor)

    texto = str(valor).strip()

    try:
        return float(texto)

    except ValueError:
        pass

    # Trata valores no formato brasileiro.
    texto = (
        texto
        .replace(".", "")
        .replace(",", ".")
    )

    try:
        return float(texto)

    except ValueError:
        return np.nan


# ==========================================================
# CONTAGEM DE AGENTES
# ==========================================================

def contar_agentes(df, agentes):
    """
    Determina o número de agentes considerados no experimento.
    """

    if agentes is not None:
        return len(agentes)

    if "semaforo" in df.columns:
        return df["semaforo"].nunique()

    return 1


# ==========================================================
# LEITURA DOS DADOS
# ==========================================================

def carregar_dados(caminho, agentes=None):
    """
    Carrega e prepara o arquivo de métricas experimentais.
    """

    if not caminho.exists():
        raise FileNotFoundError(
            f"\nArquivo não encontrado:\n{caminho}"
        )

    df = pd.read_csv(caminho)

    # Padronização dos nomes das colunas.
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    colunas_numericas = [
        "episodio",
        "reward_total",
        "delay_total",
        "fila_total",
        "throughput_total",
        "co2_total",
    ]

    for coluna in colunas_numericas:

        if coluna in df.columns:

            df[coluna] = (
                df[coluna]
                .apply(para_numero)
            )

    # ------------------------------------------------------
    # Seleção dos agentes
    # ------------------------------------------------------

    if agentes is not None:

        if "semaforo" not in df.columns:
            raise ValueError(
                "A coluna 'semaforo' não foi encontrada."
            )

        df = df[
            df["semaforo"].isin(agentes)
        ].copy()

    # ------------------------------------------------------
    # Remoção de episódios inválidos
    # ------------------------------------------------------

    if "episodio" not in df.columns:
        raise ValueError(
            "A coluna 'episodio' não foi encontrada."
        )

    df = df.dropna(
        subset=["episodio"]
    )

    if df.empty:
        raise ValueError(
            f"Sem dados válidos em:\n{caminho}"
        )

    return df


# ==========================================================
# CÁLCULO DAS MÉTRICAS
# ==========================================================

def calcular_metricas(df, n_agentes):
    """
    Calcula as métricas médias e os desvios padrão
    considerando a janela final de episódios.
    """

    colunas_necessarias = [
        "episodio",
        "reward_total",
        "delay_total",
        "fila_total",
        "throughput_total",
        "co2_total",
    ]

    colunas_faltantes = [
        coluna
        for coluna in colunas_necessarias
        if coluna not in df.columns
    ]

    if colunas_faltantes:

        raise ValueError(
            "Colunas necessárias não encontradas: "
            + ", ".join(colunas_faltantes)
        )

    # ------------------------------------------------------
    # Agregação por episódio
    # ------------------------------------------------------

    episodios = (
        df.groupby("episodio")
        .agg(
            reward=("reward_total", "mean"),
            delay=("delay_total", "mean"),
            fila=("fila_total", "mean"),
            throughput=("throughput_total", "mean"),
            co2=("co2_total", "mean"),
        )
        .reset_index()
        .sort_values("episodio")
    )

    # ------------------------------------------------------
    # Conversão das unidades
    # ------------------------------------------------------

    episodios["delay"] = (
        episodios["delay"]
        / FATOR_DELAY
    )

    episodios["co2"] = (
        episodios["co2"]
        / FATOR_CO2
    )

    # ------------------------------------------------------
    # Seleção da janela final
    # ------------------------------------------------------

    final = episodios.tail(
        JANELA_FINAL
    )

    if final.empty:
        raise ValueError(
            "Não existem episódios suficientes para a análise."
        )

    # ------------------------------------------------------
    # Reward
    # ------------------------------------------------------

    reward_mean = final["reward"].mean()

    reward_std = final["reward"].std()

    # Normalização pelo número de agentes.
    reward_mean_norm = (
        reward_mean / n_agentes
    )

    reward_std_norm = (
        reward_std / n_agentes
    )

    # ------------------------------------------------------
    # Resultado
    # ------------------------------------------------------

    return {
        "reward_mean": reward_mean,
        "reward_std": reward_std,

        "reward_norm_mean": reward_mean_norm,
        "reward_norm_std": reward_std_norm,

        "delay_mean": final["delay"].mean(),
        "delay_std": final["delay"].std(),

        "fila_mean": final["fila"].mean(),

        "fila_std": final["fila"].std(),

        "throughput_mean": (
            final["throughput"].mean()
        ),

        "throughput_std": (
            final["throughput"].std()
        ),

        "co2_mean": final["co2"].mean(),

        "co2_std": final["co2"].std(),

        "episodios_analisados": len(final),

    }


# ==========================================================
# GANHOS PERCENTUAIS
# ==========================================================

def ganho_percentual(sem, com):
    """
    Calcula o ganho percentual para métricas em que valores menores representam melhor desempenho.

    Exemplos:
        - Delay
        - Fila
        - CO2
    """

    if sem == 0:
        return np.nan

    return (
        100
        * (sem - com)
        / abs(sem)
    )


def ganho_throughput(sem, com):
    """
    Calcula o ganho percentual de throughput, para o qual valores maiores representam melhor desempenho.
    """

    if sem == 0:
        return np.nan

    return (
        100
        * (com - sem)
        / abs(sem)
    )


# ==========================================================
# FORMATAÇÃO
# ==========================================================

def fmt(media, desvio):
    """
    Formata média ± desvio padrão.
    """

    return (
        f"{media:.0f} ± {desvio:.0f}"
    )


def fmt_simples(valor):
    """
    Formata valores sem desvio padrão.
    """

    return f"{valor:.0f}"


# ==========================================================
# ANÁLISE DE UM CASO
# ==========================================================

def analisar_caso(item):
    """
    Carrega os dados e calcula as métricas de um caso.
    """

    print(
        f"\n{item['caso']} | "
        f"vizinhança = {item['vizinhanca']}"
    )

    print(
        f"Arquivo: {item['arquivo']}"
    )

    df = carregar_dados(
        item["arquivo"],
        item["agentes"]
    )

    n_agentes = contar_agentes(
        df,
        item["agentes"]
    )

    metricas = calcular_metricas(
        df,
        n_agentes
    )

    metricas["numero_agentes"] = n_agentes

    return metricas


# ==========================================================
# EXECUÇÃO DOS CASOS
# ==========================================================

def executar_experimentos():
    """
    Processa todos os casos definidos.
    """

    linhas = []

    numerico = []

    print(
        "\n" + "=" * 70
    )

    print(
        "LEITURA DOS RESULTADOS EXPERIMENTAIS"
    )

    print(
        "=" * 70
    )

    for item in casos:

        metricas = analisar_caso(
            item
        )

        # --------------------------------------------------
        # Tabela formatada
        # --------------------------------------------------

        linhas.append(
            {
                "Caso": item["caso"],

                "Viz": item["vizinhanca"],

                "Reward": fmt(
                    metricas["reward_mean"],
                    metricas["reward_std"]
                ),

                "Reward_norm": fmt(
                    metricas["reward_norm_mean"],
                    metricas["reward_norm_std"]
                ),

                "Delay (s)": fmt(
                    metricas["delay_mean"],
                    metricas["delay_std"]
                ),

                "Fila": fmt_simples(
                    metricas["fila_mean"]
                ),

                "Throughput": fmt_simples(
                    metricas["throughput_mean"]
                ),

                "CO₂": fmt_simples(
                    metricas["co2_mean"]
                ),
            }
        )

        # --------------------------------------------------
        # Tabela numérica
        # --------------------------------------------------

        numerico.append(
            {
                "Caso": item["caso"],

                "Viz": item["vizinhanca"],

                "Agentes": metricas[
                    "numero_agentes"
                ],

                "Reward": metricas[
                    "reward_mean"
                ],

                "Reward_std": metricas[
                    "reward_std"
                ],

                "Reward_norm": metricas[
                    "reward_norm_mean"
                ],

                "Reward_norm_std": metricas[
                    "reward_norm_std"
                ],

                "Delay": metricas[
                    "delay_mean"
                ],

                "Delay_std": metricas[
                    "delay_std"
                ],

                "Fila": metricas[
                    "fila_mean"
                ],

                "Fila_std": metricas[
                    "fila_std"
                ],

                "Throughput": metricas[
                    "throughput_mean"
                ],

                "Throughput_std": metricas[
                    "throughput_std"
                ],

                "CO2": metricas[
                    "co2_mean"
                ],

                "CO2_std": metricas[
                    "co2_std"
                ],

                "Episodios": metricas[
                    "episodios_analisados"
                ],
            }
        )

    return (
        pd.DataFrame(linhas),
        pd.DataFrame(numerico)
    )


# ==========================================================
# COMPARAÇÃO COM E SEM VIZINHANÇA
# ==========================================================

def calcular_ganhos(tabela_numerica):
    """
    Calcula os ganhos percentuais proporcionados pela utilização da vizinhança.
    """

    ganhos = []

    print(
        "\n" + "=" * 70
    )

    print(
        "GANHO PROPORCIONADO PELA VIZINHANÇA (%)"
    )

    print(
        "=" * 70
    )

    for caso in [
        "2 agentes",
        "8 agentes",
        "12 agentes"
    ]:

        sem = tabela_numerica[
            (tabela_numerica["Caso"] == caso)
            &
            (tabela_numerica["Viz"] == "Não")
        ]

        com = tabela_numerica[
            (tabela_numerica["Caso"] == caso)
            &
            (tabela_numerica["Viz"] == "Sim")
        ]

        if sem.empty or com.empty:

            print(
                f"\nDados incompletos para: {caso}"
            )

            continue

        sem = sem.iloc[0]

        com = com.iloc[0]

        reward_gain = ganho_percentual(
            abs(sem["Reward_norm"]),
            abs(com["Reward_norm"])
        )

        delay_gain = ganho_percentual(
            sem["Delay"],
            com["Delay"]
        )

        fila_gain = ganho_percentual(
            sem["Fila"],
            com["Fila"]
        )

        throughput_gain = ganho_throughput(
            sem["Throughput"],
            com["Throughput"]
        )

        co2_gain = ganho_percentual(
            sem["CO2"],
            com["CO2"]
        )

        print(
            f"\n{caso}"
        )

        print(
            f"Reward (norm): "
            f"{reward_gain:.2f}%"
        )

        print(
            f"Delay        : "
            f"{delay_gain:.2f}%"
        )

        print(
            f"Fila         : "
            f"{fila_gain:.2f}%"
        )

        print(
            f"Throughput   : "
            f"{throughput_gain:.2f}%"
        )

        print(
            f"CO2          : "
            f"{co2_gain:.2f}%"
        )

        ganhos.append(
            {
                "Caso": caso,

                "Reward_norm_gain_%":
                    reward_gain,

                "Delay_gain_%":
                    delay_gain,

                "Fila_gain_%":
                    fila_gain,

                "Throughput_gain_%":
                    throughput_gain,

                "CO2_gain_%":
                    co2_gain,
            }
        )

    return pd.DataFrame(ganhos)


# ==========================================================
# EXPORTAÇÃO DOS RESULTADOS
# ==========================================================

def salvar_resultados(
    tabela,
    tabela_numerica,
    ganhos
):
    """
    Salva as tabelas resultantes da análise.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------
    # Tabela formatada
    # ------------------------------------------------------

    arquivo_tabela = (
        OUTPUT_DIR
        / "comparacao_vizinhanca.csv"
    )

    tabela.to_csv(
        arquivo_tabela,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------
    # Tabela numérica
    # ------------------------------------------------------

    arquivo_numerico = (
        OUTPUT_DIR
        / "comparacao_vizinhanca_numerica.csv"
    )

    tabela_numerica.to_csv(
        arquivo_numerico,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------
    # Ganhos da vizinhança
    # ------------------------------------------------------

    arquivo_ganhos = (
        OUTPUT_DIR
        / "ganhos_vizinhanca.csv"
    )

    ganhos.to_csv(
        arquivo_ganhos,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ARQUIVOS GERADOS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTabela consolidada:\n"
        f"{arquivo_tabela}"
    )

    print(
        f"\nTabela numérica:\n"
        f"{arquivo_numerico}"
    )

    print(
        f"\nGanhos da vizinhança:\n"
        f"{arquivo_ganhos}"
    )


# ==========================================================
# PROGRAMA PRINCIPAL
# ==========================================================

def main():
    """
    Executa a análise completa.
    """

    tabela, tabela_numerica = (
        executar_experimentos()
    )

    # ------------------------------------------------------
    # Tabela final
    # ------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TABELA FINAL"
    )

    print(
        "=" * 70
    )

    print(
        tabela.to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Ganhos da vizinhança
    # ------------------------------------------------------

    ganhos = calcular_ganhos(
        tabela_numerica
    )

    # ------------------------------------------------------
    # Exportação
    # ------------------------------------------------------

    salvar_resultados(
        tabela,
        tabela_numerica,
        ganhos
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    main()