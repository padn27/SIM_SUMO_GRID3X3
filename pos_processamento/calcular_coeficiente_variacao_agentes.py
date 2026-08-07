"""
calculate_cv_agents.py

Descrição:
    Cálculo da variabilidade das recompensas obtidas pelos agentes MARL.

Funções:
    - Leitura das métricas experimentais
    - Filtragem dos experimentos MARL
    - Agrupamento das recompensas por agente
    - Cálculo de média, desvio padrão e coeficiente de variação
    - Classificação da variabilidade espacial
    - Exportação do arquivo CV_agentes.csv

Entrada:
    results/metrics/

Saída:
    results/statistics/CV_agentes.csv

Autor:
    Priscila A. D. Nicácio
"""


from pathlib import Path
import pandas as pd
import numpy as np


# ==========================================================
# DIRETÓRIOS DO PROJETO
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = BASE_DIR / "results" / "metrics"

OUTPUT_DIR = BASE_DIR / "results" / "statistics"


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

ARQUIVO_ENTRADA = (
    INPUT_DIR /
    "metricas_marl_proc_001.csv"
)

ARQUIVO_SAIDA = (
    OUTPUT_DIR /
    "CV_agentes.csv"
)


COLUNA_AGENTE = "semaforo"
COLUNA_RECOMPENSA = "reward_total"
COLUNA_TIPO = "tipo"


# ==========================================================
# CLASSIFICAÇÃO DO COEFICIENTE DE VARIAÇÃO
# ==========================================================

def classificar_cv(cv):
    """
    Classifica o coeficiente de variação.

    Critério utilizado:
        CV <= 25%       -> Baixa
        25% < CV < 75%  -> Média
        CV >= 75%       -> Alta
    """

    if cv <= 25:
        return "Baixa"

    elif cv < 75:
        return "Média"

    else:
        return "Alta"


# ==========================================================
# LEITURA DOS DADOS
# ==========================================================

def carregar_dados(arquivo):
    """
    Realiza a leitura do arquivo de métricas.
    """

    if not arquivo.exists():
        raise FileNotFoundError(
            f"\nArquivo não encontrado:\n{arquivo}"
        )


    if arquivo.suffix.lower() == ".csv":

        df = pd.read_csv(arquivo)


    elif arquivo.suffix.lower() in [".xlsx", ".xls"]:

        df = pd.read_excel(arquivo)


    else:

        raise ValueError(
            "Formato não suportado. Utilize CSV ou XLSX."
        )


    return df


# ==========================================================
# PROCESSAMENTO
# ==========================================================

def calcular_cv(df):
    """
    Calcula estatísticas de recompensa por agente.
    """


    print("\nColunas encontradas:")
    print(df.columns.tolist())


    # ------------------------------------------------------
    # Seleção apenas dos agentes RL
    # ------------------------------------------------------

    if COLUNA_TIPO in df.columns:

        df = df[
            df[COLUNA_TIPO] == "RL"
        ].copy()


    # ------------------------------------------------------
    # Remoção de valores inválidos
    # ------------------------------------------------------

    df = df.dropna(
        subset=[
            COLUNA_AGENTE,
            COLUNA_RECOMPENSA
        ]
    )


    df[COLUNA_RECOMPENSA] = pd.to_numeric(
        df[COLUNA_RECOMPENSA],
        errors="coerce"
    )


    df = df.dropna(
        subset=[
            COLUNA_RECOMPENSA
        ]
    )


    print("\nQuantidade de amostras:")
    print(df.shape)


    # ------------------------------------------------------
    # Estatísticas por agente
    # ------------------------------------------------------

    resultado = (

        df.groupby(COLUNA_AGENTE)[COLUNA_RECOMPENSA]

        .agg(
            recompensa_media="mean",
            recompensa_min="min",
            recompensa_max="max",
            numero_ep="count"
        )

        .reset_index()

    )


    desvio = (

        df.groupby(COLUNA_AGENTE)[COLUNA_RECOMPENSA]

        .std(ddof=0)

        .reset_index(
            name="desvio_padrao"
        )

    )


    resultado = resultado.merge(
        desvio,
        on=COLUNA_AGENTE
    )


    # ------------------------------------------------------
    # Coeficiente de variação
    # ------------------------------------------------------

    resultado["CV_%"] = np.where(

        resultado["recompensa_media"].abs() > 1e-12,

        (
            resultado["desvio_padrao"]
            /
            resultado["recompensa_media"].abs()
        ) * 100,

        np.nan

    )


    resultado["classificacao"] = (
        resultado["CV_%"]
        .apply(classificar_cv)
    )


    return resultado


# ==========================================================
# ORDENAÇÃO ESPACIAL DA REDE 4x3
# ==========================================================

def ordenar_agentes(resultado):
    """
    Organiza os agentes conforme a posição espacial da rede.
    """


    ordem = [

        "n00", "n01", "n02", "n03",

        "n10", "n11", "n12", "n13",

        "n20", "n21", "n22", "n23"

    ]


    resultado[COLUNA_AGENTE] = pd.Categorical(

        resultado[COLUNA_AGENTE],

        categories=ordem,

        ordered=True

    )


    resultado = resultado.sort_values(
        COLUNA_AGENTE
    )


    return resultado


# ==========================================================
# EXPORTAÇÃO
# ==========================================================

def salvar_resultado(resultado):
    """
    Exporta o arquivo final de variabilidade.
    """


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    resultado.to_csv(

        ARQUIVO_SAIDA,

        index=False,

        encoding="utf-8-sig"

    )


    print("\nArquivo salvo:")
    print(ARQUIVO_SAIDA)


# ==========================================================
# EXECUÇÃO PRINCIPAL
# ==========================================================

def main():

    print("=" * 70)
    print("VARIABILIDADE DAS RECOMPENSAS POR AGENTE MARL")
    print("=" * 70)


    df = carregar_dados(
        ARQUIVO_ENTRADA
    )


    resultado = calcular_cv(
        df
    )


    resultado = ordenar_agentes(
        resultado
    )


    print(
        "\n=============================================="
    )

    print(
        "RESULTADO FINAL"
    )

    print(
        "==============================================\n"
    )


    print(

        resultado[

            [

                COLUNA_AGENTE,

                "recompensa_media",

                "desvio_padrao",

                "CV_%",

                "classificacao",

                "numero_ep"

            ]

        ]

        .to_string(

            index=False,

            formatters={

                "recompensa_media":
                    "{:.2f}".format,

                "desvio_padrao":
                    "{:.2f}".format,

                "CV_%":
                    "{:.2f}".format

            }

        )

    )


    salvar_resultado(
        resultado
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    main()