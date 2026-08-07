"""
Geração das análises gráficas de escalabilidade do arcabouço experimental MARL.

Análises:

1) Escalabilidade da cobertura adaptativa na rede 3x3:

   - Central
   - Parcial
   - Intermediária
   - Total


2) Expansão da rede:

   - 3x3
   - 4x3


Métricas avaliadas:

- recompensa;
- atraso médio;
- vazão;
- emissão de CO2;
- quantidade de agentes.


Saída:

results/scalability/plots/

Autor:
Priscila A. D. Nicácio
"""


from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt



# ==========================================================
# DIRETÓRIOS
# ==========================================================


BASE_DIR = Path(__file__).resolve().parents[2]


INPUT_DIR = (
    BASE_DIR /
    "results" /
    "scalability"
)


OUTPUT_DIR = (
    INPUT_DIR /
    "plots"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



# ==========================================================
# ARQUIVOS
# ==========================================================


ARQUIVO_COVERAGE = (
    INPUT_DIR /
    "coverage_3x3.csv"
)


ARQUIVO_NETWORK = (
    INPUT_DIR /
    "network_scalability.csv"
)



# ==========================================================
# CONFIGURAÇÃO VISUAL
# ==========================================================


METRICAS = [

    ("reward",
     "Recompensa média",
     "reward_scalability.pdf"),


    ("delay",
     "Atraso médio (s)",
     "delay_scalability.pdf"),


    ("throughput",
     "Vazão (veículos/h)",
     "throughput_scalability.pdf"),


    ("co2",
     "Emissão de CO$_2$",
     "co2_scalability.pdf")

]



# ==========================================================
# LEITURA
# ==========================================================


def carregar_csv(arquivo):


    if not arquivo.exists():

        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{arquivo}"
        )


    return pd.read_csv(arquivo)



# ==========================================================
# COBERTURA 3x3
# ==========================================================


def plot_coverage(df):

    """
    Gera gráficos da escalabilidade
    por cobertura de agentes na rede 3x3.
    """


    ordem = [

        "Central",

        "Parcial",

        "Intermediario",

        "Total"

    ]


    df["config"] = pd.Categorical(

        df["config"],

        categories=ordem,

        ordered=True

    )


    df = (
        df
        .sort_values("config")
    )


    for coluna,titulo,arquivo in METRICAS:


        plt.figure(
            figsize=(7,4)
        )


        plt.plot(

            df["config"],

            df[coluna],

            marker="o"

        )


        plt.xlabel(
            "Cobertura adaptativa"
        )


        plt.ylabel(
            titulo
        )


        plt.grid(
            True
        )


        plt.tight_layout()



        plt.savefig(

            OUTPUT_DIR /
            arquivo,

            dpi=300,

            bbox_inches="tight"

        )


        plt.close()



    # quantidade de agentes

    plt.figure(
        figsize=(7,4)
    )


    plt.plot(

        df["config"],

        df["agents"],

        marker="o"

    )


    plt.xlabel(
        "Cobertura adaptativa"
    )


    plt.ylabel(
        "Número de agentes"
    )


    plt.grid(
        True
    )


    plt.tight_layout()


    plt.savefig(

        OUTPUT_DIR /
        "agents_scalability.pdf",

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()



# ==========================================================
# EXPANSÃO 3x3 -> 4x3
# ==========================================================


def plot_network_scalability(df):

    """
    Comparação entre dimensões de rede.
    """


    for coluna,titulo,arquivo in METRICAS:


        plt.figure(
            figsize=(6,4)
        )


        plt.bar(

            df["network"],

            df[coluna]

        )


        plt.xlabel(
            "Configuração da rede"
        )


        plt.ylabel(
            titulo
        )


        plt.grid(
            axis="y"
        )


        plt.tight_layout()


        plt.savefig(

            OUTPUT_DIR /
            f"network_{arquivo}",

            dpi=300,

            bbox_inches="tight"

        )


        plt.close()



    # número de agentes


    plt.figure(
        figsize=(6,4)
    )


    plt.bar(

        df["network"],

        df["agents"]

    )


    plt.xlabel(
        "Configuração da rede"
    )


    plt.ylabel(
        "Número de agentes"
    )


    plt.grid(
        axis="y"
    )


    plt.tight_layout()


    plt.savefig(

        OUTPUT_DIR /
        "network_agents.pdf",

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()



# ==========================================================
# EXECUÇÃO
# ==========================================================


def main():


    print(
        "="*70
    )

    print(
        "ANÁLISE DE ESCALABILIDADE MARL"
    )

    print(
        "="*70
    )



    df_coverage = carregar_csv(

        ARQUIVO_COVERAGE

    )


    df_network = carregar_csv(

        ARQUIVO_NETWORK

    )



    plot_coverage(

        df_coverage

    )


    plot_network_scalability(

        df_network

    )



    print(
        "\nGráficos gerados em:"
    )


    print(
        OUTPUT_DIR
    )



if __name__ == "__main__":

    main()