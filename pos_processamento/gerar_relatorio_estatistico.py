"""
Geração do relatório estatístico consolidado dos experimentos de controle de tráfego urbano baseados em aprendizado por reforço multiagente.

Funções:
- Leitura das métricas experimentais;
- Estatísticas descritivas consolidadas;
- Intervalo de confiança de 95%;
- Comparação estatística entre estratégias;
- Geração de tabelas para documentação científica.

Entrada:
results/metrics/

Saída:
results/post_processing/statistics/

Arquivos gerados:
- resumo_metricas.csv
- resumo_metricas.json
- teste_estatistico.csv
- estatisticas_descritivas.tex

Autor:
Priscila A. D. Nicácio
"""


from pathlib import Path
import pandas as pd
import numpy as np
import json

from scipy import stats



# ==========================================================
# DIRETÓRIOS
# ==========================================================


BASE_DIR = Path(__file__).resolve().parents[2]


INPUT_DIR = (
    BASE_DIR /
    "results" /
    "metrics"
)


OUTPUT_DIR = (
    BASE_DIR /
    "results" /
    "post_processing" /
    "statistics"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



# ==========================================================
# ARQUIVO DE ENTRADA
# ==========================================================


ARQUIVO_METRICAS = (
    INPUT_DIR /
    "metricas_marl_proc_001.xlsx"
)



# ==========================================================
# LEITURA
# ==========================================================


def carregar_metricas(arquivo):

    """
    Carrega as métricas experimentais.
    """

    if not arquivo.exists():

        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{arquivo}"
        )


    if arquivo.suffix.lower() == ".xlsx":

        df = pd.read_excel(arquivo)


    elif arquivo.suffix.lower() == ".csv":

        df = pd.read_csv(arquivo)


    else:

        raise ValueError(
            "Formato inválido. Utilize CSV ou XLSX."
        )


    return df



# ==========================================================
# INTERVALO DE CONFIANÇA
# ==========================================================


def intervalo_confianca(series, confianca=0.95):

    """
    Calcula intervalo de confiança da média.
    """

    dados = series.dropna()


    n = len(dados)


    if n < 2:

        return np.nan, np.nan


    media = dados.mean()


    erro = (
        stats.sem(dados)
        *
        stats.t.ppf(
            (1 + confianca) / 2,
            n - 1
        )
    )


    return (
        media - erro,
        media + erro
    )



# ==========================================================
# ESTATÍSTICAS DESCRITIVAS
# ==========================================================


def gerar_estatisticas(df):

    """
    Calcula estatísticas das métricas numéricas.
    """


    metricas = (

        df.select_dtypes(
            include=np.number
        )
        .columns
    )


    resultados = []


    for metrica in metricas:


        serie = df[metrica]


        ic_inf, ic_sup = intervalo_confianca(
            serie
        )


        resultados.append({

            "metrica": metrica,

            "media":
            serie.mean(),

            "desvio_padrao":
            serie.std(),

            "minimo":
            serie.min(),

            "maximo":
            serie.max(),

            "ic95_inferior":
            ic_inf,

            "ic95_superior":
            ic_sup

        })


    return pd.DataFrame(resultados)



# ==========================================================
# TESTE ESTATÍSTICO
# ==========================================================


def gerar_teste_estatistico(df):


    """
    Comparação estatística não paramétrica entre grupos experimentais.
    """


    if "tipo" not in df.columns:

        print(
            "Coluna 'tipo' não encontrada."
        )

        return pd.DataFrame()



    if "reward_total" not in df.columns:

        print(
            "Coluna reward_total não encontrada."
        )

        return pd.DataFrame()



    grupos = (

        df["tipo"]
        .unique()

    )


    resultados = []


    for i in range(len(grupos)):

        for j in range(i+1, len(grupos)):


            g1 = grupos[i]

            g2 = grupos[j]


            dados1 = (
                df[df["tipo"] == g1]
                ["reward_total"]
            )


            dados2 = (
                df[df["tipo"] == g2]
                ["reward_total"]
            )


            teste = stats.mannwhitneyu(
                dados1,
                dados2,
                alternative="two-sided"
            )


            resultados.append({

                "grupo_1": g1,

                "grupo_2": g2,

                "estatistica":
                teste.statistic,

                "p_valor":
                teste.pvalue

            })


    return pd.DataFrame(resultados)



# ==========================================================
# TABELA LATEX
# ==========================================================


def gerar_latex(df):

    """

    Gera tabela LaTeX das estatísticas.

    """


    linhas = []


    for _, row in df.iterrows():


        linhas.append(

            f"{row['metrica']} & "
            f"{row['media']:.3f} & "
            f"{row['desvio_padrao']:.3f} & "
            f"{row['minimo']:.3f} & "
            f"{row['maximo']:.3f} & "
            f"[{row['ic95_inferior']:.3f}, "
            f"{row['ic95_superior']:.3f}] \\\\"

        )


    return "\n".join(linhas)



# ==========================================================
# EXECUÇÃO PRINCIPAL
# ==========================================================


def main():


    print("="*70)

    print(
        "RELATÓRIO ESTATÍSTICO DOS EXPERIMENTOS MARL"
    )

    print("="*70)



    df = carregar_metricas(
        ARQUIVO_METRICAS
    )



    resumo = gerar_estatisticas(
        df
    )


    resumo.to_csv(

        OUTPUT_DIR /
        "resumo_metricas.csv",

        index=False

    )



    with open(

        OUTPUT_DIR /
        "resumo_metricas.json",

        "w",

        encoding="utf-8"

    ) as arquivo:


        json.dump(

            resumo.to_dict(
                orient="records"
            ),

            arquivo,

            indent=4,

            ensure_ascii=False

        )



    teste = gerar_teste_estatistico(
        df
    )


    teste.to_csv(

        OUTPUT_DIR /
        "teste_estatistico.csv",

        index=False

    )



    latex = gerar_latex(
        resumo
    )


    with open(

        OUTPUT_DIR /
        "estatisticas_descritivas.tex",

        "w",

        encoding="utf-8"

    ) as arquivo:


        arquivo.write(
            latex
        )



    print("\nArquivos gerados:")

    print(
        OUTPUT_DIR /
        "resumo_metricas.csv"
    )

    print(
        OUTPUT_DIR /
        "resumo_metricas.json"
    )

    print(
        OUTPUT_DIR /
        "teste_estatistico.csv"
    )

    print(
        OUTPUT_DIR /
        "estatisticas_descritivas.tex"
    )



# ==========================================================
# MAIN
# ==========================================================


if __name__ == "__main__":

    main()