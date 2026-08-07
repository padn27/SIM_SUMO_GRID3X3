#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pós-processamento dos resultados de avaliação.

Este script calcula estatísticas descritivas dos métodos de controle de tráfego avaliados na rede 3x3:

- Fixed-Time
- Max Pressure
- Controle Local
- Controle Cooperativo
- MARL Distribuído

São calculadas:

- recompensa média;
- atraso médio;
- vazão média;
- emissão média de CO2;
- ganhos percentuais relativos ao controle Local;
- coeficiente de variação (CV) da recompensa por agente.

Também são exportados arquivos auxiliares para análise estatística e geração das tabelas da dissertação.
"""


import os
import glob
import pandas as pd


# ==========================================================
# CONFIGURAÇÃO DE CAMINHOS
# ==========================================================


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


CSV_DIR = os.path.join(
    BASE_DIR,
    "DadosCSV",
    "avaliacao"
)


SAIDA_DIR = os.path.join(
    BASE_DIR,
    "resultados"
)


os.makedirs(
    SAIDA_DIR,
    exist_ok=True
)


SAIDA_TEX = os.path.join(
    SAIDA_DIR,
    "tabela_comparativa.tex"
)


# ==========================================================
# MÉTODOS AVALIADOS
# ==========================================================


METODOS = {

    "Fixed-Time": "fixed",

    "Max Pressure": "maxpressure",

    "Local": "local",

    "Cooperativa": "cooperativo",

    "MARL distribuído": "marl"

}


# ==========================================================
# LEITURA DOS CSVs
# ==========================================================


def carregar_csv(padrao):

    arquivos = glob.glob(
        os.path.join(
            CSV_DIR,
            f"*{padrao}*.csv"
        )
    )


    if not arquivos:

        raise FileNotFoundError(
            f"Nenhum CSV encontrado para: {padrao}"
        )


    dados = []


    for arquivo in arquivos:

        print("Lendo:", arquivo)

        df = pd.read_csv(
            arquivo
        )

        dados.append(df)


    return pd.concat(
        dados,
        ignore_index=True
    )



# ==========================================================
# ESTATÍSTICAS
# ==========================================================


def calcular_estatisticas(df):


    colunas = {

        "reward": "reward",

        "delay": "delay",

        "throughput": "throughput",

        "co2": "co2"

    }


    resultado = {}


    for nome, coluna in colunas.items():


        resultado[nome] = (

            df[coluna].mean(),

            df[coluna].std()

        )


    return resultado



# ==========================================================
# CV POR AGENTE
# ==========================================================


def calcular_cv_agentes(df):


    if "agent" not in df.columns:

        return None


    resultados = []


    for agente, grupo in df.groupby("agent"):


        media = grupo["reward"].mean()

        desvio = grupo["reward"].std()


        cv = abs(
            desvio / media
        ) * 100


        resultados.append({

            "agente": agente,

            "media_reward": media,

            "desvio_reward": desvio,

            "cv_percentual": cv

        })


    return pd.DataFrame(resultados)



# ==========================================================
# EXPORTAÇÃO DOS RESULTADOS
# ==========================================================


def salvar_resumo(resultados):


    lista = []


    for metodo, dados in resultados.items():


        lista.append({

            "metodo": metodo,

            "reward_media": dados["reward"][0],

            "reward_desvio": dados["reward"][1],

            "delay_media": dados["delay"][0],

            "delay_desvio": dados["delay"][1],

            "throughput_media": dados["throughput"][0],

            "throughput_desvio": dados["throughput"][1],

            "co2_media": dados["co2"][0],

            "co2_desvio": dados["co2"][1]

        })


    df = pd.DataFrame(lista)


    df.to_csv(

        os.path.join(
            SAIDA_DIR,
            "resumo_metricas.csv"
        ),

        index=False,

        encoding="utf-8"

    )


    df.to_json(

        os.path.join(
            SAIDA_DIR,
            "resumo_metricas.json"
        ),

        orient="records",

        indent=4,

        force_ascii=False

    )



# ==========================================================
# CÁLCULO DOS GANHOS
# ==========================================================


def reducao(base, valor):

    return (
        (base - valor)
        /
        base
        *
        100
    )



def aumento(base, valor):

    return (
        (valor - base)
        /
        base
        *
        100
    )



def imprimir_ganhos(resultados):


    local = resultados["Local"]


    for metodo in [

        "Cooperativa",

        "MARL distribuído"

    ]:


        dados = resultados[metodo]


        print("\nMétodo:", metodo)


        print(

            f"Redução atraso: "
            f"{reducao(local['delay'][0], dados['delay'][0]):.1f}%"

        )


        print(

            f"Redução CO2: "
            f"{reducao(local['co2'][0], dados['co2'][0]):.1f}%"

        )


        print(

            f"Aumento vazão: "
            f"{aumento(local['throughput'][0], dados['throughput'][0]):.1f}%"

        )



# ==========================================================
# FORMATAÇÃO
# ==========================================================


def numero(valor, casas=1):

    return (
        f"{valor:.{casas}f}"
        .replace(".", ",")
    )



# ==========================================================
# GERAÇÃO LATEX
# ==========================================================


def gerar_latex(resultados):


    referencia = resultados["Local"]


    linhas = []


    for metodo, dados in resultados.items():


        reward = dados["reward"]

        delay = dados["delay"]

        throughput = dados["throughput"]

        co2 = dados["co2"]


        atraso_pct = ""

        vazao_pct = ""

        co2_pct = ""


        if metodo in [

            "Cooperativa",

            "MARL distribuído"

        ]:


            atraso_pct = (
                f", {reducao(referencia['delay'][0], delay[0]):.0f}\\%"
            )


            vazao_pct = (
                f", {aumento(referencia['throughput'][0], throughput[0]):.0f}\\%"
            )


            co2_pct = (
                f", {reducao(referencia['co2'][0], co2[0]):.0f}\\%"
            )



        nome = metodo


        if metodo == "MARL distribuído":

            nome = (
                "\\rowcolor{teal!25}"
                "\\textbf{MARL distribuído}"
            )



        linha = f"""
{nome} &
${numero(reward[0])} \\pm {numero(reward[1])}$ &
${numero(delay[0],0)} \\pm {numero(delay[1])}{atraso_pct}$ &
${numero(throughput[0],0)} \\pm {numero(throughput[1])}{vazao_pct}$ &
${numero(co2[0],0)} \\pm {numero(co2[1])}{co2_pct}$
\\\\
"""


        linhas.append(linha)



    tabela = r"""
\begin{table}[H]
\centering

\caption{Comparação de desempenho entre os métodos de controle avaliados.}

\label{tab:local_coop_marl_percent}

\renewcommand{\arraystretch}{1.18}

\resizebox{\columnwidth}{!}{

\begin{tabular}{lcccc}

\toprule

\textbf{Método} &
\textbf{Recompensa Média} &
\textbf{Atraso Médio} &
\textbf{Vazão Média} &
\textbf{CO$_2$ Médio}
\\

\midrule

"""


    tabela += "\n".join(linhas)


    tabela += r"""

\bottomrule

\end{tabular}

}

\end{table}
"""


    return tabela



# ==========================================================
# EXECUÇÃO
# ==========================================================


if __name__ == "__main__":


    resultados = {}


    dados_cv = None


    for metodo, arquivo in METODOS.items():


        df = carregar_csv(
            arquivo
        )


        resultados[metodo] = calcular_estatisticas(
            df
        )


        if metodo == "MARL distribuído":

            dados_cv = calcular_cv_agentes(
                df
            )



    salvar_resumo(
        resultados
    )


    if dados_cv is not None:


        dados_cv.to_csv(

            os.path.join(
                SAIDA_DIR,
                "cv_agentes.csv"
            ),

            index=False,

            encoding="utf-8"

        )



    latex = gerar_latex(
        resultados
    )


    with open(

        SAIDA_TEX,

        "w",

        encoding="utf-8"

    ) as arquivo:


        arquivo.write(
            latex
        )


    imprimir_ganhos(
        resultados
    )


    print("\nTabela gerada:")
    print(SAIDA_TEX)