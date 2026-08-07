"""
Análise da calibração dos parâmetros da função de recompensa.

Calcula:

- recompensa média;
- emissão média de CO2;
- atraso médio;
- número médio de paradas;

e gera:

- tabela LaTeX para a seção:
"Calibração dos Parâmetros da Função de Recompensa";

- arquivo CSV contendo o resumo dos experimentos.

"""


import pandas as pd
from pathlib import Path



# ==========================================================
# CONFIGURAÇÃO
# ==========================================================


PASTA_RESULTADOS = Path(
    r"resultados_reward_calibration"
)


ARQUIVO_LATEX = (
    "tabela_reward_calibration.tex"
)


ARQUIVO_RESUMO = (
    "reward_calibration_summary.csv"
)



# ==========================================================
# CONFIGURAÇÕES AVALIADAS
# ==========================================================


CONFIGURACOES = {

    "1B": {
        "alpha": 1.0,
        "beta": 0.005,
        "lambda": 0.5
    },

    "3B": {
        "alpha": 1.0,
        "beta": 0.010,
        "lambda": 1.0
    },

    "1C": {
        "alpha": 1.5,
        "beta": 0.010,
        "lambda": 0.3
    },

    "1A": {
        "alpha": 1.0,
        "beta": 0.010,
        "lambda": 0.3
    },

    "3A": {
        "alpha": 1.0,
        "beta": 0.010,
        "lambda": 0.6
    },

    "2B": {
        "alpha": 1.0,
        "beta": 0.050,
        "lambda": 0.3
    },

    "2A": {
        "alpha": 1.0,
        "beta": 0.020,
        "lambda": 0.3
    }

}



# ==========================================================
# PROCESSAMENTO DOS RESULTADOS
# ==========================================================


resultados = []


for caso, parametros in CONFIGURACOES.items():


    arquivo = (
        PASTA_RESULTADOS /
        f"caso_{caso}.csv"
    )


    if not arquivo.exists():

        print(
            f"[AVISO] Resultado não encontrado: {arquivo}"
        )

        continue



    df = pd.read_csv(arquivo)



    colunas_necessarias = [

        "reward",
        "co2",
        "delay",
        "stops"

    ]


    for coluna in colunas_necessarias:

        if coluna not in df.columns:

            raise ValueError(
                f"Coluna '{coluna}' ausente em {arquivo}"
            )



    resultado = {


        "Caso":
        caso,


        "alpha":
        parametros["alpha"],


        "beta":
        parametros["beta"],


        "lambda":
        parametros["lambda"],



        "Recompensa":
        df["reward"].mean(),



        "CO2":
        df["co2"].mean(),



        "CO2_std":
        df["co2"].std(),



        "Atraso":
        df["delay"].mean(),



        "Paradas":
        df["stops"].mean()

    }



    resultados.append(resultado)



# DataFrame final

df_resultados = pd.DataFrame(resultados)



if df_resultados.empty:

    raise RuntimeError(
        "Nenhum resultado de calibração foi encontrado."
    )



# ==========================================================
# ORGANIZAÇÃO DOS RESULTADOS
# ==========================================================


# Melhor recompensa primeiro

df_resultados = (

    df_resultados

    .sort_values(

        by="Recompensa",

        ascending=False

    )

)



# Salva resumo completo

df_resultados.to_csv(

    ARQUIVO_RESUMO,

    index=False,

    float_format="%.4f"

)



# ==========================================================
# GERAÇÃO DA TABELA LATEX
# ==========================================================


linhas = []


for _, row in df_resultados.iterrows():


    destaque = ""


    if row["Caso"] == "1B":

        destaque = (
            "\\rowcolor{teal!25}\n"
        )



    linha = (

        f"{destaque}"

        f"\\textbf{{{row['Caso']}}} & "

        f"{row['alpha']:.1f} & "

        f"{row['beta']:.3f} & "

        f"{row['lambda']:.1f} & "

        f"{row['Recompensa']:.0f} & "

        f"{row['CO2']:.2f} "

        f"$\\pm$ "

        f"{row['CO2_std']:.2f} & "

        f"{row['Atraso']:.1f} & "

        f"{row['Paradas']:.0f} \\\\"

    )


    linhas.append(linha)



tabela = "\n".join(linhas)



with open(

    ARQUIVO_LATEX,

    "w",

    encoding="utf-8"

) as arquivo:


    arquivo.write(tabela)



print("\nProcessamento concluído.")

print(
    f"Tabela LaTeX gerada: {ARQUIVO_LATEX}"
)

print(
    f"Resumo CSV gerado: {ARQUIVO_RESUMO}"
)