```python
"""
plot_reward_curves.py

Descrição:
    Geração das curvas de recompensa por agente durante o treinamento dos experimentos MARL, 
    considerando diferentes níveis de cobertura da rede e a utilização ou não de coordenação baseada em vizinhança.

Funções:
    - Leitura dos arquivos de métricas experimentais
    - Seleção dos agentes de acordo com o nível de cobertura
    - Cálculo da recompensa média por episódio e agente
    - Remoção de valores extremos
    - Suavização das curvas por média móvel e média móvel exponencial
    - Geração das curvas de recompensa
    - Exportação das figuras em PDF

Configurações avaliadas:
    - Parcial com vizinhança
    - Parcial sem vizinhança
    - Intermediário com vizinhança
    - Intermediário sem vizinhança
    - Total com vizinhança
    - Total sem vizinhança

Entrada:
    results/metrics/

Saída:
    results/figures/generated/

Autor:
    Priscila A. D. Nicácio
"""


from pathlib import Path
from itertools import cycle

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# DIRETÓRIOS DO PROJETO
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    BASE_DIR
    / "results"
    / "metrics"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "figures"
    / "generated"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# ARQUIVOS DE ENTRADA
# ==========================================================

arquivos = {

    "Intermediário com vizinhança":
        INPUT_DIR
        / "intermediario_comvizinhanca"
        / "metricas_marl_proc_0.csv",

    "Intermediário sem vizinhança":
        INPUT_DIR
        / "intermediario_semvizinhanca"
        / "metricas_marl_proc_0.csv",

    "Parcial com vizinhança":
        INPUT_DIR
        / "parcial_comvizinhanca"
        / "metricas_marl_proc_0.csv",

    "Parcial sem vizinhança":
        INPUT_DIR
        / "parcial_semvizinhanca"
        / "metricas_marl_proc_0.csv",

    "Total com vizinhança":
        INPUT_DIR
        / "total_comvizinhanca"
        / "metricas_marl_proc_0.csv",

    "Total sem vizinhança":
        INPUT_DIR
        / "total_semvizinhanca"
        / "metricas_marl_proc_0.csv",
}


# ==========================================================
# CONFIGURAÇÕES DO GRÁFICO
# ==========================================================

XMAX = 4050

ROLLING_WINDOW = 150

EWM_SPAN = 60

FIG_WIDTH = 8.5

FIG_HEIGHT = 6.2

LINE_WIDTH = 2.3

RAW_LINE_WIDTH = 0.9

RAW_ALPHA = 0.18

LABEL_FONT_SIZE = 18

TICK_FONT_SIZE = 13

LEGEND_FONT_SIZE = 10

LABEL_PAD = 15


# ==========================================================
# CONFIGURAÇÃO VISUAL
# ==========================================================

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.9,
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ==========================================================
# PERFIS DE COBERTURA
# ==========================================================

perfis = {

    "Parcial": [
        "n11",
        "n12",
    ],

    "Intermediário": [
        "n01",
        "n02",
        "n10",
        "n11",
        "n12",
        "n13",
        "n21",
        "n22",
    ],

    "Total": [
        "n00",
        "n01",
        "n02",
        "n03",
        "n10",
        "n11",
        "n12",
        "n13",
        "n20",
        "n21",
        "n22",
        "n23",
    ],
}


# ==========================================================
# LOCALIZAÇÃO FLEXÍVEL DAS COLUNAS
# ==========================================================

def encontrar_coluna(df, opcoes):
    """
    Localiza uma coluna considerando diferentes possíveis
    nomenclaturas utilizadas nos arquivos experimentais.
    """

    colunas = {
        coluna.lower().strip(): coluna
        for coluna in df.columns
    }

    for opcao in opcoes:

        chave = opcao.lower().strip()

        if chave in colunas:
            return colunas[chave]

    raise ValueError(
        "Coluna não encontrada. "
        f"Opções procuradas: {opcoes}\n"
        f"Colunas disponíveis: {list(df.columns)}"
    )


# ==========================================================
# LEITURA DOS ARQUIVOS
# ==========================================================

def ler_arquivo(caminho):
    """
    Lê arquivos CSV ou Excel.
    """

    extensao = caminho.suffix.lower()

    if extensao in [".xlsx", ".xls"]:
        return pd.read_excel(caminho)

    if extensao == ".csv":
        return pd.read_csv(caminho)

    raise ValueError(
        f"Formato não suportado: {extensao}"
    )


# ==========================================================
# SUAVIZAÇÃO DAS CURVAS
# ==========================================================

def suavizar_curva(serie):
    """
    Remove valores extremos e aplica suavização por:

        1. clipping dos extremos;
        2. média móvel;
        3. média móvel exponencial.
    """

    if serie.empty:
        return serie

    # ------------------------------------------------------
    # Remoção de valores extremos
    # ------------------------------------------------------

    limite_inferior = serie.quantile(0.01)

    limite_superior = serie.quantile(0.99)

    serie = serie.clip(
        lower=limite_inferior,
        upper=limite_superior
    )

    # ------------------------------------------------------
    # Média móvel
    # ------------------------------------------------------

    serie = (
        serie
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=1
        )
        .mean()
    )

    # ------------------------------------------------------
    # Média móvel exponencial
    # ------------------------------------------------------

    serie = (
        serie
        .ewm(
            span=EWM_SPAN,
            adjust=False
        )
        .mean()
    )

    return serie


# ==========================================================
# LIMPEZA DO NOME DO ARQUIVO
# ==========================================================

def limpar_nome(nome):
    """
    Converte o nome da estratégia em um nome adequado para o arquivo de saída.
    """

    substituicoes = {
        " ": "_",
        "ç": "c",
        "ã": "a",
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "â": "a",
        "ê": "e",
        "ô": "o",
    }

    nome = nome.lower()

    for antigo, novo in substituicoes.items():
        nome = nome.replace(
            antigo,
            novo
        )

    return nome


# ==========================================================
# IDENTIFICAÇÃO DO PERFIL
# ==========================================================

def identificar_perfil(nome):
    """
    Identifica o nível de cobertura a partir do nome da configuração experimental.
    """

    if "Parcial" in nome:
        return "Parcial"

    if "Intermediário" in nome:
        return "Intermediário"

    if "Total" in nome:
        return "Total"

    raise ValueError(
        f"Perfil não identificado: {nome}"
    )


# ==========================================================
# GERAÇÃO DO GRÁFICO
# ==========================================================

def gerar_grafico(
    nome_estrategia,
    caminho_arquivo
):
    """
    Gera a curva de recompensa por agente.
    """

    print(
        f"\nProcessando: "
        f"{nome_estrategia}"
    )

    # ------------------------------------------------------
    # Verificação do arquivo
    # ------------------------------------------------------

    if not caminho_arquivo.exists():

        print(
            "Arquivo não encontrado:"
        )

        print(
            caminho_arquivo
        )

        return

    # ------------------------------------------------------
    # Leitura
    # ------------------------------------------------------

    df = ler_arquivo(
        caminho_arquivo
    )

    # ------------------------------------------------------
    # Identificação das colunas
    # ------------------------------------------------------

    coluna_episodio = encontrar_coluna(
        df,
        [
            "episodio",
            "episode",
            "Episode",
            "Episódio",
        ]
    )

    coluna_reward = encontrar_coluna(
        df,
        [
            "reward_total",
            "Total_Reward",
            "total_reward",
            "Reward",
        ]
    )

    coluna_semaforo = encontrar_coluna(
        df,
        [
            "semaforo",
            "traffic_signal",
            "agent",
            "agente",
            "tls",
            "id",
        ]
    )

    # ------------------------------------------------------
    # Seleção das colunas
    # ------------------------------------------------------

    df = df[
        [
            coluna_episodio,
            coluna_semaforo,
            coluna_reward,
        ]
    ].copy()

    # ------------------------------------------------------
    # Conversão dos tipos
    # ------------------------------------------------------

    df[coluna_semaforo] = (
        df[coluna_semaforo]
        .astype(str)
        .str.strip()
    )

    df[coluna_reward] = pd.to_numeric(
        df[coluna_reward],
        errors="coerce"
    )

    df[coluna_episodio] = pd.to_numeric(
        df[coluna_episodio],
        errors="coerce"
    )

    # ------------------------------------------------------
    # Remoção de dados inválidos
    # ------------------------------------------------------

    df = df.dropna()

    # ------------------------------------------------------
    # Identificação do perfil
    # ------------------------------------------------------

    perfil = identificar_perfil(
        nome_estrategia
    )

    agentes = perfis[perfil]

    # ------------------------------------------------------
    # Seleção dos agentes
    # ------------------------------------------------------

    df = df[
        df[coluna_semaforo].isin(
            agentes
        )
    ].copy()

    if df.empty:

        print(
            "Nenhum dado encontrado "
            f"para o perfil {perfil}."
        )

        return

    # ======================================================
    # FIGURA
    # ======================================================

    fig, ax = plt.subplots(
        figsize=(
            FIG_WIDTH,
            FIG_HEIGHT
        )
    )

    # ======================================================
    # CORES
    # ======================================================

    cores = cycle(
        plt.cm.tab10.colors
    )

    curvas_plotadas = []

    # ======================================================
    # CURVAS DOS AGENTES
    # ======================================================

    for semaforo in agentes:

        df_semaforo = df[
            df[coluna_semaforo]
            == semaforo
        ]

        if df_semaforo.empty:
            continue

        # --------------------------------------------------
        # Recompensa média por episódio
        # --------------------------------------------------

        recompensa = (
            df_semaforo
            .groupby(
                coluna_episodio
            )[coluna_reward]
            .mean()
            .sort_index()
        )

        if recompensa.empty:
            continue

        # --------------------------------------------------
        # Suavização
        # --------------------------------------------------

        recompensa_suave = (
            suavizar_curva(
                recompensa
            )
        )

        # --------------------------------------------------
        # Cor
        # --------------------------------------------------

        cor = next(
            cores
        )

        curvas_plotadas.append(
            recompensa_suave
        )

        # --------------------------------------------------
        # Curva bruta
        # --------------------------------------------------

        ax.plot(
            recompensa.index,
            recompensa.values,
            linewidth=RAW_LINE_WIDTH,
            color=cor,
            alpha=RAW_ALPHA
        )

        # --------------------------------------------------
        # Curva suavizada
        # --------------------------------------------------

        ax.plot(
            recompensa_suave.index,
            recompensa_suave.values,
            linewidth=LINE_WIDTH,
            color=cor,
            label=f"Semáforo {semaforo}"
        )

    # ======================================================
    # CONFIGURAÇÃO DOS EIXOS
    # ======================================================

    ax.set_xlabel(
        "Episódio de Treinamento",
        fontsize=LABEL_FONT_SIZE,
        labelpad=LABEL_PAD
    )

    ax.set_ylabel(
        "Recompensa Média",
        fontsize=LABEL_FONT_SIZE,
        labelpad=LABEL_PAD
    )

    ax.tick_params(
        axis="both",
        labelsize=TICK_FONT_SIZE
    )

    # ======================================================
    # GRADE
    # ======================================================

    ax.grid(
        True,
        alpha=0.40,
        linewidth=0.8
    )

    # ======================================================
    # LIMITE DO EIXO X
    # ======================================================

    ax.set_xlim(
        0,
        XMAX
    )

    # ======================================================
    # LIMITE DINÂMICO DO EIXO Y
    # ======================================================

    if curvas_plotadas:

        serie_global = pd.concat(
            curvas_plotadas
        )

        ymin = serie_global.min()

        ymax = serie_global.max()

        diferenca = ymax - ymin

        # Proteção para curvas constantes
        if diferenca == 0:
            margem = max(
                abs(ymax) * 0.05,
                1.0
            )

        else:
            margem = (
                0.12
                * diferenca
            )

        ax.set_ylim(
            ymin - margem,
            ymax + margem
        )

    # ======================================================
    # LEGENDA
    # ======================================================

    ax.legend(
        fontsize=LEGEND_FONT_SIZE,
        loc="best",
        frameon=True
    )

    # ======================================================
    # AJUSTE FINAL
    # ======================================================

    plt.tight_layout(
        pad=1.5
    )

    # ======================================================
    # ARQUIVO DE SAÍDA
    # ======================================================

    nome_saida = (
        limpar_nome(
            nome_estrategia
        )
        + "_curvas.pdf"
    )

    caminho_saida = (
        OUTPUT_DIR
        / nome_saida
    )

    # ======================================================
    # EXPORTAÇÃO
    # ======================================================

    plt.savefig(
        caminho_saida,
        format="pdf",
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"PDF salvo em:\n"
        f"{caminho_saida}"
    )


# ==========================================================
# PROGRAMA PRINCIPAL
# ==========================================================

def main():
    """
    Gera todas as curvas de recompensa definidas.
    """

    print(
        "=" * 70
    )

    print(
        "GERAÇÃO DAS CURVAS DE RECOMPENSA"
    )

    print(
        "=" * 70
    )

    for nome, caminho in arquivos.items():

        try:

            gerar_grafico(
                nome,
                caminho
            )

        except Exception as erro:

            print(
                f"\nERRO ao processar "
                f"{nome}:"
            )

            print(
                erro
            )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TODOS OS GRÁFICOS FORAM PROCESSADOS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nResultados salvos em:\n"
        f"{OUTPUT_DIR}"
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    main()
```
