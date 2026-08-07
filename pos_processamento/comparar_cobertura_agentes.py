"""
Descrição:
    Geração das curvas comparativas de recompensa e emissões de CO₂ para diferentes níveis de cobertura de agentes na rede 3×3.

Configurações avaliadas:
    - Central (1 agente)
    - Parcial (2 agentes)
    - Intermediário (6 agentes)
    - Total (9 agentes)

Funções:
    - Leitura dos arquivos de métricas experimentais
    - Seleção dos agentes de cada configuração
    - Agregação das métricas por episódio
    - Suavização por média móvel
    - Tratamento dos valores extremos das séries brutas
    - Geração das curvas de recompensa
    - Geração das curvas de emissões de CO₂
    - Exportação em PDF e PNG

Entrada:
    results/metrics/

Saída:
    results/figures/generated/

Autor:
    Priscila A. D. Nicácio
"""


from pathlib import Path

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
# CONFIGURAÇÕES GERAIS
# ==========================================================

WINDOW = 1800

EPISODE_DURATION = 3600

USE_CO2_RATE = True

SHOW_RAW_SERIES = True

CLIP_RAW_SERIES_ONLY = True


# ==========================================================
# CONFIGURAÇÕES DA FIGURA
# ==========================================================

FIG_WIDTH = 9.2

FIG_HEIGHT = 2.9

RAW_ALPHA = 0.10

RAW_LINE_WIDTH = 0.55

SMOOTH_LINE_WIDTH = 1.6

GRID_ALPHA = 0.22

LABEL_FONT_SIZE = 30

TICK_FONT_SIZE = 27

LEGEND_FONT_SIZE = 24


# ==========================================================
# CONFIGURAÇÃO VISUAL
# ==========================================================

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.6,
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ==========================================================
# CONFIGURAÇÕES DE COBERTURA
# ==========================================================

configurations = {

    "Central (1)": [
        "n11"
    ],

    "Parcial (2)": [
        "n10",
        "n11"
    ],

    "Intermediário (6)": [
        "n22",
        "n21",
        "n11",
        "n01",
        "n10",
        "n12"
    ],

    "Total (9)": [
        "n00",
        "n01",
        "n02",
        "n10",
        "n11",
        "n12",
        "n20",
        "n21",
        "n22"
    ],
}


# ==========================================================
# ARQUIVOS DE MÉTRICAS
# ==========================================================

files = {

    "Central (1)":
        INPUT_DIR
        / "central"
        / "metricas_marl_proc_0.csv",

    "Parcial (2)":
        INPUT_DIR
        / "parcial"
        / "metricas_marl_proc_p.csv",

    "Intermediário (6)":
        INPUT_DIR
        / "intermediario"
        / "metricas_marl_proc_i.csv",

    "Total (9)":
        INPUT_DIR
        / "total"
        / "metricas_marl_proc_t.csv",
}


# ==========================================================
# FUNÇÕES AUXILIARES
# ==========================================================

def moving_average(
    series,
    window
):
    """
    Calcula a média móvel da série temporal.
    """

    return (
        series
        .rolling(
            window=window,
            min_periods=1
        )
        .mean()
    )


# ==========================================================

def winsorize_series(
    series,
    lower_q=0.01,
    upper_q=0.99
):
    """
    Limita valores extremos da série utilizando
    os quantis especificados.

    O tratamento é aplicado somente à série utilizada
    na visualização dos dados brutos.
    """

    low = series.quantile(
        lower_q
    )

    high = series.quantile(
        upper_q
    )

    return series.clip(
        lower=low,
        upper=high
    )


# ==========================================================

def prepare_episode_series(
    df_config,
    column
):
    """
    Calcula a média da métrica por episódio para
    os agentes pertencentes à configuração.
    """

    return (
        df_config
        .groupby("episodio")[column]
        .mean()
        .sort_index()
    )


# ==========================================================

def carregar_dados(
    caminho
):
    """
    Carrega um arquivo CSV e realiza validações básicas.
    """

    if not caminho.exists():

        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{caminho}"
        )

    df = pd.read_csv(
        caminho
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    colunas_obrigatorias = [
        "episodio",
        "semaforo",
        "reward_total",
        "co2_total",
    ]

    ausentes = [
        coluna
        for coluna in colunas_obrigatorias
        if coluna not in df.columns
    ]

    if ausentes:

        raise ValueError(
            "Colunas obrigatórias ausentes: "
            f"{ausentes}\n"
            f"Colunas disponíveis: "
            f"{list(df.columns)}"
        )

    return df


# ==========================================================

def preparar_series():
    """
    Carrega os dados e prepara as séries temporais
    de recompensa e CO₂ para cada configuração.
    """

    reward_raw = {}

    reward_plot_raw = {}

    reward_smooth = {}

    co2_raw = {}

    co2_plot_raw = {}

    co2_smooth = {}

    # ------------------------------------------------------
    # Processamento das configurações
    # ------------------------------------------------------

    for config, semaforos in configurations.items():

        print(
            f"\nProcessando: {config}"
        )

        caminho = files[config]

        df = carregar_dados(
            caminho
        )

        # --------------------------------------------------
        # Seleção dos agentes
        # --------------------------------------------------

        df_config = df[
            df["semaforo"]
            .isin(semaforos)
        ].copy()

        if df_config.empty:

            raise ValueError(
                f"Nenhum dado encontrado "
                f"para {config}."
            )

        # --------------------------------------------------
        # Ordenação
        # --------------------------------------------------

        df_config = (
            df_config
            .sort_values(
                [
                    "episodio",
                    "semaforo"
                ]
            )
        )

        # ==================================================
        # RECOMPENSA
        # ==================================================

        reward_episode = (
            prepare_episode_series(
                df_config,
                "reward_total"
            )
        )

        reward_raw[config] = (
            reward_episode
        )

        if CLIP_RAW_SERIES_ONLY:

            reward_plot_raw[config] = (
                winsorize_series(
                    reward_episode,
                    0.01,
                    0.99
                )
            )

        else:

            reward_plot_raw[config] = (
                reward_episode.copy()
            )

        reward_smooth[config] = (
            moving_average(
                reward_episode,
                WINDOW
            )
        )

        # ==================================================
        # CO₂
        # ==================================================

        co2_episode = (
            prepare_episode_series(
                df_config,
                "co2_total"
            )
        )

        if USE_CO2_RATE:

            co2_episode = (
                co2_episode
                / EPISODE_DURATION
            )

        co2_raw[config] = (
            co2_episode
        )

        if CLIP_RAW_SERIES_ONLY:

            co2_plot_raw[config] = (
                winsorize_series(
                    co2_episode,
                    0.01,
                    0.99
                )
            )

        else:

            co2_plot_raw[config] = (
                co2_episode.copy()
            )

        co2_smooth[config] = (
            moving_average(
                co2_episode,
                WINDOW
            )
        )

    return (
        reward_raw,
        reward_plot_raw,
        reward_smooth,
        co2_raw,
        co2_plot_raw,
        co2_smooth
    )


# ==========================================================
# GERAÇÃO DOS GRÁFICOS
# ==========================================================

def plot_metric(
    raw_dict,
    raw_plot_dict,
    smooth_dict,
    ylabel,
    output_name,
    legend_loc="upper right",
    ylim=None,
    xmax=None
):
    """
    Gera e salva uma figura comparativa para uma métrica.
    """

    fig, ax = plt.subplots(
        figsize=(
            FIG_WIDTH,
            FIG_HEIGHT
        ),
        facecolor="white",
        constrained_layout=True
    )

    # ------------------------------------------------------
    # Curvas
    # ------------------------------------------------------

    for config in configurations.keys():

        x = (
            raw_dict[config]
            .index
            .values
        )

        y_raw = (
            raw_plot_dict[config]
            .values
        )

        y_smooth = (
            smooth_dict[config]
            .values
        )

        # --------------------------------------------------
        # Série bruta
        # --------------------------------------------------

        if SHOW_RAW_SERIES:

            ax.plot(
                x,
                y_raw,
                alpha=RAW_ALPHA,
                linewidth=RAW_LINE_WIDTH,
                label="_nolegend_"
            )

        # --------------------------------------------------
        # Série suavizada
        # --------------------------------------------------

        ax.plot(
            x,
            y_smooth,
            linewidth=SMOOTH_LINE_WIDTH,
            label=config
        )

    # ======================================================
    # EIXOS
    # ======================================================

    ax.set_xlabel(
        "Episódio de Treinamento",
        fontsize=LABEL_FONT_SIZE,
        labelpad=4
    )

    ax.set_ylabel(
        ylabel,
        fontsize=LABEL_FONT_SIZE,
        labelpad=4
    )

    ax.tick_params(
        axis="both",
        labelsize=TICK_FONT_SIZE,
        length=3,
        width=0.6
    )

    # ======================================================
    # GRADE
    # ======================================================

    ax.grid(
        True,
        alpha=GRID_ALPHA,
        linewidth=0.4
    )

    # ======================================================
    # LEGENDA
    # ======================================================

    ax.legend(
        fontsize=LEGEND_FONT_SIZE,
        frameon=True,
        loc=legend_loc,
        borderpad=0.22,
        labelspacing=0.22,
        handlelength=1.5,
        handletextpad=0.4,
        borderaxespad=0.2
    )

    # ======================================================
    # LIMITES
    # ======================================================

    ax.set_xlim(
        left=0
    )

    if xmax is not None:

        ax.set_xlim(
            0,
            xmax
        )

    if ylim is not None:

        ax.set_ylim(
            ylim
        )

    ax.margins(
        x=0,
        y=0.02
    )

    # ======================================================
    # EXPORTAÇÃO
    # ======================================================

    pdf_path = (
        OUTPUT_DIR
        / f"{output_name}.pdf"
    )

    png_path = (
        OUTPUT_DIR
        / f"{output_name}.png"
    )

    plt.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0
    )

    plt.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close(fig)

    print(
        f"\nPDF salvo em:\n{pdf_path}"
    )

    print(
        f"PNG salvo em:\n{png_path}"
    )


# ==========================================================
# PROGRAMA PRINCIPAL
# ==========================================================

def main():
    """
    Executa o processamento completo e gera as figuras
    comparativas de recompensa e CO₂.
    """

    print(
        "=" * 70
    )

    print(
        "COMPARAÇÃO DE COBERTURA DE AGENTES"
    )

    print(
        "=" * 70
    )

    (
        reward_raw,
        reward_plot_raw,
        reward_smooth,
        co2_raw,
        co2_plot_raw,
        co2_smooth
    ) = preparar_series()

    # ======================================================
    # RECOMPENSA
    # ======================================================

    plot_metric(
        raw_dict=reward_raw,
        raw_plot_dict=reward_plot_raw,
        smooth_dict=reward_smooth,
        ylabel="Recompensa Total",
        output_name="rec24",
        legend_loc="upper right",
        ylim=(-8500, -90),
        xmax=4000
    )

    # ======================================================
    # CO₂
    # ======================================================

    if USE_CO2_RATE:

        co2_ylabel = (
            r"Emissões de CO$_2$ (mg/s)"
        )

    else:

        co2_ylabel = (
            r"Emissões Totais de CO$_2$ (mg)"
        )

    plot_metric(
        raw_dict=co2_raw,
        raw_plot_dict=co2_plot_raw,
        smooth_dict=co2_smooth,
        ylabel=co2_ylabel,
        output_name="co24",
        legend_loc="upper right",
        ylim=None,
        xmax=4000
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PROCESSAMENTO CONCLUÍDO"
    )

    print(
        "=" * 70
    )

    print(
        f"\nFiguras salvas em:\n"
        f"{OUTPUT_DIR}"
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    main()
```
