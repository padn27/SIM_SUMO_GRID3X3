import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================================
# CONFIGURAÇÃO
# ==========================================================

ARQUIVO_CV = Path("resultados") / "CV_agentes.csv"
SAIDA_FIGURA = Path("resultados") / "heatmap_cv_classes.png"


# ==========================================================
# LEITURA DOS DADOS
# ==========================================================

df = pd.read_csv(ARQUIVO_CV)

print("\nDados de entrada:")
print(df)


# ==========================================================
# CLASSIFICAÇÃO DO CV
# ==========================================================
# Critérios definidos na dissertação:
#
# Baixa  : CV <= 25%
# Média  : 25% < CV < 75%
# Alta   : CV >= 75%
# ==========================================================

def classificar_cv(cv):

    if cv <= 25:
        return "Baixa"

    elif cv < 75:
        return "Média"

    else:
        return "Alta"


df["classe_cv"] = df["CV_%"].apply(classificar_cv)


# ==========================================================
# CONVERSÃO DAS CLASSES PARA VALORES NUMÉRICOS
# ==========================================================

mapa_classes = {
    "Baixa": 1,
    "Média": 2,
    "Alta": 3
}

df["valor_classe"] = df["classe_cv"].map(mapa_classes)


# ==========================================================
# ORGANIZAÇÃO DA MATRIZ ESPACIAL 4x3
# ==========================================================

agentes = [
    ["n00", "n01", "n02", "n03"],
    ["n10", "n11", "n12", "n13"],
    ["n20", "n21", "n22", "n23"]
]

matriz_classe = np.zeros((3, 4))
matriz_cv = np.zeros((3, 4))


for i in range(3):

    for j in range(4):

        agente = agentes[i][j]

        linha = df.loc[df["semaforo"] == agente]

        if linha.empty:
            raise ValueError(
                f"Agente {agente} não encontrado no arquivo "
                f"{ARQUIVO_CV}"
            )

        matriz_classe[i, j] = linha["valor_classe"].iloc[0]
        matriz_cv[i, j] = linha["CV_%"].iloc[0]


# ==========================================================
# GERAÇÃO DO HEATMAP
# ==========================================================

plt.figure(figsize=(8, 4))


plt.imshow(
    matriz_classe,
    cmap="YlOrRd",
    interpolation="nearest",
    vmin=1,
    vmax=3
)


# ==========================================================
# BARRA DE CORES
# ==========================================================

cbar = plt.colorbar(ticks=[1, 2, 3])

cbar.ax.set_yticklabels(
    [
        "Baixa",
        "Média",
        "Alta"
    ]
)

cbar.set_label(
    "Variabilidade das recompensas"
)


# ==========================================================
# IDENTIFICAÇÃO DOS AGENTES
# ==========================================================

plt.xticks(
    range(4),
    ["n0", "n1", "n2", "n3"]
)

plt.yticks(
    range(3),
    ["linha 0", "linha 1", "linha 2"]
)


# ==========================================================
# VALORES NAS CÉLULAS
# ==========================================================

mapa_classes_inverso = {
    1: "Baixa",
    2: "Média",
    3: "Alta"
}


for i in range(3):

    for j in range(4):

        classe = mapa_classes_inverso[
            int(matriz_classe[i, j])
        ]

        valor_cv = matriz_cv[i, j]

        plt.text(
            j,
            i,
            f"{classe}\n{valor_cv:.1f}%",
            ha="center",
            va="center"
        )


# ==========================================================
# RÓTULOS DOS EIXOS
# ==========================================================

plt.xlabel("Coluna da interseção")
plt.ylabel("Linha da interseção")


plt.tight_layout()


# ==========================================================
# SALVAMENTO
# ==========================================================

SAIDA_FIGURA.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    SAIDA_FIGURA,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print("\nFigura salva em:")
print(SAIDA_FIGURA)