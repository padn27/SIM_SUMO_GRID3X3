"""
    Conversão, padronização e otimização de figuras experimentais para publicação científica.

Funções:
    - Conversão PNG -> PDF
    - Conversão e padronização PDF -> PDF
    - Ajuste de resolução
    - Remoção de bordas brancas
    - Padronização das dimensões das figuras
    - Aumento de contraste e nitidez
    - Exportação em alta qualidade

Entrada:
    results/figures/png/
    results/figures/source_pdf/

Saída:
    results/figures/pdf/
    results/figures/standardized_pdf/

Observação: Nesse script poderão ser adicionadas outras imagens que careçam de ajustes.

Autor:
    Priscila A. D. Nicácio
"""


from pathlib import Path

import fitz  # PyMuPDF
from PIL import (
    Image,
    ImageOps,
    ImageFilter,
    ImageEnhance,
    ImageChops,
)


# ==========================================================
# DIRETÓRIOS DO PROJETO
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    BASE_DIR
    / "results"
    / "figures"
    / "png"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "figures"
    / "pdf"
)

INPUT_PDF_DIR = (
    BASE_DIR
    / "results"
    / "figures"
    / "source_pdf"
)

OUTPUT_STANDARDIZED_DIR = (
    BASE_DIR
    / "results"
    / "figures"
    / "standardized_pdf"
)


# ==========================================================
# CONFIGURAÇÕES — PNG -> PDF
# ==========================================================

DPI_PNG = 600

WIDTH = 6400

CO2_HEIGHT = 3600
COMP_HEIGHT = 3200

BORDER_RATIO = 0.02


# ==========================================================
# CONFIGURAÇÕES — PDF -> PDF PADRONIZADO
# ==========================================================

DPI_PDF = 400

MARGEM_PX = 20

# Dimensões finais da figura:
# 16 cm x 10 cm

LARGURA_CM = 16
ALTURA_CM = 10

LARGURA_PX = int(
    LARGURA_CM / 2.54 * DPI_PDF
)

ALTURA_PX = int(
    ALTURA_CM / 2.54 * DPI_PDF
)


# ==========================================================
# FIGURAS PNG DE ENTRADA
# ==========================================================

ARQUIVOS_PNG = [
    (
        INPUT_DIR / "figco2.png",
        OUTPUT_DIR / "figco2.pdf"
    ),

    (
        INPUT_DIR / "figcomparativo.png",
        OUTPUT_DIR / "figcomparativo.pdf"
    ),

    (
        INPUT_DIR / "variabilidade.png",
        OUTPUT_DIR / "variabilidade.pdf"
    ),
]


#Adicionar mais imagens aqui, caso necessite;


# ==========================================================
# FIGURAS PDF DE ENTRADA
# ==========================================================
#
# Estes arquivos correspondem a figuras já existentes em PDF que precisam apenas ser padronizadas.
#
# Os PDFs originais devem ser colocados em:
#
# results/figures/source_pdf/
#
# Os arquivos processados serão salvos em:
#
# results/figures/standardized_pdf/
#
# ==========================================================

ARQUIVOS_PDF = [
    "SentidosCentral.pdf",
    "F1_padronizada.pdf",
    "F2_padronizada.pdf",
    "F3_padronizada.pdf",
    "F4_padronizada.pdf",
    "figsumo3x3.pdf",
    "figsumo4x3.pdf",
    "figsv.pdf",
    "figsvb.pdf",
    "fig2.pdf",
    "fig2b.pdf",
    "fig3x3.pdf",
    "fig4x3.pdf",
    "fig8.pdf",
    "fig8b.pdf",
    "fig12.pdf",
    "fig12a.pdf",
    "fig12b.pdf",
    "figco2.pdf",
    "figco24.pdf",
    "figcomparativo.pdf",
    "figheatmap.pdf",
    "figrec24.pdf",
    "figrelefotoeletrico.pdf",
]


# ==========================================================
# MELHORIA VISUAL
# ==========================================================

def melhorar_imagem(img):
    """
    Aplica ajustes visuais para melhorar a qualidade da figura.
    """

    # Ajuste automático de contraste
    img = ImageOps.autocontrast(img)

    # Aumento de nitidez
    img = img.filter(
        ImageFilter.SHARPEN
    )

    # Máscara de nitidez
    img = img.filter(
        ImageFilter.UnsharpMask(
            radius=2,
            percent=180,
            threshold=3
        )
    )

    # Aumento leve do contraste perceptual
    enhancer = ImageEnhance.Contrast(img)

    img = enhancer.enhance(
        1.15
    )

    return img


# ==========================================================
# AJUSTE DE RESOLUÇÃO
# ==========================================================

def ajustar_imagem(img, tipo="geral"):
    """
    Redimensiona e prepara a imagem para exportação em alta resolução.
    """

    img = melhorar_imagem(img)

    if tipo == "co2":

        target_size = (
            WIDTH,
            CO2_HEIGHT
        )

    else:

        target_size = (
            WIDTH,
            COMP_HEIGHT
        )

    # ------------------------------------------------------
    # Upscale usando interpolação Lanczos
    # ------------------------------------------------------

    img = img.resize(
        target_size,
        Image.Resampling.LANCZOS
    )

    # ------------------------------------------------------
    # Nitidez final após ampliação
    # ------------------------------------------------------

    img = img.filter(
        ImageFilter.UnsharpMask(
            radius=2,
            percent=220,
            threshold=2
        )
    )

    # ------------------------------------------------------
    # Margem branca
    # ------------------------------------------------------

    border_px = int(
        target_size[1]
        * BORDER_RATIO
    )

    img = ImageOps.expand(
        img,
        border=border_px,
        fill="white"
    )

    return img


# ==========================================================
# CONVERSÃO PNG -> PDF
# ==========================================================

def preparar_figura(input_path, output_path):
    """
    Realiza a preparação completa de uma figura PNG e sua conversão para PDF.
    """

    if not input_path.exists():

        print(
            f"Arquivo não encontrado:\n"
            f"{input_path}"
        )

        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nProcessando PNG: "
        f"{input_path.name}"
    )

    img = Image.open(
        input_path
    ).convert("RGB")

    # Identificação do tipo da figura
    if "co2" in input_path.name.lower():

        tipo = "co2"

    else:

        tipo = "comparativo"

    img = ajustar_imagem(
        img,
        tipo=tipo
    )

    img.save(
        output_path,
        "PDF",
        resolution=DPI_PNG
    )

    print(
        f"PDF gerado com sucesso:\n"
        f"{output_path}"
    )


# ==========================================================
# REMOÇÃO DE BORDAS BRANCAS
# ==========================================================

def cortar_branco(img, tolerancia=245):
    """
    Remove áreas brancas externas da imagem.

    Parâmetro:
        tolerancia:
            controla a sensibilidade da remoção das bordas.
    """

    img = img.convert(
        "RGB"
    )

    fundo = Image.new(
        "RGB",
        img.size,
        "white"
    )

    diferenca = ImageChops.difference(
        img,
        fundo
    )

    diferenca = ImageOps.grayscale(
        diferenca
    )

    diferenca = diferenca.point(
        lambda p:
        255
        if p < (255 - tolerancia)
        else 0
    )

    bbox = diferenca.getbbox()

    if bbox:

        return img.crop(
            bbox
        )

    return img


# ==========================================================
# PADRONIZAÇÃO PDF -> PDF
# ==========================================================

def padronizar_pdf(
    input_path,
    output_path
):
    """
    Converte a primeira página de um PDF em imagem, remove bordas brancas e padroniza suas dimensões para 16 x 10 cm a 400 dpi.
    """

    input_path = Path(
        input_path
    )

    output_path = Path(
        output_path
    )

    if not input_path.exists():

        print(
            f"Arquivo não encontrado:\n"
            f"{input_path}"
        )

        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nPadronizando PDF: "
        f"{input_path.name}"
    )

    # ------------------------------------------------------
    # Abre o PDF
    # ------------------------------------------------------

    doc = fitz.open(
        input_path
    )

    if len(doc) == 0:

        doc.close()

        raise ValueError(
            f"PDF sem páginas: "
            f"{input_path}"
        )

    # Utiliza a primeira página
    pagina = doc[0]

    # ------------------------------------------------------
    # Renderização
    # ------------------------------------------------------

    zoom = (
        DPI_PDF / 72
    )

    matriz = fitz.Matrix(
        zoom,
        zoom
    )

    pix = pagina.get_pixmap(
        matrix=matriz,
        alpha=False
    )

    img = Image.frombytes(
        "RGB",
        [
            pix.width,
            pix.height
        ],
        pix.samples
    )

    # ------------------------------------------------------
    # Remove bordas brancas reais
    # ------------------------------------------------------

    img = cortar_branco(
        img,
        tolerancia=245
    )

    # ------------------------------------------------------
    # Adiciona margem branca controlada
    # ------------------------------------------------------

    img = ImageOps.expand(
        img,
        border=MARGEM_PX,
        fill="white"
    )

    # ------------------------------------------------------
    # Redimensiona mantendo proporção
    # ------------------------------------------------------

    img.thumbnail(
        (
            LARGURA_PX,
            ALTURA_PX
        ),
        Image.Resampling.LANCZOS
    )

    # ------------------------------------------------------
    # Cria canvas padronizado
    # ------------------------------------------------------

    canvas = Image.new(
        "RGB",
        (
            LARGURA_PX,
            ALTURA_PX
        ),
        "white"
    )

    # ------------------------------------------------------
    # Centraliza a figura
    # ------------------------------------------------------

    x = (
        LARGURA_PX
        - img.width
    ) // 2

    y = (
        ALTURA_PX
        - img.height
    ) // 2

    canvas.paste(
        img,
        (x, y)
    )

    # ------------------------------------------------------
    # Salva PDF padronizado
    # ------------------------------------------------------

    canvas.save(
        output_path,
        "PDF",
        resolution=DPI_PDF
    )

    doc.close()

    print(
        f"PDF padronizado salvo em:\n"
        f"{output_path}"
    )


# ==========================================================
# PROCESSAMENTO DOS PDFs
# ==========================================================

def processar_pdfs():
    """
    Processa todos os PDFs definidos em ARQUIVOS_PDF.
    """

    for nome_arquivo in ARQUIVOS_PDF:

        entrada = (
            INPUT_PDF_DIR
            / nome_arquivo
        )

        saida = (
            OUTPUT_STANDARDIZED_DIR
            / nome_arquivo
        )

        try:

            padronizar_pdf(
                entrada,
                saida
            )

        except Exception as erro:

            print(
                f"\nERRO em "
                f"{nome_arquivo}: "
                f"{erro}"
            )


# ==========================================================
# EXECUÇÃO PRINCIPAL
# ==========================================================

def main():
    """
    Executa os processos de preparação das figuras.
    """

    print(
        "=" * 70
    )

    print(
        "PREPARAÇÃO DE FIGURAS PARA PUBLICAÇÃO"
    )

    print(
        "=" * 70
    )

    # ------------------------------------------------------
    # PNG -> PDF
    # ------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CONVERSÃO PNG -> PDF"
    )

    print(
        "=" * 70
    )

    for entrada, saida in ARQUIVOS_PNG:

        try:

            preparar_figura(
                entrada,
                saida
            )

        except Exception as erro:

            print(
                f"\nERRO em "
                f"{entrada.name}: "
                f"{erro}"
            )

    # ------------------------------------------------------
    # PDF -> PDF padronizado
    # ------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PADRONIZAÇÃO PDF -> PDF"
    )

    print(
        "=" * 70
    )

    processar_pdfs()

    # ------------------------------------------------------
    # Finalização
    # ------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PROCESSAMENTO FINALIZADO"
    )

    print(
        "=" * 70
    )

    print(
        f"\nPDFs gerados em:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        f"\nPDFs padronizados em:\n"
        f"{OUTPUT_STANDARDIZED_DIR}"
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    main()