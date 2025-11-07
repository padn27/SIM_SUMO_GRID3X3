# analise_teste.py

import os
import xml.etree.ElementTree as ET
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

def contar_veiculos_agregados(caminho_arquivo):
    if not os.path.isfile(caminho_arquivo):
        return 0
    try:
        tree = ET.parse(caminho_arquivo)
        root = tree.getroot()
        total_veiculos = 0
        for interval in root.findall('interval'):
            total_veiculos += int(interval.get('nVehContrib', 0))
        return total_veiculos
    except (ET.ParseError, ValueError):
        return 0

def imprimir_resumo(resumo_final):
    headers = [
        "Cenário", "Tipo", "Carros", "Ônibus", "Fila Média", "Esp Média", 
        "Esp Máx", "Veículos Saída (n11)", "Recompensa Média RL", "Convergiu", 
        "Filas por TLS", "Espera por TLS"
    ]
    print("\n" + "="*80)
    print("Resumo Final da Execução em Lote")
    print("="*80)
    print(tabulate(resumo_final, headers=headers, tablefmt="fancy_grid", numalign="right"))
    print("="*80)

def calcular_tempo_medio_viagem(caminho_arquivo):
    if not os.path.isfile(caminho_arquivo):
        return 0
    try:
        tree = ET.parse(caminho_arquivo)
        root = tree.getroot()
        # Encontra todos os elementos 'tripinfo' e pega o valor do atributo 'duration'
        durations = [float(trip.get('duration')) for trip in root.findall('tripinfo')]
        if not durations:
            return 0
        return round(np.mean(durations), 2)
    except (ET.ParseError, ValueError):
        return 0

#def plotar_recompensas(recompensas_cenarios, modo_validacao):

