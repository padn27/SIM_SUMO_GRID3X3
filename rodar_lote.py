import os
from sumolib import checkBinary
import traci
import numpy as np
from tabulate import tabulate
import xml.etree.ElementTree as ET
import json
import matplotlib.pyplot as plt

# ==============================
# Configurações gerais
# ==============================
CENARIOS_DIR = os.path.join(os.getcwd(), "cenarios")
SUMO_BINARY = checkBinary("sumo")

TLS_IDS = ["n00","n01","n02","n10","n11","n12","n20","n21","n22"]
RL_TLS = ["n11","n12"]

TLS_EDGES = {
    "n00": ["n01_n00","n10_n00"],
    "n01": ["n00_n01","n02_n01","n11_n01"],
    "n02": ["n01_n02","n12_n02"],
    "n10": ["n00_n10","n11_n10","n20_n10"],
    "n11": ["n01_n11","n10_n11","n12_n11","n21_n11"],
    "n12": ["n02_n12","n11_n12","n22_n12"],
    "n20": ["n10_n20","n21_n20"],
    "n21": ["n11_n21","n20_n21","n22_n21"],
    "n22": ["n12_n22","n21_n22"]
}

DELTA = 0.05
WINDOW = 7

# ==============================
# Funções auxiliares
# ==============================
def contar_veiculos(rou_file):
    tree = ET.parse(rou_file)
    root = tree.getroot()
    carros = sum(int(f.get("vehsPerHour",0)) for f in root.findall("flow") if f.get("type")=="car")
    onibus = sum(int(f.get("vehsPerHour",0)) for f in root.findall("flow") if f.get("type")=="bus")
    return carros, onibus, carros+onibus

def check_convergencia(recompensas, delta=DELTA, window=WINDOW):
    if len(recompensas)<window: 
        return False
    ultimas = recompensas[-window:]
    return all(r!=0 for r in ultimas) and (max(ultimas)-min(ultimas)<=delta)

def route_edges_exist(rou_file, net_file):
    tree_net = ET.parse(net_file)
    edges_validos = {e.get("id") for e in tree_net.getroot().findall("edge")}
    tree_rou = ET.parse(rou_file)
    for flow in tree_rou.getroot().findall("flow"):
        for edge in flow.get("edges", "").split():
            if edge not in edges_validos:
                print(f"ERRO: Flow {flow.get('id')} contém edge desconhecida '{edge}'")
                return False
    return True

# ==============================
# Carrega cenários válidos
# ==============================
cenarios_info = []
for scenario in sorted(os.listdir(CENARIOS_DIR)):
    scenario_path = os.path.join(CENARIOS_DIR, scenario)
    if not os.path.isdir(scenario_path):
        continue
    rou_file = os.path.join(scenario_path, f"{scenario}.rou.xml")
    cfg_file = os.path.join(scenario_path, "grid.sumocfg")
    net_file = os.path.join(scenario_path, "grid.net.xml")
    tls_config_file = os.path.join(scenario_path, "tls_config.json")
    if not (os.path.isfile(rou_file) and os.path.isfile(cfg_file) and os.path.isfile(net_file) and os.path.isfile(tls_config_file)):
        continue
    carros, onibus, total = contar_veiculos(rou_file)
    tipo = "baixo" if scenario.startswith("c1b") else "medio" if scenario.startswith("c1m") else "alto"
    cenarios_info.append((scenario, cfg_file, net_file, rou_file, carros, onibus, tipo))
    print(f"{scenario}: {carros} carros, {onibus} ônibus, total {total} veículos/h")

# ==============================
# Execução da simulação
# ==============================
resumo_final = []
recompensas_cenarios = {}  # salva recompensas RL de cada cenário

for scenario, cfg_file, net_file, rou_file, carros_total, onibus_total, tipo in cenarios_info:
    if not route_edges_exist(rou_file, net_file):
        print(f"Pulando {scenario} por rota inválida.\n")
        continue

    print(f"\nRodando {scenario} ({tipo}) ...")

    try:
        traci.close()
    except:
        pass

    try:
        traci.start([SUMO_BINARY,"-c",cfg_file])
    except Exception as e:
        traci.close()
        print(f"Erro ao iniciar SUMO para {scenario}: {e}\n")
        continue

    step = 0
    tempo_esp_car = {}
    tempo_esp_bus = {}
    tempo_esp_max = 0
    fila_total = 0

    fila_tls = {tls: [] for tls in TLS_IDS}
    espera_tls = {tls: [] for tls in TLS_IDS}
    tls_recompensas = {tls: [] for tls in RL_TLS}
    convergiu_step = None

    try:
        while traci.simulation.getMinExpectedNumber()>0:
            traci.simulationStep()
            step += 1

            for vid in traci.vehicle.getIDList():
                vtype = traci.vehicle.getTypeID(vid)
                waiting = traci.vehicle.getWaitingTime(vid)
                tempo_esp_max = max(tempo_esp_max, waiting)
                if vtype=="car":
                    tempo_esp_car[vid] = waiting
                else:
                    tempo_esp_bus[vid] = waiting

            fila_step = 0
            for tls in TLS_IDS:
                fila_step_tls = sum(traci.edge.getLastStepHaltingNumber(edge) for edge in TLS_EDGES[tls])
                fila_tls[tls].append(fila_step_tls)
                fila_step += fila_step_tls
                for edge in TLS_EDGES[tls]:
                    for v in traci.edge.getLastStepVehicleIDs(edge):
                        espera_tls[tls].append(traci.vehicle.getWaitingTime(v))
            fila_total += fila_step

            for tls in RL_TLS:
                lanes = traci.trafficlight.getControlledLanes(tls)
                total_queue = sum(sum(1 for v in traci.lane.getLastStepVehicleIDs(l) if traci.vehicle.getSpeed(v)<0.1) for l in lanes)
                total_wait = sum(sum(traci.vehicle.getWaitingTime(v) for v in traci.lane.getLastStepVehicleIDs(l)) for l in lanes)
                reward_step = np.tanh((1000 - (total_queue + total_wait))/500.0)
                cumulative = tls_recompensas[tls][-1]*0.99 + reward_step if tls_recompensas[tls] else reward_step
                tls_recompensas[tls].append(cumulative)

            if convergiu_step is None and all(check_convergencia(tls_recompensas[t]) for t in RL_TLS):
                convergiu_step = step
    finally:
        traci.close()

    tempo_esp_car_med = round(np.mean(list(tempo_esp_car.values())) if tempo_esp_car else 0,2)
    tempo_esp_bus_med = round(np.mean(list(tempo_esp_bus.values())) if tempo_esp_bus else 0,2)
    fila_media = round(fila_total/step,2) if step>0 else 0.0
    recompensa_media = round(np.mean([tls_recompensas[t][-1] for t in RL_TLS if tls_recompensas[t]]),2) if RL_TLS else 0
    convergiu = f"SIM (step {convergiu_step})" if convergiu_step else "NÃO"

    fila_media_tls = {tls: round(np.mean(fila_tls[tls]),2) for tls in TLS_IDS}
    tempo_esp_med_tls = {tls: round(np.mean(espera_tls[tls]),2) if espera_tls[tls] else 0.0 for tls in TLS_IDS}

    fila_tls_str = " | ".join(f"{tls}:{fila_media_tls[tls]}" for tls in TLS_IDS)
    tempo_tls_str = " | ".join(f"{tls}:{tempo_esp_med_tls[tls]}" for tls in TLS_IDS)

    resumo_final.append([
        scenario, tipo, carros_total, onibus_total, carros_total+onibus_total,
        carros_total, onibus_total,
        fila_media,
        tempo_esp_max, recompensa_media, convergiu,
        fila_tls_str, tempo_tls_str
    ])

    recompensas_cenarios[scenario] = tls_recompensas  # salva recompensas do cenário

# ==============================
# Impressão do resumo
# ==============================
headers = [
    "Cenário","Tipo","C","B","Total",
    "Ins","BIns",
    "FMed",
    "TMax","RRL","Conv",
    "F_TLS","TMed_TLS"
]
print("\nResumo global e por TLS dos cenários:\n")
print(tabulate(resumo_final, headers=headers, tablefmt="fancy_grid", numalign="right"))

# ==============================
# Plot das curvas de recompensa RL por TLS
# ==============================
plt.figure(figsize=(14,7))

for scenario in recompensas_cenarios:
    rewards_scenario = recompensas_cenarios[scenario]
    for tls in rewards_scenario:
        plt.plot(rewards_scenario[tls], label=f"{scenario} - {tls}")

plt.xlabel("Step")
plt.ylabel("Recompensa acumulada RL")
plt.title("Evolução da recompensa RL por TLS e cenário")
plt.legend()
plt.grid(True)
plt.show()

