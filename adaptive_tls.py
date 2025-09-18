import traci
import matplotlib.pyplot as plt
import numpy as np
import math  

# -----------------------------
# Configuração SUMO
# -----------------------------
SUMO_BINARY = "sumo-gui"
SUMO_CONFIG = "grid.sumocfg"

# -----------------------------
# Definição de TLS por tipo
# -----------------------------
fixed_tls = ["n00", "n02", "n20", "n12", "n22"]  # semáforos fixos (não adaptativos)
adaptive_tls = ["n01", "n10", "n21"]            # semáforos adaptativos sem RL
rl_tls = ["n11"]                                 # semáforo adaptativo com RL (recompensa)

rl_focus = ["n11"]  # apenas monitora a recompensa do n11

# -----------------------------
# Inicializa simulação SUMO
# -----------------------------
traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])
tls_ids = list(traci.trafficlight.getIDList())
print("TLS encontrados na rede:", tls_ids)

# -----------------------------
# Parâmetros gerais da simulação
# -----------------------------
MAX_STEPS = 3600       # duração total da simulação em steps
MIN_GREEN = 10         # tempo mínimo de verde
MAX_GREEN = 90         # tempo máximo de verde
BUS_PRIORITY = 17      # peso para veículos prioritários (ônibus)
DETECTOR_RANGE = 15    # distância antes do semáforo para contagem de veículos

# -----------------------------
# Dicionários para armazenar métricas de cada semáforo
# -----------------------------
metrics = {tls_id: {"total_queue": [], "total_wait": [], "phase_changes": 0} for tls_id in tls_ids}
last_green_phase = {tls_id: 0 for tls_id in tls_ids}

# -----------------------------
# Inicializa recompensas RL
# -----------------------------
cumulative_reward = 0       # recompensa acumulada do semáforo n11
reward_history = []         # histórico da recompensa acumulada
moving_avg_history = []     # média móvel da recompensa
MOVING_WINDOW = 300         # janela da média móvel
PLOT_UPDATE = 10            # atualizar gráfico (a cada 10 steps)

# -----------------------------
# Inicializa gráfico interativo
# -----------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(12,6))
line1, = ax.plot([], [], color='lightblue', label='Recompensa acumulada')
line2, = ax.plot([], [], color='red', label='Média móvel')
ax.set_xlabel("Step")
ax.set_ylabel("Recompensa")
ax.set_title("Curva de Recompensa RL - cruzamento n11 (central)")
ax.legend()
ax.grid(True)
plt.show()

# -----------------------------
# Loop principal
# -----------------------------
try:
    for step in range(MAX_STEPS):
        traci.simulationStep()  # avança a simulação (um passo)

        for tls_id in tls_ids:
            try:
                # Obtém fases, cores, duração do semáforo
                tls_programs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
                if not tls_programs:
                    continue
                tls_program = tls_programs[0]

                # Lanes controladas pelo semáforo
                lanes = traci.trafficlight.getControlledLanes(tls_id)
                if not lanes:
                    continue

                # -----------------------------
                # Conta trocas de fase
                # -----------------------------
                current_phase = traci.trafficlight.getPhase(tls_id)
                if 'last_phase' not in metrics[tls_id]:
                    metrics[tls_id]['last_phase'] = current_phase
                elif current_phase != metrics[tls_id]['last_phase']:
                    metrics[tls_id]['phase_changes'] += 1
                    metrics[tls_id]['last_phase'] = current_phase

                # -----------------------------
                # Conta congestionamento por fase verde
                # -----------------------------
                congestion_per_phase = [0] * len(tls_program.phases)
                green_phases = []

                for i, phase in enumerate(tls_program.phases):
                    state = phase.state
                    if "G" not in state:
                        continue  # ignora fases sem verde total
                    green_phases.append(i)
                    count = 0
                    for lane_index, lane in enumerate(lanes):
                        if lane_index >= len(state):
                            continue
                        if state[lane_index] == "G":
                            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                            for v in vehicle_ids:
                                pos = traci.vehicle.getLanePosition(v)
                                if pos >= traci.lane.getLength(lane) - DETECTOR_RANGE:
                                    if traci.vehicle.getTypeID(v) == "car":
                                        count += 1
                                    elif traci.vehicle.getTypeID(v) == "bus":
                                        count += BUS_PRIORITY
                    congestion_per_phase[i] = count

                # -----------------------------
                # Aplica fase amarela e verde
                # -----------------------------
                apply_adaptive = tls_id in adaptive_tls + rl_tls
                if green_phases:
                    max_index = max(green_phases, key=lambda x: congestion_per_phase[x])
                    duration = min(MAX_GREEN, max(MIN_GREEN, 10 + congestion_per_phase[max_index]*2))

                    yellow_phase_index = None
                    for i, phase in enumerate(tls_program.phases):
                        if "y" in phase.state.lower() and i != current_phase:
                            yellow_phase_index = i
                            break
                    if yellow_phase_index is not None and yellow_phase_index != current_phase:
                        traci.trafficlight.setPhase(tls_id, yellow_phase_index)
                        traci.simulationStep()

                    if apply_adaptive:
                        traci.trafficlight.setPhase(tls_id, max_index)
                        traci.trafficlight.setPhaseDuration(tls_id, duration)
                        last_green_phase[tls_id] = max_index
                    else:
                        traci.trafficlight.setPhase(tls_id, max_index)

                # -----------------------------
                # Métricas de fila e espera
                # -----------------------------
                total_queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
                total_wait = sum(traci.lane.getWaitingTime(l) for l in lanes)
                metrics[tls_id]["total_queue"].append(total_queue)
                metrics[tls_id]["total_wait"].append(total_wait)

                # -----------------------------
                # Recompensa RL acumulativa para n11
                # -----------------------------
                if tls_id in rl_focus:
                    reward_step = 1000 - (total_queue + total_wait)
                    reward_step = np.tanh(reward_step / 500.0)  # normaliza

                    # Atualiza cumulative_reward com decaimento para convergência
                    cumulative_reward = cumulative_reward * 0.99 + reward_step
                    reward_history.append(cumulative_reward)

                    # Média móvel
                    if len(reward_history) >= MOVING_WINDOW:
                        moving_avg = sum(reward_history[-MOVING_WINDOW:]) / MOVING_WINDOW
                    else:
                        moving_avg = cumulative_reward
                    moving_avg_history.append(moving_avg)

            except Exception as e:
                print(f"[ERRO] TLS {tls_id}: {e}")
                continue

        # -----------------------------
        # Atualiza gráfico a cada PLOT_UPDATE steps
        # -----------------------------
        if step % PLOT_UPDATE == 0:
            line1.set_data(range(len(reward_history)), reward_history)
            line2.set_data(range(len(moving_avg_history)), moving_avg_history)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

except traci.FatalTraCIError as e:
    print(f"Erro crítico na simulação no step {step}: {e}")

finally:
    # -----------------------------
    # Fecha SUMO e gráfico
    # -----------------------------
    traci.close()
    plt.ioff()
    plt.show()
    print("SUMO finalizado")

    # -----------------------------
# Exibe métricas finais detalhadas por TLS
# -----------------------------
def print_metrics(tls_list, tipo):
    print(f"\n===== Métricas {tipo} =====")
    print("TLS\tFila\tEsp\tFases")
    for tls in tls_list:
        fila = metrics[tls]["total_queue"][-1] if metrics[tls]["total_queue"] else 0
        esp = metrics[tls]["total_wait"][-1] if metrics[tls]["total_wait"] else 0
        fases = metrics[tls]["phase_changes"]
        print(f"{tls}\t{fila:.1f}\t{esp:.1f}\t{fases}")

# Imprime métricas separadas por tipo
print_metrics(rl_tls, "RL")
print_metrics(adaptive_tls, "Adaptativo")
print_metrics(fixed_tls, "Fixo")

# -----------------------------
# Também podemos mostrar médias por tipo (opcional)
# -----------------------------
def summarize_tls(tls_list):
    filas = []
    espera = []
    fases = []
    for tls in tls_list:
        filas.append(metrics[tls]["total_queue"][-1] if metrics[tls]["total_queue"] else 0)
        espera.append(metrics[tls]["total_wait"][-1] if metrics[tls]["total_wait"] else 0)
        fases.append(metrics[tls]["phase_changes"])
    filas_avg = np.mean(filas)
    espera_avg = np.mean(espera)
    fases_sum = np.sum(fases)
    return filas_avg, espera_avg, fases_sum

filas_rl, espera_rl, fases_rl = summarize_tls(rl_tls)
filas_ad, espera_ad, fases_ad = summarize_tls(adaptive_tls)
filas_fx, espera_fx, fases_fx = summarize_tls(fixed_tls)

print("\n===== MÉTRICAS FINAIS (Resumo) =====")
print("Tipo\tFila\tEsp\tFases")
print(f"RL\t{filas_rl:.1f}\t{espera_rl:.1f}\t{fases_rl}")
print(f"AD\t{filas_ad:.1f}\t{espera_ad:.1f}\t{fases_ad}")
print(f"FX\t{filas_fx:.1f}\t{espera_fx:.1f}\t{fases_fx}")



