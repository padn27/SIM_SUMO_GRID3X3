import traci
import matplotlib.pyplot as plt
import numpy as np

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

# -----------------------------
# Dicionários para armazenar métricas de cada semáforo
# -----------------------------
metrics = {tls_id: {"total_queue": [], "total_wait": [], "phase_changes": 0} for tls_id in tls_ids}
last_green_phase = {tls_id: 0 for tls_id in tls_ids}

# -----------------------------
# Contagem de fluxo (entradas/saídas/balanceamento)
# -----------------------------
flow_balance = {tls_id: {"in": 0, "out": 0, "balance": []} for tls_id in tls_ids}

# -----------------------------
# Inicializa recompensas RL
# -----------------------------
cumulative_reward = 0       # recompensa total acumulada
reward_history = []         # histórico de recompensa step a step
moving_avg_history = []     # média móvel curta
MOVING_WINDOW_SHORT = 50    # média móvel curta para visualização
PLOT_UPDATE = 10            # atualizar gráfico a cada 10 steps

# -----------------------------
# Inicializa gráfico interativo
# -----------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(12,6))
line1, = ax.plot([], [], color='lightblue', label='Recompensa step')
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
                tls_programs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
                if not tls_programs:
                    continue
                tls_program = tls_programs[0]

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
                        continue
                    green_phases.append(i)
                    count = 0
                    for lane_index, lane in enumerate(lanes):
                        if lane_index >= len(state):
                            continue
                        if state[lane_index] == "G":
                            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                            count += sum(1 for v in vehicle_ids if traci.vehicle.getSpeed(v) < 0.1)
                            count += sum(BUS_PRIORITY for v in vehicle_ids if traci.vehicle.getTypeID(v) == "bus")
                    congestion_per_phase[i] = count

                # -----------------------------
                # Ajuste adaptativo e balanceamento
                # -----------------------------
                apply_adaptive = tls_id in adaptive_tls + rl_tls
                if green_phases:
                    max_index = max(green_phases, key=lambda x: congestion_per_phase[x])
                    base_duration = 10 + congestion_per_phase[max_index]*2

                    if apply_adaptive:
                        in_count = sum(len(traci.lane.getLastStepVehicleIDs(l)) for l in lanes)
                        out_count = sum(1 for l in lanes for v in traci.lane.getLastStepVehicleIDs(l) if traci.vehicle.getLaneID(v) != l)
                        flow_balance[tls_id]["in"] += in_count
                        flow_balance[tls_id]["out"] += out_count
                        net_balance = flow_balance[tls_id]["in"] - flow_balance[tls_id]["out"]

                        adjusted_duration = base_duration + int(net_balance * 0.1)
                        green_duration = max(MIN_GREEN, min(MAX_GREEN, adjusted_duration))

                        traci.trafficlight.setPhase(tls_id, max_index)
                        traci.trafficlight.setPhaseDuration(tls_id, green_duration)
                        last_green_phase[tls_id] = max_index
                    else:
                        traci.trafficlight.setPhase(tls_id, max_index)

                # -----------------------------
                # Métricas de fila e espera
                # -----------------------------
                total_queue = 0
                total_wait = 0
                for lane in lanes:
                    vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                    total_queue += sum(1 for v in vehicle_ids if traci.vehicle.getSpeed(v) < 0.1)
                    total_wait += sum(traci.vehicle.getWaitingTime(v) for v in vehicle_ids)

                metrics[tls_id]["total_queue"].append(total_queue)
                metrics[tls_id]["total_wait"].append(total_wait)

                # -----------------------------
                # Recompensa RL ajustada (sem suavização forte)
                # -----------------------------
                if tls_id in rl_focus:
                    # Penalidade congestionamento e veículos prioritários
                    bus_penalty = sum(BUS_PRIORITY for lane in lanes for v in traci.lane.getLastStepVehicleIDs(lane) if traci.vehicle.getTypeID(v) == "bus")
                    in_count = sum(len(traci.lane.getLastStepVehicleIDs(l)) for l in lanes)
                    out_count = sum(1 for l in lanes for v in traci.lane.getLastStepVehicleIDs(l) if traci.vehicle.getLaneID(v) != l)
                    net_flow_penalty = in_count - out_count

                    reward_step = - (total_queue + total_wait + bus_penalty + net_flow_penalty)
                    reward_history.append(reward_step)

                    # Média móvel curta para gráfico
                    if len(reward_history) >= MOVING_WINDOW_SHORT:
                        moving_avg = np.mean(reward_history[-MOVING_WINDOW_SHORT:])
                    else:
                        moving_avg = reward_step
                    moving_avg_history.append(moving_avg)

                    cumulative_reward += reward_step

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
    traci.close()
    plt.ioff()
    plt.show()
    print("SUMO finalizado")

# -----------------------------
# Exibe métricas finais detalhadas por TLS
# -----------------------------
def print_metrics(tls_list, tipo):
    print(f"\n===== Métricas {tipo} =====")
    print("TLS\tFila\tEsp\tFases\tNetFlow")
    for tls in tls_list:
        fila = metrics[tls]["total_queue"][-1] if metrics[tls]["total_queue"] else 0
        esp = metrics[tls]["total_wait"][-1] if metrics[tls]["total_wait"] else 0
        fases = metrics[tls]["phase_changes"]
        net = flow_balance[tls]["balance"][-1] if flow_balance[tls]["balance"] else 0
        print(f"{tls}\t{fila:.1f}\t{esp:.1f}\t{fases}\t{net}")

print_metrics(rl_tls, "RL")
print_metrics(adaptive_tls, "Adaptativo")
print_metrics(fixed_tls, "Fixo")

# -----------------------------
# Resumo por tipo
# -----------------------------
def summarize_tls(tls_list):
    filas = []
    espera = []
    fases = []
    nets = []

