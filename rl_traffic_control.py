import traci
import traci.constants as tc
import sumolib
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time

# -----------------------------
# PARÂMETROS SUMO
# -----------------------------
SUMO_BINARY = "sumo-gui"  # Executável SUMO com GUI
SUMO_CONFIG = "/home/priscila/sumo_grid/grid.sumocfg"  # Arquivo de configuração da rede
NUM_STEPS = 3000  # Número de steps da simulação
tls_ids = ["n00","n01","n02","n10","n11","n12","n20","n21","n22"]  # IDs de semáforos

# -----------------------------
# PARÂMETROS RL
# -----------------------------
STATE_SIZE = 8  # Estado: ns_queue, ew_queue, ns_wait, ew_wait, fase_atual, fase_sugerida, duracao_sugerida, verde_restante
ACTION_SIZE = 4  # 0=verde curto, 1=verde médio, 2=verde longo, 3=trocar fase
BATCH_SIZE = 64
GAMMA = 0.99  # Fator de desconto
EPSILON_START = 1.0  # Inicial exploração
EPSILON_END = 0.1  # Exploração mínima
EPSILON_DECAY = 0.995  # Decaimento da exploração
LR = 0.001  # Taxa de aprendizado
MEMORY_SIZE = 10000  # Capacidade do Replay Buffer
WINDOW_MOVING_AVG = 300  # Média móvel para plotagem da recompensa

# -----------------------------
# REDE NEURAL DQN
# -----------------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # Rede simples com duas camadas ocultas de 64 neurônios
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Saída: Q-values para cada ação

# -----------------------------
# REPLAY BUFFER
# -----------------------------
class ReplayBuffer:
    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.memory)

# -----------------------------
# FINALIZA SUMO ANTIGO
# -----------------------------
os.system("pkill -f sumo")
time.sleep(1)

# -----------------------------
# INICIA SUMO
# -----------------------------
try:
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])
    time.sleep(1)
    print("SUMO iniciado com sucesso")
except Exception as e:
    print("Erro ao iniciar SUMO:", e)
    sys.exit(1)

# -----------------------------
# INICIALIZA AGENTES E MÉTRICAS
# -----------------------------
agents = {}
# Métricas de cada semáforo
metrics_rl = {tls_id: {"total_queue": [], "total_wait": [], "phase_changes": 0,
                       "vehicles_passed": 0, "reward_history": []} for tls_id in tls_ids}
simulation_log = {tls_id: {"states": [], "actions": [], "rewards": []} for tls_id in tls_ids}

for tls_id in tls_ids:
    model = DQN(STATE_SIZE, ACTION_SIZE)
    agents[tls_id] = {
        "current_phase": 0,  # Fase atual do semáforo (0=NS,1=EW)
        "phase_time": 0,
        "verde_restante": 0,  # Contador de duração do verde
        "reward_history": [],
        "phase_changes": 0,
        "memory": ReplayBuffer(MEMORY_SIZE),
        "model": model,
        "optimizer": optim.Adam(model.parameters(), lr=LR),
        "epsilon": EPSILON_START  # Inicialmente explora bastante
    }

# -----------------------------
# FUNÇÕES AUXILIARES
# -----------------------------

# Computa o estado adaptativo do semáforo considerando filas, ônibus, etc.
def compute_adaptive_state(tls_id, bus_priority=10, min_green=10, max_green=60):
    tls_program = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    congestion_per_phase = [0]*len(tls_program.phases)
    green_phases = []

    # Avalia cada fase do semáforo
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
                vehicles = traci.lane.getLastStepVehicleIDs(lane)
                car_count = len([v for v in vehicles if traci.vehicle.getTypeID(v)=="car"])
                bus_count = len([v for v in vehicles if traci.vehicle.getTypeID(v)=="bus"])
                # Aumenta peso para ônibus
                count += car_count + bus_count*bus_priority
        congestion_per_phase[i] = count

    if green_phases:
        # Seleciona a fase com maior congestionamento
        best_phase = max(green_phases, key=lambda x: congestion_per_phase[x])
        duration = min(max_green, max(min_green, 10 + congestion_per_phase[best_phase]*2))
        return best_phase, duration
    else:
        return 0, 30

# Separa lanes por direção (NS, EW, outras)
def get_lanes_by_phase(tls_id):
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    if tls_id in ["n00","n02","n20","n22"]:
        return lanes[:2], lanes[2:4], []
    elif tls_id in ["n01","n10","n12","n21"]:
        return lanes[:2], lanes[2:4], lanes[4:]
    elif tls_id == "n11":
        return lanes[:2]+lanes[4:6], lanes[2:4]+lanes[6:8], []
    else:
        mid = len(lanes)//2
        return lanes[:mid], lanes[mid:], []

# Cria vetor de estado para o RL
def get_state(tls_id):
    NS_lanes, EW_lanes, other_lanes = get_lanes_by_phase(tls_id)
    ns_queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in NS_lanes)
    ew_queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in EW_lanes)
    ns_wait = sum(traci.lane.getWaitingTime(lane) for lane in NS_lanes)
    ew_wait = sum(traci.lane.getWaitingTime(lane) for lane in EW_lanes)
    phase = agents[tls_id]["current_phase"]
    best_phase, suggested_duration = compute_adaptive_state(tls_id)
    verde_restante = agents[tls_id]["verde_restante"]
    # Estado contém informações de fila, espera, fase atual e sugerida, verde restante
    return np.array([ns_queue, ew_queue, ns_wait, ew_wait, phase, best_phase, suggested_duration, verde_restante],
                    dtype=np.float32)

# Executa ação escolhida pelo agente
def take_action(tls_id, action):
    agent = agents[tls_id]
    short, medium, long = 20, 30, 45  # tempos em steps

    if action == 0:
        agent["verde_restante"] = short
    elif action == 1:
        agent["verde_restante"] = medium
    elif action == 2:
        agent["verde_restante"] = long
    elif action == 3:
        # Troca de fase
        agent["current_phase"] = 1 - agent["current_phase"]
        traci.trafficlight.setPhase(tls_id, agent["current_phase"])
        agent["verde_restante"] = short
        agent["phase_changes"] += 1
        metrics_rl[tls_id]["phase_changes"] += 1

# Calcula recompensa do agente
def compute_reward(tls_id, step):
    NS_lanes, EW_lanes, other_lanes = get_lanes_by_phase(tls_id)
    
    # Número de veículos em cada direção
    ns_vehicles = sum(len(traci.lane.getLastStepVehicleIDs(lane)) for lane in NS_lanes)
    ew_vehicles = sum(len(traci.lane.getLastStepVehicleIDs(lane)) for lane in EW_lanes)
    other_vehicles = sum(len(traci.lane.getLastStepVehicleIDs(lane)) for lane in other_lanes)
    total_vehicles = ns_vehicles + ew_vehicles + other_vehicles

    # Tempo de espera acumulado
    ns_wait = sum(traci.lane.getWaitingTime(lane) for lane in NS_lanes)
    ew_wait = sum(traci.lane.getWaitingTime(lane) for lane in EW_lanes)
    other_wait = sum(traci.lane.getWaitingTime(lane) for lane in other_lanes)
    total_wait = ns_wait + ew_wait + other_wait

    # Número de veículos que completaram a simulação (passaram pelo TLS)
    passed = traci.simulation.getArrivedNumber()

    # Bônus específico para ônibus, incentivando prioridade
    bus_passed = sum(len([v for v in traci.lane.getLastStepVehicleIDs(lane)
                         if traci.vehicle.getTypeID(v)=="bus"])
                     for lane in NS_lanes+EW_lanes)

    # Normalização para evitar valores muito grandes
    norm_queue = np.tanh(total_vehicles / 10.0)  # fila normalizada
    norm_wait = np.tanh(total_wait / 100.0)      # espera normalizada

    # Coeficientes escolhidos empiricamente para balancear recompensas e penalidades
    alpha, beta, gamma, delta = 15.0, 4.0, 3.0, 30.0
    
    # Recompensa positiva: veículos que passaram (meta principal do TLS)
    passed_reward = alpha * passed
    
    # Penalidade por filas grandes (evita congestionamento)
    queue_penalty = - beta * norm_queue
    
    # Penalidade por tempo de espera longo (conforto do tráfego)
    wait_penalty = - gamma * norm_wait
    
    # Recompensa extra para ônibus (prioridade)
    bus_reward = delta * bus_passed

    # Recompensa base combinando todos os fatores
    base_reward = passed_reward + queue_penalty + wait_penalty + bus_reward

    # Curva sigmoide para crescimento gradual da recompensa ao longo dos steps
    growth = 1 / (1 + np.exp(-(step-300)/200))
    
    # Offset garante que a recompensa não fique negativa (ajuste de estabilidade)
    offset = abs(min(0, base_reward)) + 0.1
    
    # Recompensa final aplicada ao agente
    reward = (base_reward + offset) * growth

    # Atualiza histórico para análise
    agents[tls_id]["reward_history"].append(reward)
    metrics_rl[tls_id]["total_queue"].append(total_vehicles)
    metrics_rl[tls_id]["total_wait"].append(total_wait)
    metrics_rl[tls_id]["vehicles_passed"] += passed
    
    return reward


# Treina a rede usando replay buffer
def train(agent):
    if len(agent["memory"]) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = agent["memory"].sample(BATCH_SIZE)
    states = torch.tensor(states, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Q-learning update
    q_values = agent["model"](states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        q_next = agent["model"](next_states).max(1)[0]
        q_target = rewards + GAMMA * q_next * (1 - dones)

    loss = nn.MSELoss()(q_values, q_target)
    agent["optimizer"].zero_grad()
    loss.backward()
    agent["optimizer"].step()

# Log detalhado por step (para análise acadêmica)
def log_step(tls_id, step, state, action, reward):
    action_map = {0:"Verde curto",1:"Verde médio",2:"Verde longo",3:"Trocar fase"}
    phase_map = {0:"NS",1:"EW"}
    NS_queue, EW_queue, NS_wait, EW_wait, phase, best_phase, sug_dur, verde_rest = state
    print(f"Step {step:04d} | TLS {tls_id} | Fase atual: {phase_map[int(phase)]} | Verde restante: {int(verde_rest)} | "
          f"NS: {int(NS_queue)} carros / {int(NS_wait)} s espera | EW: {int(EW_queue)} carros / {int(EW_wait)} s espera | "
          f"Ação: {action_map[action]} | Recompensa: {reward:.2f}")

# -----------------------------
# LOOP PRINCIPAL
# -----------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(12,6))
line1, = ax.plot([], [], color='lightblue', label='Recompensa por step')
line2, = ax.plot([], [], color='red', label='Média móvel')
ax.set_xlabel("Step")
ax.set_ylabel("Recompensa")
ax.set_title("Curva de Recompensa RL")
ax.legend()
plt.show()

rewards_all = []
moving_avg_all = []

for step in range(NUM_STEPS):
    step_rewards = []
    for tls_id, agent in agents.items():
        # Obtem estado atual
        state = get_state(tls_id)
        # Escolhe ação: exploração ou política
        if random.random() < agent["epsilon"]:
            action = random.choice([0,1,2,3])
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = agent["model"](state_tensor)
            action = torch.argmax(q_values).item()

        take_action(tls_id, action)  # Executa ação no SUMO
        reward = compute_reward(tls_id, step)  # Calcula recompensa
        next_state = get_state(tls_id)
        done = False
        agent["memory"].add(state, action, reward, next_state, done)
        train(agent)  # Treina rede DQN
        agent["epsilon"] = max(EPSILON_END, agent["epsilon"]*EPSILON_DECAY)  # Decaimento epsilon
        step_rewards.append(reward)

        simulation_log[tls_id]["states"].append(state)
        simulation_log[tls_id]["actions"].append(action)
        simulation_log[tls_id]["rewards"].append(reward)
        log_step(tls_id, step, state, action, reward)

    # Atualiza recompensas globais
    total_reward = sum(step_rewards)
    rewards_all.append(total_reward)
    if step >= WINDOW_MOVING_AVG:
        moving_avg_all.append(np.mean(rewards_all[-WINDOW_MOVING_AVG:]))
    else:
        moving_avg_all.append(np.mean(rewards_all[:step+1]))

    traci.simulationStep()  # Avança SUMO
    line1.set_data(range(len(rewards_all)), rewards_all)
    line2.set_data(range(len(moving_avg_all)), moving_avg_all)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)

# -----------------------------
# RELATÓRIO FINAL
# -----------------------------
print("\n--- RELATÓRIO FINAL DA SIMULAÇÃO ---\n")
for tls_id in tls_ids:
    print(f"TLS {tls_id}:")
    print(f"  Total de mudanças de fase: {metrics_rl[tls_id]['phase_changes']}")
    print(f"  Veículos passados acumulados: {metrics_rl[tls_id]['vehicles_passed']}")
    print(f"  Total de fila final: {metrics_rl[tls_id]['total_queue'][-1]}")
    print(f"  Total de espera final: {metrics_rl[tls_id]['total_wait'][-1]}")
