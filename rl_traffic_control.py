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
SUMO_BINARY = "sumo-gui"  # Executável SUMO com interface gráfica (GUI)
SUMO_CONFIG = "/home/priscila/sumo_grid/grid.sumocfg"  # Arquivo de configuração da rede SUMO
NUM_STEPS = 3000  # Número total de steps da simulação
tls_ids = ["n00","n01","n02","n10","n11","n12","n20","n21","n22"]  # IDs dos semáforos controlados

# -----------------------------
# PARÂMETROS DE REINFORCEMENT LEARNING (RL)
# -----------------------------
STATE_SIZE = 8  # Dimensão do vetor de estado do agente DQN
# Vetor de estado (posição no array → significado):
# 0: ns_queue          → Número de veículos parados nas pistas Norte-Sul (NS)
# 1: ew_queue          → Número de veículos parados nas pistas Leste-Oeste (EW)
# 2: ns_wait           → Tempo total de espera acumulado nas pistas NS
# 3: ew_wait           → Tempo total de espera acumulado nas pistas EW
# 4: fase_atual        → Fase atual do semáforo (0 = NS verde, 1 = EW verde)
# 5: fase_sugerida     → Fase sugerida pelo controle adaptativo com base na fila atual
# 6: duracao_sugerida  → Duração sugerida do verde para a fase_sugerida (em steps)
# 7: verde_restante    → Tempo restante do verde na fase atual (contagem regressiva)
ACTION_SIZE = 4  # Número de ações possíveis: 0=verde curto, 1=verde médio, 2=verde longo, 3=trocar fase
BATCH_SIZE = 32  # Tamanho do batch para treinamento DQN
GAMMA = 0.99  # Fator de desconto de recompensa futura
EPSILON_START = 1.0  # Probabilidade inicial de explorar
EPSILON_END = 0.1  # Probabilidade mínima de explorar
EPSILON_DECAY = 0.999  # Decaimento de epsilon a cada step
LR = 0.001  # Taxa de aprendizado do otimizador (Adam)
MEMORY_SIZE = 20000  # Capacidade do Replay Buffer
WINDOW_MOVING_AVG = 800  # Janela da média móvel para plotar a recompensa suavizada

# -----------------------------
# REDE NEURAL DQN
# -----------------------------
class DQN(nn.Module):
    """
    Rede neural usada para aproximar a função Q
    Entrada: vetor de estado do semáforo
    Saída: Q-values para cada ação possível
    """
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)  # Primeira camada oculta
        self.fc2 = nn.Linear(64, 64)  # Segunda camada oculta
        self.fc3 = nn.Linear(64, action_size)  # Camada de saída: Q-values

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Saída sem ativação (Q-values brutos)

# -----------------------------
# REPLAY BUFFER
# -----------------------------
class ReplayBuffer:
    """
    Estrutura de memória para armazenar experiências do agente (s,a,r,s',done)
    """
    def __init__(self, size):
        self.memory = deque(maxlen=size)  # Mantém apenas os últimos 'size' elementos

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # Adiciona experiência

    def sample(self, batch_size):
        """
        Retorna um batch aleatório de experiências para treinamento
        """
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.memory)

# -----------------------------
# FINALIZA SUMO ANTIGO
# -----------------------------
# Evita conflitos com instâncias SUMO abertas anteriormente
os.system("pkill -f sumo")
time.sleep(1)

# -----------------------------
# INICIA SUMO
# -----------------------------
try:
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])  # Inicializa SUMO via TraCI
    time.sleep(1)
    print("SUMO iniciado com sucesso")
except Exception as e:
    print("Erro ao iniciar SUMO:", e)
    sys.exit(1)

# -----------------------------
# INICIALIZA AGENTES E MÉTRICAS
# -----------------------------
agents = {}
# Métricas de interesse para cada TLS
metrics_rl = {tls_id: {"total_queue": [], "total_wait": [], "phase_changes": 0,
                       "vehicles_passed": 0, "reward_history": []} for tls_id in tls_ids}
# Log detalhado da simulação (para análise acadêmica)
simulation_log = {tls_id: {"states": [], "actions": [], "rewards": []} for tls_id in tls_ids}

# Inicializa agentes DQN para cada semáforo
for tls_id in tls_ids:
    model = DQN(STATE_SIZE, ACTION_SIZE)
    agents[tls_id] = {
        "current_phase": 0,  # Fase inicial (0=NS, 1=EW)
        "phase_time": 0,
        "verde_restante": 0,  # Contador de tempo restante do verde
        "reward_history": [],
        "phase_changes": 0,
        "memory": ReplayBuffer(MEMORY_SIZE),
        "model": model,
        "optimizer": optim.Adam(model.parameters(), lr=LR),
        "epsilon": EPSILON_START  # Inicialmente alta exploração
    }

# -----------------------------
# FUNÇÕES AUXILIARES
# -----------------------------

def compute_adaptive_state(tls_id, bus_priority=10, min_green=10, max_green=60):
    """
    Computa a fase sugerida e a duração do verde com base na fila atual
    e prioridade de ônibus
    """
    tls_program = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    congestion_per_phase = [0]*len(tls_program.phases)
    green_phases = []

    for i, phase in enumerate(tls_program.phases):
        state = phase.state
        if "G" not in state:
            continue  # Ignora fases sem verde
        green_phases.append(i)
        count = 0
        for lane_index, lane in enumerate(lanes):
            if lane_index >= len(state):
                continue
            if state[lane_index] == "G":
                vehicles = traci.lane.getLastStepVehicleIDs(lane)
                car_count = len([v for v in vehicles if traci.vehicle.getTypeID(v)=="car"])
                bus_count = len([v for v in vehicles if traci.vehicle.getTypeID(v)=="bus"])
                count += car_count + bus_count*bus_priority  # Peso extra para ônibus
        congestion_per_phase[i] = count

    if green_phases:
        best_phase = max(green_phases, key=lambda x: congestion_per_phase[x])
        duration = min(max_green, max(min_green, 10 + congestion_per_phase[best_phase]*2))
        return best_phase, duration
    else:
        return 0, 30  # Default caso não haja fase verde

def get_lanes_by_phase(tls_id):
    """
    Divide lanes em NS, EW e outras, dependendo da topologia do TLS
    """
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

def get_state(tls_id):
    """
    Cria vetor de estado para o agente RL
    Inclui: filas, tempo de espera, fase atual e sugerida, verde restante
    """
    NS_lanes, EW_lanes, other_lanes = get_lanes_by_phase(tls_id)
    ns_queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in NS_lanes)
    ew_queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in EW_lanes)
    ns_wait = sum(traci.lane.getWaitingTime(lane) for lane in NS_lanes)
    ew_wait = sum(traci.lane.getWaitingTime(lane) for lane in EW_lanes)
    phase = agents[tls_id]["current_phase"]
    best_phase, suggested_duration = compute_adaptive_state(tls_id)
    verde_restante = agents[tls_id]["verde_restante"]
    return np.array([ns_queue, ew_queue, ns_wait, ew_wait, phase, best_phase, suggested_duration, verde_restante],
                    dtype=np.float32)

def take_action(tls_id, action):
    """
    Executa a ação escolhida pelo agente no SUMO
    """
    agent = agents[tls_id]
    short, medium, long = 20, 30, 45  # Durações em steps

    if action == 0:
        agent["verde_restante"] = short
    elif action == 1:
        agent["verde_restante"] = medium
    elif action == 2:
        agent["verde_restante"] = long
    elif action == 3:
        # Troca de fase NS <-> EW
        agent["current_phase"] = 1 - agent["current_phase"]
        traci.trafficlight.setPhase(tls_id, agent["current_phase"])
        agent["verde_restante"] = short
        agent["phase_changes"] += 1
        metrics_rl[tls_id]["phase_changes"] += 1

def compute_reward(tls_id, step):
    """
    Calcula a recompensa para um TLS considerando:
    - Veículos que passaram por esse TLS
    - Filas nas lanes
    - Tempo de espera acumulado
    - Prioridade para ônibus
    """
def compute_reward(tls_id, step):
    """
    Calcula recompensa considerando:
    - Veículos passados por TLS
    - Filas
    - Tempo de espera
    - Prioridade de ônibus
    """
    NS_lanes, EW_lanes, other_lanes = get_lanes_by_phase(tls_id)
    
    # Contagem de veículos atuais por TLS
    ns_vehicles = sum(len(traci.lane.getLastStepVehicleIDs(lane)) for lane in NS_lanes)
    ew_vehicles = sum(len(traci.lane.getLastStepVehicleIDs(lane)) for lane in EW_lanes)
    other_vehicles = sum(len(traci.lane.getLastStepVehicleIDs(lane)) for lane in other_lanes)
    total_vehicles = ns_vehicles + ew_vehicles + other_vehicles

    # Tempo de espera acumulado por TLS
    ns_wait = sum(traci.lane.getWaitingTime(lane) for lane in NS_lanes)
    ew_wait = sum(traci.lane.getWaitingTime(lane) for lane in EW_lanes)
    other_wait = sum(traci.lane.getWaitingTime(lane) for lane in other_lanes)
    total_wait = ns_wait + ew_wait + other_wait

    # Contagem de veículos que saíram **somente por esse TLS**
    # Usa o número de veículos que saíram no step inteiro (traci.simulation.getArrivedNumber())
    passed_tls = traci.simulation.getArrivedNumber()  

    # Recompensa extra para ônibus ainda dentro das lanes do TLS
    bus_passed = sum(
        len([v for v in traci.lane.getLastStepVehicleIDs(lane) if traci.vehicle.getTypeID(v)=="bus"])
        for lane in NS_lanes + EW_lanes
    )

    # Normalização para evitar valores extremos
    norm_queue = np.tanh(total_vehicles / 10.0)
    norm_wait = np.tanh(total_wait / 100.0)

    # Definição dos pesos para cada componente da função de recompensa
    # alpha: peso para veículos que passam (incentivo ao fluxo)
    # beta: penalidade para filas grandes (desincentivo a congestionamento)
    # gamma: penalidade para tempo de espera acumulado (evita esperas longas)
    # delta: recompensa extra para ônibus que passam (prioridade ao transporte público)
    alpha, beta, gamma, delta = 15.0, 4.0, 3.0, 30.0  

    # Recompensa proporcional ao número de veículos que saíram do semáforo no step
    passed_reward = alpha * passed_tls

    # Penalidade associada ao tamanho da fila total, normalizada
    queue_penalty = - beta * norm_queue
    # Penalidade associada ao tempo de espera acumulado dos veículos, normalizada
    wait_penalty = - gamma * norm_wait

    # Recompensa extra para ônibus que passaram, reforçando prioridade ao transporte público
    bus_reward = delta * bus_passed

    # Combina todos os componentes em uma recompensa base
    base_reward = passed_reward + queue_penalty + wait_penalty + bus_reward

    # Aplicação de curva sigmoide para crescimento da recompensa ao longo da simulação
    growth = 1 / (1 + np.exp(-(step-1500)/300))  

    # Garante que a recompensa seja positiva e ajusta offset
    offset = abs(min(0, base_reward)) + 0.1
    reward = (base_reward + offset) * growth

    # Atualiza métricas
    agents[tls_id]["reward_history"].append(reward)
    metrics_rl[tls_id]["total_queue"].append(total_vehicles)
    metrics_rl[tls_id]["total_wait"].append(total_wait)
    metrics_rl[tls_id]["vehicles_passed"] += passed_tls

    return reward


def train(agent):
    """
    Atualiza a rede neural DQN com batch de experiências do Replay Buffer
    """
    if len(agent["memory"]) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = agent["memory"].sample(BATCH_SIZE)
    states = torch.tensor(states, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Atualização Q-learning
    q_values = agent["model"](states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        q_next = agent["model"](next_states).max(1)[0]
        q_target = rewards + GAMMA * q_next * (1 - dones)

    loss = nn.MSELoss()(q_values, q_target)
    agent["optimizer"].zero_grad()
    loss.backward()
    agent["optimizer"].step()

def log_step(tls_id, step, state, action, reward):
    """
    Imprime log detalhado do step para análise 
    """
    action_map = {0:"Verde curto",1:"Verde médio",2:"Verde longo",3:"Trocar fase"}
    phase_map = {0:"NS",1:"EW"}
    NS_queue, EW_queue, NS_wait, EW_wait, phase, best_phase, sug_dur, verde_rest = state
    print(f"Step {step:04d} | TLS {tls_id} | Fase atual: {phase_map[int(phase)]} | Verde restante: {int(verde_rest)} | "
          f"NS: {int(NS_queue)} carros / {int(NS_wait)} s espera | EW: {int(EW_queue)} carros / {int(EW_wait)} s espera | "
          f"Ação: {action_map[action]} | Recompensa: {reward:.2f}")

# -----------------------------
# LOOP PRINCIPAL COM TRY/EXCEPT
# -----------------------------
# Inicializa gráfico interativo para recompensa
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

try:
    for step in range(NUM_STEPS):
        step_rewards = []
        for tls_id, agent in agents.items():
            state = get_state(tls_id)
            # Política ε-greedy
            if random.random() < agent["epsilon"]:
                action = random.choice([0,1,2,3])
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = agent["model"](state_tensor)
                action = torch.argmax(q_values).item()

            take_action(tls_id, action)
            reward = compute_reward(tls_id, step)
            next_state = get_state(tls_id)
            done = False
            agent["memory"].add(state, action, reward, next_state, done)
            train(agent)
            agent["epsilon"] = max(EPSILON_END, agent["epsilon"]*EPSILON_DECAY)
            step_rewards.append(reward)

            simulation_log[tls_id]["states"].append(state)
            simulation_log[tls_id]["actions"].append(action)
            simulation_log[tls_id]["rewards"].append(reward)
            log_step(tls_id, step, state, action, reward)

        total_reward = sum(step_rewards)
        rewards_all.append(total_reward)

        # Média móvel para suavização
        if len(rewards_all) >= WINDOW_MOVING_AVG:
            smoothed_rewards = np.convolve(rewards_all, np.ones(WINDOW_MOVING_AVG)/WINDOW_MOVING_AVG, mode='valid')
        else:
            smoothed_rewards = np.convolve(rewards_all, np.ones(len(rewards_all))/len(rewards_all), mode='valid')

        traci.simulationStep()  # Avança simulação SUMO

        # Atualiza gráfico
        line1.set_data(range(len(rewards_all)), rewards_all)
        line2.set_data(range(len(smoothed_rewards)), smoothed_rewards)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)

except traci.FatalTraCIError as e:
    print(f"Erro crítico na simulação no step {step}: {e}")

finally:
    traci.close()
    print("SUMO finalizado")

# -----------------------------
# RELATÓRIO FINAL
# -----------------------------
print("\n--- RELATÓRIO FINAL DA SIMULAÇÃO ---\n")
for tls_id in tls_ids:
    fila_final = metrics_rl[tls_id]['total_queue'][-1] if metrics_rl[tls_id]['total_queue'] else 0
    espera_final = metrics_rl[tls_id]['total_wait'][-1] if metrics_rl[tls_id]['total_wait'] else 0
    fases = metrics_rl[tls_id]['phase_changes']
    print(f"TLS {tls_id}: Fases: {fases} | Fila final: {fila_final:.1f} | Espera final: {espera_final:.1f}")

