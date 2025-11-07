#!/usr/bin/env python3
# treinar_dqn_sumorl.py (versão adaptada para DQN por agente - usa sumo_rl 1.4.5)
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import sumo_rl
import gymnasium as gym
import matplotlib.pyplot as plt
import json

# ---------------- Configurações do DQN ----------------
GAMMA = 0.95
LR = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 10000
BATCH_SIZE = 64
TARGET_UPDATE = 10
DEFAULT_STATE_SIZE = 20
DEFAULT_ACTION_SIZE = 2

# ---------------- DQN ----------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.action_size = action_size

    def select_action(self, state):
        if state is None:
            return random.randrange(self.action_size)
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_net(s)
            return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones).astype(float), dtype=torch.float32).unsqueeze(1)
        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target = rewards + (1 - dones) * GAMMA * next_q_values
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())



# ---------------- recompensa diferente ----------------
def minha_recompensa(traffic_signal):
    try:
        lanes = traffic_signal.lanes
        if not lanes:
            return 0.0
        queue = sum(traffic_signal.env.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)
        wait = sum(traffic_signal.env.sumo.lane.getWaitingTime(l) for l in lanes)
        return -(queue + wait)
    except Exception as e:
        print(f"erro de recompensa de {getattr(traffic_signal,'id', 'unknown')}: {e}")
        return 0.0

# ---------------- main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validacao", action="store_true", help="usar GUI")
    parser.add_argument("--nomecenario", type=str, required=True, help="nome do cenário")
    parser.add_argument("--episodios", type=int, default=100, help="número de episódios")
    parser.add_argument("--rotasdir", type=str, default="rotas_jtr", help="diretório com as rotas geradas (.rou.xml)")
    parser.add_argument("--troca", type=int, default=10, help="trocar rota a cada N episódios")

    args = parser.parse_args()
    USE_GUI = args.validacao
    NUM_EPISODES = args.episodios
    CENARIOS_DIR = os.path.join(os.getcwd(), "cenarios")
    SCENARIO_NAME = args.nomecenario

    scenario_path = os.path.join(CENARIOS_DIR, SCENARIO_NAME)
    if not os.path.isdir(scenario_path):
        raise Exception(f"cenário '{SCENARIO_NAME}' não encontrado em {CENARIOS_DIR}")

    net_file = os.path.join(scenario_path, "grid.net.xml")
    route_file = os.path.join(scenario_path, f"{SCENARIO_NAME}.rou.xml")
    add_file = os.path.join(scenario_path, "grid.add.xml")

    ### >>> ADIÇÃO
    ROTAS_DIR = args.rotasdir
    route_files = sorted([os.path.join(ROTAS_DIR, f) for f in os.listdir(ROTAS_DIR) if f.endswith(".rou.xml")])
    if not route_files:
        raise FileNotFoundError(f"arquivo .rou.xml nao encontrado em {ROTAS_DIR}")
    current_route_idx = 0
    TROCA = args.troca
    print(f"{len(route_files)} arquivos de rota achados em {ROTAS_DIR}")

    if not os.path.exists(net_file):
        raise FileNotFoundError(f"nao encontrado: {net_file}")
    if not os.path.exists(route_file):
        raise FileNotFoundError(f"nao encontrado: {route_file}")

    env = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=route_files[current_route_idx],
        use_gui=USE_GUI,
        num_seconds=3600,
        delta_time=10,
        reward_fn='diff-waiting-time'
    )

    with open(os.path.join(scenario_path, "tls_config.json"), "r") as f:
        tls_config = json.load(f)
    agentes_rl = [k for k, v in tls_config.items() if v == 'R']
    agentes_fixos = [k for k, v in tls_config.items() if v in ['F', 'A']]
    print(f"cenário: {SCENARIO_NAME} ---")
    print(f"agentes RL: {agentes_rl} | agentes Fixos/Adaptativos: {agentes_fixos}")

    agents_dict = {}
    recompensas_episodios = {agent_id: [] for agent_id in agentes_rl}
    observations, infos = env.reset()
    for agent_id in agentes_rl:
        state_size = None
        action_size = None
        try:
            obs_space = env.observation_space(agent_id)
            act_space = env.action_space(agent_id)
            if isinstance(obs_space, gym.spaces.Box):
                state_size = int(np.prod(obs_space.shape))
            elif agent_id in observations and observations[agent_id] is not None:
                state_size = len(observations[agent_id])
            if isinstance(act_space, gym.spaces.Discrete):
                action_size = act_space.n
            else:
                action_size = getattr(act_space, "n", None)
        except Exception:
            pass
        if state_size is None:
            if agent_id in observations and observations[agent_id] is not None:
                try:
                    state_size = len(observations[agent_id])
                except Exception:
                    state_size = DEFAULT_STATE_SIZE
            else:
                state_size = DEFAULT_STATE_SIZE
        if action_size is None or action_size <= 0:
            action_size = DEFAULT_ACTION_SIZE
        print(f"[INIT] Agente {agent_id}: state_size={state_size}, action_size={action_size}")
        agents_dict[agent_id] = DQNAgent(state_size, action_size)
    other_state = {aid: 0 for aid in agentes_fixos}

    for ep in range(NUM_EPISODES):
        ### >>> ADIÇÃO
        if ep > 0 and ep % TROCA == 0:
            env.close()
            current_route_idx = (current_route_idx + 1) % len(route_files)
            nova_rota = route_files[current_route_idx]
            print(f"\n[TROCA DE ROTA] Episódio {ep} -> {os.path.basename(nova_rota)}\n")
            env = sumo_rl.parallel_env(
                net_file=net_file,
                route_file=nova_rota,
                use_gui=USE_GUI,
                num_seconds=3600,
                delta_time=10,
                reward_fn='diff-waiting-time'
            )

        observations, infos = env.reset()
        states = observations
        total_rewards_episode = {agent_id: 0.0 for agent_id in agents_dict}
        step_count = 0
        while env.agents:
            actions_dict = {}
            for agent_id in agentes_rl:
                if agent_id in states:
                    s = states[agent_id]
                    try:
                        s_vec = np.array(s, dtype=np.float32).ravel()
                    except Exception:
                        s_vec = None
                    action = agents_dict[agent_id].select_action(s_vec)
                    try:
                        a_space = env.action_space(agent_id)
                        if isinstance(a_space, gym.spaces.Discrete):
                            max_a = a_space.n - 1
                            action = max(0, min(action, max_a))
                    except Exception:
                        pass
                    actions_dict[agent_id] = int(action)
            for agent_id in agentes_fixos:
                if agent_id in states and agent_id in env.agents:
                    other_state[agent_id] = 1 - other_state[agent_id]
                    actions_dict[agent_id] = int(other_state[agent_id])
            next_obs, rewards, terminated, truncated, infos = env.step(actions_dict)
            for agent_id in agentes_rl:
                if agent_id in states and agent_id in next_obs:
                    try:
                        s = np.array(states[agent_id], dtype=np.float32).ravel()
                        a = actions_dict.get(agent_id, 0)
                        r = rewards.get(agent_id, 0.0)
                        s2 = np.array(next_obs[agent_id], dtype=np.float32).ravel()
                        done = terminated.get(agent_id, False) or truncated.get(agent_id, False)
                        agents_dict[agent_id].remember(s, a, r, s2, done)
                        agents_dict[agent_id].replay()
                        total_rewards_episode[agent_id] += r
                    except Exception as e:
                        print(f"[WARN] processamento transição {agent_id}: {e}")
            states = next_obs
            step_count += 1
        if ep % TARGET_UPDATE == 0:
            for agent_id in agentes_rl:
                agents_dict[agent_id].update_target()
        for agent_id in agentes_rl:
            recompensas_episodios[agent_id].append(total_rewards_episode.get(agent_id, 0.0))
        print(f"Episódio {ep+1}/{NUM_EPISODES} | steps_agente: {step_count} | recompensas: {total_rewards_episode}")

    env.close()
    print("finalizado")
    plot_recompensas_multiplos_agentes(recompensas_episodios, window=10, save_path="recompensa_multiplos.png")

