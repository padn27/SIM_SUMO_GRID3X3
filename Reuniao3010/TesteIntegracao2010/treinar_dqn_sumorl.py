#!/usr/bin/env python3
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
# vai mudar so pra ter a definicao
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
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1, state_size)
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

        # decay epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ---------------- plotting ----------------
def plot_recompensas_multiplos_agentes(recompensas_episodios, window=5, save_path=None):
    plt.figure(figsize=(12, 6))
    all_rewards = []
    for agent_id, rewards in recompensas_episodios.items():
        rewards = np.array(rewards)
        all_rewards.append(rewards)
        plt.plot(rewards, alpha=0.3, label=f'{agent_id} (raw)')
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards)), smoothed, label=f'{agent_id} (smoothed)')

    if len(all_rewards) > 0:
        stacked = np.array([r for r in all_rewards], dtype=object)
        # pad to same length
        max_len = max(len(r) for r in stacked)
        padded = np.array([np.concatenate([r, np.full(max_len - len(r), np.nan)]) if len(r) < max_len else r for r in stacked])
        if padded.shape[0] > 1:
            mean_rewards = np.nanmean(padded, axis=0)
            std_rewards = np.nanstd(padded, axis=0)
            plt.plot(mean_rewards, color='black', linewidth=2, label='Média agentes RL')
            plt.fill_between(range(len(mean_rewards)), mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2, label='Desvio padrão')

    plt.xlabel("Episódios")
    plt.ylabel("Recompensa total")
    plt.title("Evolução da Recompensa por Episódio - Agentes RL")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Figura salva em: {save_path}")
    plt.show()

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
    args = parser.parse_args()

    USE_GUI = args.validacao
    NUM_EPISODES = args.episodios
    CENARIOS_DIR = os.path.join(os.getcwd(), "cenarios")
    SCENARIO_NAME = args.nomecenario

    scenario_path = os.path.join(CENARIOS_DIR, SCENARIO_NAME)
    if not os.path.isdir(scenario_path):
        raise Exception(f"Cenário '{SCENARIO_NAME}' não encontrado em {CENARIOS_DIR}")

    net_file = os.path.join(scenario_path, "grid.net.xml")
    route_file = os.path.join(scenario_path, f"{SCENARIO_NAME}.rou.xml")
    add_file = os.path.join(scenario_path, "grid.add.xml")

    if not os.path.exists(net_file):
        raise FileNotFoundError(f"Arquivo net não encontrado: {net_file}")
    if not os.path.exists(route_file):
        raise FileNotFoundError(f"Arquivo rou não encontrado: {route_file}")

    # cria ambiente (passa função de recompensa)
    env = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=route_file,
        use_gui=USE_GUI,
        num_seconds=3600,
        delta_time=10,
        reward_fn='diff-waiting-time' #minha_recompensa
    )

    # carrega tipo semaforo
    with open(os.path.join(scenario_path, "tls_config.json"), "r") as f:
        tls_config = json.load(f)

    agentes_rl = [k for k, v in tls_config.items() if v == 'R']
    agentes_fixos = [k for k, v in tls_config.items() if v in ['F', 'A']]

    print(f"--- Treinamento no cenário: {SCENARIO_NAME} ---")
    print(f"Agentes RL: {agentes_rl} | Agentes Fixos/Adaptativos: {agentes_fixos}")

    # estrutura para agentes
    agents_dict = {}
    recompensas_episodios = {agent_id: [] for agent_id in agentes_rl}

    # primeiro reset para confirmar obs e pegar o espaço
    observations, infos = env.reset()

    # um agente dqn por semaforo (burro)
    for agent_id in agentes_rl:
        # tenta extrair observation_space e action_space
        state_size = None
        action_size = None
        try:
            obs_space = env.observation_space(agent_id)
            act_space = env.action_space(agent_id)
            # se for Box
            if isinstance(obs_space, gym.spaces.Box):
                state_size = int(np.prod(obs_space.shape))
            else:
                # fallback para len da observação inicial (se disponível)
                if agent_id in observations and observations[agent_id] is not None:
                    state_size = len(observations[agent_id])
            if isinstance(act_space, gym.spaces.Discrete):
                action_size = act_space.n
            else:
                action_size = getattr(act_space, "n", None)
        except Exception:
            # fallback se a API não expuser diretamente
            pass

        if state_size is None:
            # tamanho do espaço de observações, já é geral
            if agent_id in observations and observations[agent_id] is not None:
                try:
                    state_size = len(observations[agent_id])
                except Exception:
                    state_size = DEFAULT_STATE_SIZE
            else:
                state_size = DEFAULT_STATE_SIZE

        if action_size is None or action_size <= 0:
            action_size = DEFAULT_ACTION_SIZE

        print(f"agente semaforo {agent_id}: state_size={state_size}, action_size={action_size}")
        agents_dict[agent_id] = DQNAgent(state_size, action_size)

    # estado para agentes fixos 
    other_state = {aid: 0 for aid in agentes_fixos}

    # loop de episódios
    for ep in range(NUM_EPISODES):
        observations, infos = env.reset()
        states = observations  # dict
        total_rewards_episode = {agent_id: 0.0 for agent_id in agents_dict}
        step_count = 0

        while env.agents:
            actions_dict = {}

            # RL agents: cada um pega seu DQN e seu espaço
            for agent_id in agentes_rl:
                if agent_id in states:
                    s = states[agent_id]
                    # transformar o estado em vetor 1D
                    try:
                        if isinstance(s, (list, tuple)):
                            s_vec = np.array(s, dtype=np.float32).ravel()
                        else:
                            s_vec = np.array(s, dtype=np.float32).ravel()
                    except Exception:
                        s_vec = None

                    action = agents_dict[agent_id].select_action(s_vec)

                    # teste pra ver se nao esta passando açoes impossiveis
                    try:
                        a_space = env.action_space(agent_id)
                        if isinstance(a_space, gym.spaces.Discrete):
                            max_a = a_space.n - 1
                            if action > max_a:
                                action = max_a
                            if action < 0:
                                action = 0
                    except Exception:
                        # se não conseguir acessar action_space, mantenha action, mas vai dar exception
                        pass

                    actions_dict[agent_id] = int(action)

            # não-RL: alterna 0/1 a cada step 
            for agent_id in agentes_fixos:
                if agent_id in states and agent_id in env.agents:
                    other_state[agent_id] = 1 - other_state[agent_id]
                    actions_dict[agent_id] = int(other_state[agent_id])

            # step (env espera ações para todos agentes ativos; já estamos enviando)
            next_obs, rewards, terminated, truncated, infos = env.step(actions_dict)

            # armazenar transições e treinar por agente RL
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
                        # não interromper o loop por erros de shape
                        print(f"processamento transição {agent_id}: {e}")

            states = next_obs
            step_count += 1

        # atualizar target networks periodicamente
        if ep % TARGET_UPDATE == 0:
            for agent_id in agentes_rl:
                agents_dict[agent_id].update_target()

        # salvar recompensas por agente
        for agent_id in agentes_rl:
            recompensas_episodios[agent_id].append(total_rewards_episode.get(agent_id, 0.0))

        print(f"Episódio {ep+1}/{NUM_EPISODES} | steps_agente: {step_count} | recompensas: {total_rewards_episode}")

    env.close()
    print("--- Treinamento finalizado ---")


