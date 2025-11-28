#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gymnasium as gym
import sumo_rl
import matplotlib.pyplot as plt

#parâmetros rede
GAMMA = 0.95
LR = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995
MEMORY_SIZE = 20000
BATCH_SIZE = 64
TARGET_UPDATE = 20
DELTA_TIME = 30  # passos de simulação do SUMO
TAU = 0.01
FIXED_EPSILON = 0.3
#recompensa
def minha_recompensa(env):
    tls = env.sumo.trafficlight.getIDList()[0]
    lanes = env.sumo.trafficlight.getControlledLanes(tls)

    all_vehicles = []
    for l in lanes:
        all_vehicles.extend(env.sumo.lane.getLastStepVehicleIDs(l))

    if not all_vehicles:
        return 0.0

    wait = sum(env.sumo.vehicle.getWaitingTime(vid) for vid in all_vehicles)
    return -wait / len(all_vehicles)

#DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.tau = TAU
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START #EPSILON_START ou FIXED_EPSILON
        self.action_size = action_size

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_net(s)
            return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target = rewards + (1 - dones) * GAMMA * next_q
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY) #max(EPSILON_END, self.epsilon * EPSILON_DECAY) ou FIXED_EPSILON

    def update_target(self):
        for target_param, policy_param in zip(self.target_net.parameters(),
                                            self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )


#gráficos
plt.ion()

#grafico 1 - cada episodio
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.set_title("Recompensa Total por Episódio")
ax1.set_xlabel("Episódio")
ax1.set_ylabel("Recompensa")
ax1.grid(True)
line1, = ax1.plot([], [], 'b-', alpha=0.6)

#grafico 2 - media
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.set_title("Média Móvel da Recompensa")
ax2.set_xlabel("Episódio")
ax2.set_ylabel("Recompensa Média")
ax2.grid(True)
line2, = ax2.plot([], [], 'r-', linewidth=2)

def atualiza_graficos(rewards, rewards_media, window=10):

    arr = np.array(rewards)
    m_arr = np.array(rewards_media)

    #grafico instantanea
    line1.set_data(range(len(arr)), arr)
    ax1.set_xlim(0, max(10, len(arr)))
    ax1.set_ylim(min(arr) - 1, max(arr) + 1)
    fig1.canvas.draw()
    fig1.canvas.flush_events()

    #grafico da media
    line2.set_data(range(len(m_arr)), m_arr)
    ax2.set_xlim(0, max(10, len(m_arr)))
    ax2.set_ylim(min(m_arr) - 1, max(m_arr) + 1)
    fig2.canvas.draw()
    fig2.canvas.flush_events()



#main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validacao", action="store_true")
    parser.add_argument("--episodios", type=int, default=100)
    parser.add_argument("--troca", type=int, default=10)
    parser.add_argument("--rotasdir", type=str, default="rotas_jtr")
    parser.add_argument("--net", type=str, default="baseSumo_SA/grid.net.xml")
    parser.add_argument("--add", type=str, default="baseSumo_SA/grid.add.xml")
    args = parser.parse_args()

    USE_GUI = args.validacao
    NUM_EPISODES = args.episodios
    TROCA = args.troca
    NET_FILE = args.net
    ADD_FILE = args.add
    ROTAS_DIR = args.rotasdir



    #carregar rotas
    route_files = sorted([
        os.path.join(ROTAS_DIR, f)
        for f in os.listdir(ROTAS_DIR)
        if f.endswith(".rou.xml")
    ])
    if not route_files:
        raise FileNotFoundError(f"Nenhum arquivo .rou.xml encontrado em {ROTAS_DIR}")
    print(f"{len(route_files)} rotas encontradas")
    current_route_idx = 0

    #loop treino
    rewards_all = []
    rewards_media_movel = []
    rewards_acumuladas = 0

    for ep in range(NUM_EPISODES):
        #trocar rota a cada qtd de episodios definida nos args
        if ep % TROCA == 0:
            if ep > 0:
                env.close()
            rota_atual = route_files[current_route_idx]
            print(f"\n[Episódio {ep}] -> {os.path.basename(rota_atual)}")
            env = gym.make(
                "sumo-rl-v0",
                net_file=NET_FILE,
                route_file=rota_atual,
                use_gui=USE_GUI,
                num_seconds=3600,
                delta_time=DELTA_TIME,
                reward_fn=minha_recompensa
            )
            current_route_idx = (current_route_idx + 1) % len(route_files)
            obs, info = env.reset()
            state_size = len(obs)
            action_size = env.action_space.n
            if ep == 0:
                agent = DQNAgent(state_size, action_size)

        #reset do env a cada conjunto de episodios para trocar arquivo route
        obs, info = env.reset()
        state = np.array(obs, dtype=np.float32)
        done = False
        total_r = 0

        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_obs, dtype=np.float32)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            total_r += reward
            state = next_state

        if ep % TARGET_UPDATE == 0:
            agent.update_target()

        rewards_all.append(total_r)
        rewards_acumuladas += total_r
        rewards_media_movel.append(rewards_acumuladas/(ep + 1))
        print(f"Episódio {ep+1}/{NUM_EPISODES} | Recompensa: {total_r}")

        atualiza_graficos(rewards_all, rewards_media_movel, window=10) #atualizar os graficos

    env.close()

    #salvar gráficos
    fig1.savefig("grafico_recompensa_total.png")
    fig2.savefig("grafico_media_movel.png")

    plt.ioff()
    plt.show()

