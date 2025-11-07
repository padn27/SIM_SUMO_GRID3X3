import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from sumo_env import SumoEnv
from sumolib import checkBinary

STATE_SIZE = 8
ACTION_SIZE = 2
GAMMA = 0.95
LR = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995
MEMORY_SIZE = 10000
BATCH_SIZE = 64
TARGET_UPDATE = 10
MAX_STEPS_POR_EPISODIO = 4200

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

# ---------------- Agente DQN ----------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(ACTION_SIZE)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return torch.argmax(self.policy_net(s)).item()

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
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            target = rewards + (1 - dones) * GAMMA * self.target_net(next_states).max(1)[0].unsqueeze(1)

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ---------------- Loop Principal ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validacao", action="store_true", help="usar GUI")
    parser.add_argument("--nomecenario", type=str, required=True, help="nome do cenário")
    parser.add_argument("--episodios", type=int, default=100, help="número de episódios")
    args = parser.parse_args()

    USE_GUI = args.validacao
    SUMO_BINARY = checkBinary("sumo-gui") if USE_GUI else checkBinary("sumo")
    CENARIOS_DIR = os.path.join(os.getcwd(), "cenarios")

    scenario_path = os.path.join(CENARIOS_DIR, args.nomecenario)
    if not os.path.isdir(scenario_path):
        raise Exception(f"Cenário '{args.nomecenario}' não encontrado em {CENARIOS_DIR}")

    print(f"--- Iniciando treinamento no cenário: {args.nomecenario} ---")
    env = SumoEnv(
        sumo_cfg=os.path.join(scenario_path, "grid.sumocfg"),
        tls_config=os.path.join(scenario_path, "tls_config.json"),
        use_gui=USE_GUI,
        max_steps=MAX_STEPS_POR_EPISODIO,
        sumo_binary=SUMO_BINARY
    )

    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    recompensas_episodios = []

    for ep in range(args.episodios):
        state, _ = env.reset()
        total_reward = 0.0
        done_all = {tls: False for tls in env.rl_ids}

        while not all(done_all.values()):
            actions = {tls: agent.select_action(state[tls]) for tls in env.rl_ids}
            next_state, reward, terminated, truncated, _ = env.step(actions)

            for tls_id in env.rl_ids:
                agent.remember(state[tls_id], actions[tls_id], reward[tls_id], next_state[tls_id], terminated[tls_id] or truncated[tls_id])
                total_reward += reward[tls_id]

            state = next_state
            agent.replay()

            if ep % TARGET_UPDATE == 0:
                agent.update_target()

            done_all = terminated

        recompensas_episodios.append(total_reward)
        print(f"Episódio {ep+1}/{args.episodios} | Recompensa: {total_reward:.2f} | Epsilon: {agent.epsilon:.4f}")

    print("--- Treinamento finalizado ---")
    print(f"Média de recompensa: {np.mean(recompensas_episodios):.2f}")
    env.close()


