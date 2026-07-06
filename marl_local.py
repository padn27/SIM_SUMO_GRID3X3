#!/usr/bin/env python3

import os
import argparse
import numpy as np
import random
import json
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sumo_rl import parallel_env

DELTA_TIME = 30
NUM_SECONDS = 3600

# =========================
# RECOMPENSA LOCAL PURA
# =========================
def recompensa_local(ts):
    alpha, beta, gamma = 1.0, 0.005, 0.5

    lanes = ts.lanes
    veiculos = []
    for l in lanes:
        veiculos.extend(ts.sumo.lane.getLastStepVehicleIDs(l))

    if veiculos:
        W = sum(ts.sumo.vehicle.getWaitingTime(v) for v in veiculos) / len(veiculos)
    else:
        W = 0.0

    Q = sum(ts.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)
    E = sum(ts.sumo.lane.getCO2Emission(l) for l in lanes)

    return float(-(alpha * W + beta * Q + gamma * E))


# =========================
# DQN SIMPLES (LOCAL)
# =========================
class DQN(nn.Module):
    def __init__(self, s, a):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, a)
        )

    def forward(self, x):
        return self.net(x)


class AgentLocal:
    def __init__(self, s, a):
        self.policy = DQN(s, a)
        self.target = DQN(s, a)
        self.target.load_state_dict(self.policy.state_dict())

        self.opt = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.memory = deque(maxlen=20000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.tau = 0.005
        self.action_size = a

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return int(torch.argmax(self.policy(s)).item())

    def store(self, *args):
        self.memory.append(args)

    def train(self):
        if len(self.memory) < 64:
            return

        batch = random.sample(self.memory, 64)
        s, a, r, s2, d = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(a).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

        q = self.policy(s).gather(1, a)

        with torch.no_grad():
            q2 = self.target(s2).max(1)[0].unsqueeze(1)

        target = r + (1 - d) * self.gamma * q2

        loss = nn.MSELoss()(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def update(self):
        for p, t in zip(self.policy.parameters(), self.target.parameters()):
            t.data.copy_(self.tau * p.data + (1 - self.tau) * t.data)


# =========================
# TREINAMENTO LOCAL
# =========================
def treinar(args):

    with open(args.perfis_json) as f:
        perfis = json.load(f)

    ts_ids = perfis[args.perfil]

    routes = sorted([
        os.path.join(args.rotasdir, f)
        for f in os.listdir(args.rotasdir)
        if f.endswith(".rou.xml")
    ])

    env = None
    agents = {}
    route_i = 0

    for ep in range(args.episodios):

        if ep % args.troca == 0:
            if env:
                env.close()

            env = parallel_env(
                net_file=args.net,
                route_file=routes[route_i],
                num_seconds=NUM_SECONDS,
                delta_time=DELTA_TIME,
                reward_fn=recompensa_local,
                sumo_warnings=False
            )

            route_i = (route_i + 1) % len(routes)

        obs, _ = env.reset()

        if not agents:
            for ts in ts_ids:
                s = env.observation_space(ts).shape[0]
                a = env.action_space(ts).n
                agents[ts] = AgentLocal(s, a)

        while env.agents:
            actions = {}

            for ts in env.agents:
                if ts in ts_ids:
                    actions[ts] = agents[ts].act(obs[ts])
                else:
                    actions[ts] = 0

            next_obs, r, term, trunc, _ = env.step(actions)

            for ts in ts_ids:
                agents[ts].store(
                    obs[ts],
                    actions[ts],
                    r[ts],
                    next_obs[ts],
                    term[ts] or trunc[ts]
                )

                agents[ts].train()
                agents[ts].update()

            obs = next_obs

    print("Treinamento LOCAL finalizado.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodios", type=int, default=100)
    p.add_argument("--troca", type=int, default=10)
    p.add_argument("--rotasdir", type=str, default="rotas_jtr_marl")
    p.add_argument("--net", type=str, default="baseSumo/grid.net.xml")
    p.add_argument("--perfil", type=str, default="perfil_cruz")
    p.add_argument("--perfis_json", type=str, default="perfis_treinamento_marl_3x3.json")

    args = p.parse_args()
    treinar(args)