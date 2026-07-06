#!/usr/bin/env python3

import os
import numpy as np
import json
import random
from sumo_rl import parallel_env

DELTA_TIME = 30
NUM_SECONDS = 3600

MAPA_VIZINHOS = {
    'n00': ['n01', 'n10'],
    'n01': ['n00', 'n02', 'n11'],
    'n02': ['n01', 'n12'],
    'n10': ['n00', 'n11', 'n20'],
    'n11': ['n01', 'n10', 'n12', 'n21'],
    'n12': ['n02', 'n11', 'n22'],
    'n20': ['n10', 'n21'],
    'n21': ['n20', 'n11', 'n22'],
    'n22': ['n21', 'n12']
}


def recompensa_cooperativa(ts):

    alpha, beta, gamma = 1.0, 0.005, 0.5
    fator = 0.2

    def reward(ts_obj):
        lanes = ts_obj.lanes
        v = []
        for l in lanes:
            v.extend(ts_obj.sumo.lane.getLastStepVehicleIDs(l))

        W = sum(ts_obj.sumo.vehicle.getWaitingTime(x) for x in v) / len(v) if v else 0
        Q = sum(ts_obj.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)
        E = sum(ts_obj.sumo.lane.getCO2Emission(l) for l in lanes)

        return -(alpha * W + beta * Q + gamma * E)

    local = reward(ts)

    vizinhos = MAPA_VIZINHOS.get(ts.id, [])
    env = ts.env.traffic_signals

    r_viz = []
    for v in vizinhos:
        if v in env:
            r_viz.append(reward(env[v]))

    media_viz = sum(r_viz) / len(r_viz) if r_viz else 0.0

    return (1 - fator) * local + fator * media_viz


def rodar(args):

    with open(args.perfis_json) as f:
        perfis = json.load(f)

    ts_ids = perfis[args.perfil]

    routes = sorted([
        os.path.join(args.rotasdir, f)
        for f in os.listdir(args.rotasdir)
        if f.endswith(".rou.xml")
    ])

    env = None
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
                reward_fn=recompensa_cooperativa,
                sumo_warnings=False
            )

            route_i = (route_i + 1) % len(routes)

        obs, _ = env.reset()

        while env.agents:

            actions = {}
            for ts in env.agents:
                actions[ts] = random.randrange(env.action_space(ts).n)

            env.step(actions)

    print("Treinamento COOPERATIVO finalizado.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--episodios", type=int, default=100)
    p.add_argument("--troca", type=int, default=10)
    p.add_argument("--rotasdir", type=str, default="rotas_jtr_marl")
    p.add_argument("--net", type=str, default="baseSumo/grid.net.xml")
    p.add_argument("--perfil", type=str, default="perfil_cruz")
    p.add_argument("--perfis_json", type=str, default="perfis_treinamento_marl_3x3.json")

    args = p.parse_args()
    rodar(args)