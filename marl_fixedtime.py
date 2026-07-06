#!/usr/bin/env python3
import os
import argparse
import numpy as np
import json
import multiprocessing
import time
import csv

from sumo_rl import parallel_env

# =========================
# PARÂMETROS 
# =========================
DELTA_TIME = 30  # tempo fixo por fase


def fixed_time_policy(step_count, num_phases):
    """
    Política Fixed-Time clássica:
    cada fase dura DELTA_TIME passos.
    """
    return (step_count // DELTA_TIME) % num_phases


def minha_recompensa(ts):
    """
    MESMA recompensa do MARL
    """
    alpha = 1.0
    beta = 0.005
    gamma = 0.5

    lanes = ts.lanes

    vehicles = []
    for l in lanes:
        vehicles.extend(ts.sumo.lane.getLastStepVehicleIDs(l))

    if vehicles:
        W = sum(ts.sumo.vehicle.getWaitingTime(v) for v in vehicles) / len(vehicles)
    else:
        W = 0.0

    Q = sum(ts.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)
    E = sum(ts.sumo.lane.getCO2Emission(l) for l in lanes)

    return float(-(alpha * W + beta * Q + gamma * E))


# =========================
# TREINAMENTO FIXED-TIME
# =========================
def executar_fixed_time(agent_id, args):

    time.sleep(agent_id * 1.5)

    USE_GUI = args.validacao if agent_id == 0 else False

    NET_FILE = args.net
    ROTAS_DIR = args.rotasdir
    NUM_EPISODES = args.episodios
    TROCA = args.troca

    os.makedirs("DadosCSV", exist_ok=True)

    perfil_dir = os.path.join("DadosCSV", args.perfil)
    os.makedirs(perfil_dir, exist_ok=True)

    caminho_csv = os.path.join(perfil_dir, f"fixed_time_proc_{agent_id}.csv")

    with open(caminho_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episodio", "semaforo", "tipo",
            "reward_total", "delay_total",
            "espera_total", "fila_total",
            "throughput_total", "co2_total"
        ])

    route_files = sorted([
        os.path.join(ROTAS_DIR, f)
        for f in os.listdir(ROTAS_DIR)
        if f.endswith(".rou.xml")
    ])

    current_route_idx = 0
    env = None

    ts_ids = args.ts_selecionados

    for ep in range(NUM_EPISODES):

        if ep % TROCA == 0:
            if env:
                env.close()

            rota = route_files[current_route_idx]
            print(f"[Fixed-Time | Ep {ep}] -> {os.path.basename(rota)}")

            env = parallel_env(
                net_file=NET_FILE,
                route_file=rota,
                use_gui=USE_GUI,
                num_seconds=3600,
                delta_time=DELTA_TIME,
                reward_fn=minha_recompensa,
                observation_class=None,
                sumo_warnings=False
            )

            current_route_idx = (current_route_idx + 1) % len(route_files)

        observations, infos = env.reset()

        step_count = 0

        # acumuladores por episódio
        metricas_ep = {}
        reward_ep = {ts: 0.0 for ts in env.agents}

        while env.agents:

            actions = {}

            for ts_id in env.agents:

                num_phases = env.action_space(ts_id).n

                # FIXED-TIME PURO
                actions[ts_id] = fixed_time_policy(step_count, num_phases)

                if ts_id not in metricas_ep:
                    metricas_ep[ts_id] = {
                        "fila": 0.0,
                        "co2": 0.0,
                        "espera": 0.0,
                        "delay": 0.0,
                        "throughput": 0
                    }

                ts_obj = env.unwrapped.env.traffic_signals[ts_id]
                lanes = ts_obj.lanes

                fila = sum(ts_obj.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)
                co2 = sum(ts_obj.sumo.lane.getCO2Emission(l) for l in lanes)

                veiculos = []
                for l in lanes:
                    veiculos.extend(ts_obj.sumo.lane.getLastStepVehicleIDs(l))

                if veiculos:
                    espera = sum(ts_obj.sumo.vehicle.getWaitingTime(v) for v in veiculos)
                    delay = sum(ts_obj.sumo.vehicle.getAccumulatedWaitingTime(v) for v in veiculos)
                else:
                    espera = 0.0
                    delay = 0.0

                metricas_ep[ts_id]["fila"] += fila
                metricas_ep[ts_id]["co2"] += co2
                metricas_ep[ts_id]["espera"] += espera
                metricas_ep[ts_id]["delay"] += delay

            next_obs, rewards, terms, truncs, infos = env.step(actions)

            for ts_id in rewards:
                reward_ep[ts_id] += rewards[ts_id]

            observations = next_obs
            step_count += 1

        # =========================
        # SALVA CSV
        # =========================
        buffer = []

        for ts_id, m in metricas_ep.items():

            buffer.append([
                ep,
                ts_id,
                "Fixed-Time",
                reward_ep.get(ts_id, 0.0),
                m["delay"],
                m["espera"],
                m["fila"],
                m["throughput"],
                m["co2"]
            ])

        with open(caminho_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(buffer)

        print(f"[Fixed-Time] Ep {ep} finalizado")

    if env:
        env.close()

    print(f"Processo {agent_id} finalizado.")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--validacao", action="store_true")
    parser.add_argument("--episodios", type=int, default=100)
    parser.add_argument("--troca", type=int, default=10)
    parser.add_argument("--rotasdir", type=str, default="rotas_jtr_marl")
    parser.add_argument("--net", type=str, default="baseSumo/grid.net.xml")
    parser.add_argument("--agentes", type=int, default=1)
    parser.add_argument("--perfil", type=str, default="perfil_cruz")

    args = parser.parse_args()

    with open("perfis_treinamento_marl_3x3.json", "r") as f:
        perfis = json.load(f)

    args.ts_selecionados = perfis[args.perfil]

    if args.agentes > 1:
        processos = [
            multiprocessing.Process(target=executar_fixed_time, args=(i, args))
            for i in range(args.agentes)
        ]

        for p in processos:
            p.start()

        for p in processos:
            p.join()

    else:
        executar_fixed_time(0, args)