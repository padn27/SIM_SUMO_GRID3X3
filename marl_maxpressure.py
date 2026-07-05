#!/usr/bin/env python3

import os
import argparse
import json
import csv
import time

import numpy as np
from sumo_rl import parallel_env


DELTA_TIME = 30      # cada decisao a cada 30 s de simulacao
NUM_SECONDS = 3600   # 1 hora de simulacao por episodio

# mapa vizinhanca 3x3
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



def minha_recompensa(ts):
    alpha = 0.8     # peso para o tempo medio de espera
    beta = 1.0      # peso para o comprimento da fila
    gamma = 0.005   # peso para emissao de CO2

    lanes = ts.lanes
    all_vehicles = []
    for l in lanes:
        all_vehicles.extend(ts.sumo.lane.getLastStepVehicleIDs(l))

    if not all_vehicles:
        W = 0.0
    else:
        total_wait = sum(ts.sumo.vehicle.getWaitingTime(vid) for vid in all_vehicles)
        W = total_wait / len(all_vehicles)

    Q = sum(ts.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)
    E = sum(ts.sumo.lane.getCO2Emission(l) for l in lanes)

    return float(-(alpha * W + beta * Q + gamma * E))


def minha_recompensa_vizinhos(ts):
    alpha = 0.8
    beta = 1.0
    gamma = 0.005
    fator_coop = 0.2  # 20% do foco na "dor" dos vizinhos

    #parte local
    lanes_local = ts.lanes
    all_vehicles_local = []
    for l in lanes_local:
        all_vehicles_local.extend(ts.sumo.lane.getLastStepVehicleIDs(l))

    if not all_vehicles_local:
        W_local = 0.0
    else:
        total_wait_local = sum(ts.sumo.vehicle.getWaitingTime(vid) for vid in all_vehicles_local)
        W_local = total_wait_local / len(all_vehicles_local)

    Q_local = sum(ts.sumo.lane.getLastStepHaltingNumber(l) for l in lanes_local)
    E_local = sum(ts.sumo.lane.getCO2Emission(l) for l in lanes_local)
    recompensa_local = -(alpha * W_local + beta * Q_local + gamma * E_local)

    #parte dos vizinhos
    vizinhos_ids = MAPA_VIZINHOS.get(ts.id, [])
    todos_ts = ts.env.traffic_signals
    recompensas_viz = []

    for v_id in vizinhos_ids:
        if v_id in todos_ts:
            ts_viz = todos_ts[v_id]
            lanes_viz = ts_viz.lanes

            all_vehicles_viz = []
            for l in lanes_viz:
                all_vehicles_viz.extend(ts_viz.sumo.lane.getLastStepVehicleIDs(l))

            if not all_vehicles_viz:
                W_viz = 0.0
            else:
                total_wait_viz = sum(ts_viz.sumo.vehicle.getWaitingTime(vid) for vid in all_vehicles_viz)
                W_viz = total_wait_viz / len(all_vehicles_viz)

            Q_viz = sum(ts_viz.sumo.lane.getLastStepHaltingNumber(l) for l in lanes_viz)
            E_viz = sum(ts_viz.sumo.lane.getCO2Emission(l) for l in lanes_viz)
            recompensas_viz.append(-(alpha * W_viz + beta * Q_viz + gamma * E_viz))

    media_vizinhos = (sum(recompensas_viz) / len(recompensas_viz)) if recompensas_viz else 0.0
    return float((1.0 - fator_coop) * recompensa_local + fator_coop * media_vizinhos)


REWARD_FNS = {
    "local": minha_recompensa,
    "vizinhos": minha_recompensa_vizinhos,
}


# max pressure
class MaxPressureController:

    def __init__(self, ts_obj):
       
        self.phase_states = [p.state for p in ts_obj.green_phases]
        self.controlled_links = ts_obj.sumo.trafficlight.getControlledLinks(ts_obj.id)

    def select_action(self, ts_obj):
        sumo = ts_obj.sumo  # conexao TraCI 

        cache_fila = {}
        def fila(via):
            if via and via not in cache_fila:
                # getLastStepHaltingNumber - getLastStepVehicleNumber.
                cache_fila[via] = sumo.lane.getLastStepVehicleNumber(via)
            return cache_fila.get(via, 0)

        melhor_fase, melhor_pressao = 0, float('-inf')
        for idx, estado in enumerate(self.phase_states):
            pressao = 0.0
            n = min(len(estado), len(self.controlled_links))
            for i in range(n):
                if estado[i] in ('G', 'g'):  # movimento liberado nesta fase
                    for link in self.controlled_links[i]:
                        via_in, via_out = link[0], link[1]
                        pressao += fila(via_in) - fila(via_out)
            if pressao > melhor_pressao:
                melhor_pressao, melhor_fase = pressao, idx
        return melhor_fase



def rodar_baseline(args):
    ts_selecionados = args.ts_selecionados
    reward_fn = REWARD_FNS[args.recompensa]

    # rotas
    route_files = sorted(
        os.path.join(args.rotasdir, f)
        for f in os.listdir(args.rotasdir) if f.endswith(".rou.xml")
    )
    if not route_files:
        raise FileNotFoundError(f"Nenhum arquivo .rou.xml encontrado em {args.rotasdir}")

    # CSV de saida 
    dados_csv_dir = "DadosCSV"
    perfil_dir = os.path.join(dados_csv_dir, args.perfil)
    os.makedirs(perfil_dir, exist_ok=True)
    caminho_csv = os.path.join(perfil_dir, "metricas_maxpressure_proc_0.csv")
    with open(caminho_csv, mode='w', newline='') as f:
        csv.writer(f).writerow(
            ["episodio", "semaforo", "tipo", "reward_total", "delay_total",
             "espera_total", "fila_total", "throughput_total", "co2_total"]
        )
    print(f"CSV de saida: {caminho_csv}")

    env = None
    controladores = {}
    current_route_idx = 0

    for ep in range(args.episodios):
        # troca de rota 
        if ep % args.troca == 0:
            if env is not None:
                env.close()
            rota_atual = route_files[current_route_idx]
            print(f"\n[Ep {ep}] rota -> {os.path.basename(rota_atual)}")
            env = parallel_env(
                net_file=args.net,
                route_file=rota_atual,
                use_gui=args.gui,
                num_seconds=NUM_SECONDS,
                delta_time=DELTA_TIME,
                reward_fn=reward_fn,
                sumo_warnings=False,
            )
            current_route_idx = (current_route_idx + 1) % len(route_files)

        observations, infos = env.reset()

        # cria um max pressure por semaforo selecionado (uma vez so)
        if not controladores:
            base = env.unwrapped.env if hasattr(env.unwrapped, 'env') else env.unwrapped
            for ts_id in ts_selecionados:
                controladores[ts_id] = MaxPressureController(base.traffic_signals[ts_id])
                n_fases = env.action_space(ts_id).n
                print(f"  Semaforo {ts_id}: Max Pressure | nº de fases = {n_fases}")

        recompensa_ep = {ts_id: 0.0 for ts_id in ts_selecionados}
        metricas_ep = {}
        step_count = 0

        while env.agents:
            base = env.unwrapped.env if hasattr(env.unwrapped, 'env') else env.unwrapped

            #acoes
            actions = {}
            for ts_id in env.agents:
                if ts_id in ts_selecionados:
                    ts_obj = base.traffic_signals[ts_id]
                    actions[ts_id] = controladores[ts_id].select_action(ts_obj)
                else:
                    # semaforos fora do perfil seguem fixos-ciclicos
                    num_phases = env.action_space(ts_id).n
                    actions[ts_id] = step_count % num_phases

            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            # acumulacao de metricas
            for ts_id in env.agents:
                if ts_id not in metricas_ep:
                    metricas_ep[ts_id] = {'fila_sum': 0.0, 'co2_sum': 0.0,
                                          'espera_sum': 0.0, 'delay_sum': 0.0,
                                          'throughput_ids': set()}

                ts_obj = base.traffic_signals[ts_id]
                lanes_ts = ts_obj.lanes

                fila_step = sum(ts_obj.sumo.lane.getLastStepHaltingNumber(l) for l in lanes_ts)
                co2_step = sum(ts_obj.sumo.lane.getCO2Emission(l) for l in lanes_ts)

                veiculos_step = []
                for l in lanes_ts:
                    veiculos_step.extend(ts_obj.sumo.lane.getLastStepVehicleIDs(l))

                # NB
                espera_total_step = 0.0
                delay_total_step = 0.0
                if veiculos_step:
                    espera_total_step = sum(ts_obj.sumo.vehicle.getWaitingTime(v) for v in veiculos_step)
                    delay_total_step = sum(ts_obj.sumo.vehicle.getAccumulatedWaitingTime(v) for v in veiculos_step)

                metricas_ep[ts_id]['fila_sum'] += fila_step
                metricas_ep[ts_id]['co2_sum'] += co2_step
                metricas_ep[ts_id]['espera_sum'] += espera_total_step
                metricas_ep[ts_id]['delay_sum'] += delay_total_step

                # throughput
                for lane_out in ts_obj.out_lanes:
                    metricas_ep[ts_id]['throughput_ids'].update(
                        ts_obj.sumo.lane.getLastStepVehicleIDs(lane_out)
                    )

            for ts_id in env.agents:
                if ts_id in ts_selecionados:
                    recompensa_ep[ts_id] += rewards[ts_id]

            observations = next_observations
            step_count += 1

        # grava no csv
        buffer_csv = []
        for ts_id, dados in metricas_ep.items():
            tipo_agente = "MaxPressure" if ts_id in ts_selecionados else "Fixo"
            reward_total = recompensa_ep.get(ts_id, 0.0)
            throughput_total = len(dados['throughput_ids'])
            buffer_csv.append([
                ep, ts_id, tipo_agente, reward_total,
                dados['delay_sum'], dados['espera_sum'], dados['fila_sum'],
                throughput_total, dados['co2_sum'],
            ])
        with open(caminho_csv, mode='a', newline='') as f:
            csv.writer(f).writerows(buffer_csv)

        rec_sel = {t: round(recompensa_ep[t], 2) for t in ts_selecionados}
        print(f"Ep {ep + 1}/{args.episodios} concluido | recompensa: {rec_sel}")

    if env is not None:
        env.close()
    print(f"\ndados salvos em: {caminho_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Max Pressure para comparacao com o MARL.")
    parser.add_argument("--episodios", type=int, default=10,
                        help="Numero de execucoes (com --troca 1, uma rota por execucao).")
    parser.add_argument("--troca", type=int, default=1,
                        help="A cada quantos episodios troca de rota (1 = uma rota por episodio).")
    parser.add_argument("--rotasdir", type=str, default="rotas_jtr_marl",
                        help="Pasta com os arquivos .rou.xml de avaliacao.")
    parser.add_argument("--net", type=str, default="baseSumo/grid.net.xml",
                        help="Arquivo .net.xml da malha.")
    parser.add_argument("--perfil", type=str, default="total",
                        help="Perfil do JSON (quais semaforos o Max Pressure controla).")
    parser.add_argument("--perfis_json", type=str, default="perfis_treinamento_marl_3x3.json",
                        help="Arquivo JSON com os perfis.")
    parser.add_argument("--recompensa", type=str, default="vizinhos", choices=["local", "vizinhos"],
                        help="Recompensa usada apenas para a coluna reward_total (comparabilidade com o RL).")
    parser.add_argument("--gui", action="store_true", help="Abrir a GUI do SUMO.")
    args = parser.parse_args()

    with open(args.perfis_json, "r") as f:
        perfis = json.load(f)
    if args.perfil not in perfis:
        raise KeyError(f"Perfil '{args.perfil}' nao existe em {args.perfis_json}. "
                       f"Disponiveis: {list(perfis.keys())}")
    args.ts_selecionados = perfis[args.perfil]

    print(f"max pressure, perfil: {args.perfil} "
          f"({len(args.ts_selecionados)} semaforos) | recomenpensa no log: {args.recompensa}")
    rodar_baseline(args)