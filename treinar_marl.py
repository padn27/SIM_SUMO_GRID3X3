#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from sumo_rl import parallel_env 
import matplotlib.pyplot as plt
import multiprocessing
import time
import json
import csv

# Parâmetros rede
GAMMA = 0.95
LR = 0.001
FIXED_EPSILON = 0.3
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.95    #epsilon decai por episodio
MEMORY_SIZE = 20000
BATCH_SIZE = 64
DELTA_TIME = 30  
TAU = 0.005             # soft update

# ts é o semáforo específico sendo avaliado
def minha_recompensa(ts):
    alpha = 0.8     # peso para o tempo médio de espera
    beta = 1.0      # peso para o comprimento da fila
    gamma = 0.005  # peso para emissão de CO2

    lanes = ts.lanes
    
    #tempo medio de espera W
    all_vehicles = []
    for l in lanes:
        all_vehicles.extend(ts.sumo.lane.getLastStepVehicleIDs(l))

    if not all_vehicles:
        W = 0.0
    else:
        total_wait = sum(ts.sumo.vehicle.getWaitingTime(vid) for vid in all_vehicles)
        W = total_wait / len(all_vehicles)

    #comprimento total das filas Q
    Q = sum(ts.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)

    #emissao de CO2 E
    E = sum(ts.sumo.lane.getCO2Emission(l) for l in lanes)

    #retorno recompensa total
    reward = -(alpha * W + beta * Q + gamma * E)

    return float(reward)

# DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQNAgent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.tau = TAU
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START #ou FIXED_EPSILON, comentar tambem decaimento na linha 256 ou trocar para FIXED_EPSILON - 10/03
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
        states = torch.tensor(np.array(states), dtype=torch.float32)
        
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target = rewards + (1 - dones) * GAMMA * next_q
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

# Treinamento
def treinar_agente(agent_id, args):
    time.sleep(agent_id * 2)

    USE_GUI = args.validacao if agent_id == 0 else False
    NUM_EPISODES = args.episodios
    TROCA = args.troca
    NET_FILE = args.net
    ROTAS_DIR = args.rotasdir

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    graficos_dir = "Graficos_marl"
    os.makedirs(graficos_dir, exist_ok=True)

    dados_csv_dir = "DadosCSV"
    perfil_dir = os.path.join(dados_csv_dir, args.perfil)
    os.makedirs(perfil_dir, exist_ok=True)

    #criacao csv dados - brutos por episodio
    caminho_csv = os.path.join(perfil_dir, f"metricas_marl_proc_{agent_id}.csv")
    with open(caminho_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        #cabecalho
        writer.writerow(["episodio", "semaforo", "tipo", "reward_total", "delay_total", "espera_total", "fila_total", "throughput_total", "co2_total"])

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title(f"Recompensa por Episódio - MARL (Processo {agent_id})")
    ax1.set_xlabel("Episódio")
    ax1.set_ylabel("Recompensa")
    ax1.grid(True)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.set_title(f"Média Móvel da Recompensa - MARL (Processo {agent_id})")
    ax2.set_xlabel("Episódio")
    ax2.set_ylabel("Recompensa Média")
    ax2.grid(True)

    linhas_recompensa = {}
    linhas_media = {}
    cores_grafico = plt.cm.tab10.colors 

    def atualiza_graficos(hist_req, hist_med):
        max_len = 0
        min_y1, max_y1 = float('inf'), float('-inf')
        min_y2, max_y2 = float('inf'), float('-inf')

        for ts_id in hist_req:
            arr = np.array(hist_req[ts_id])
            m_arr = np.array(hist_med[ts_id])
            
            if len(arr) == 0: continue
            
            linhas_recompensa[ts_id].set_data(range(len(arr)), arr)
            linhas_media[ts_id].set_data(range(len(m_arr)), m_arr)
            
            max_len = max(max_len, len(arr))
            min_y1, max_y1 = min(min_y1, np.min(arr)), max(max_y1, np.max(arr))
            min_y2, max_y2 = min(min_y2, np.min(m_arr)), max(max_y2, np.max(m_arr))

        ax1.set_xlim(0, max(10, max_len))
        if min_y1 != float('inf'): ax1.set_ylim(min_y1 - 1, max_y1 + 1)
        
        ax2.set_xlim(0, max(10, max_len))
        if min_y2 != float('inf'): ax2.set_ylim(min_y2 - 1, max_y2 + 1)

    route_files = sorted([os.path.join(ROTAS_DIR, f) for f in os.listdir(ROTAS_DIR) if f.endswith(".rou.xml")])
    if not route_files:
        raise FileNotFoundError(f"Nenhum arquivo .rou.xml encontrado em {ROTAS_DIR}")
    
    current_route_idx = 0
    env = None
    dqn_agents = {} 
    
    historico_recompensas = {ts: [] for ts in args.ts_selecionados}
    historico_media_movel = {ts: [] for ts in args.ts_selecionados}
    recompensas_acumuladas = {ts: 0.0 for ts in args.ts_selecionados}

    for ep in range(NUM_EPISODES):
        if ep % TROCA == 0:
            if env is not None:
                env.close()
            rota_atual = route_files[current_route_idx]
            print(f"\n[Processo {agent_id} | Ep {ep}] -> {os.path.basename(rota_atual)}")
            
            env = parallel_env(
                net_file=NET_FILE,
                route_file=rota_atual,
                use_gui=USE_GUI,
                num_seconds=3600,
                delta_time=DELTA_TIME,
                reward_fn=minha_recompensa,
                sumo_warnings=False
            )
            current_route_idx = (current_route_idx + 1) % len(route_files)
            
            #epsilon ciclico, na troca de rota ele é aumentado para que a exploração nunca acabe devido a grande quantidade de episodios
            if dqn_agents: 
                print("\ntroca de rota - reset epsilon ciclico")
                for ts_id in dqn_agents:
                    dqn_agents[ts_id].epsilon = 0.15

        observations, infos = env.reset()
        
        #agentes e redes para os semaforos definidos no JSON
        if not dqn_agents:
            for i, ts_id in enumerate(args.ts_selecionados):
                state_size = env.observation_space(ts_id).shape[0]
                action_size = env.action_space(ts_id).n
                dqn_agents[ts_id] = DQNAgent(state_size, action_size)
                
                cor = cores_grafico[i % len(cores_grafico)]
                linhas_recompensa[ts_id], = ax1.plot([], [], label=f"Semáforo {ts_id}", color=cor, alpha=0.8)
                linhas_media[ts_id], = ax2.plot([], [], label=f"Semáforo {ts_id}", color=cor, linewidth=2)
            
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            fig1.tight_layout()
            fig2.tight_layout()

        recompensa_ep_agentes = {ts_id: 0.0 for ts_id in args.ts_selecionados}
        step_count = 0
        
        #vetor de soma das metricas para cada semaforo ao longo dos steps de um mesmo episodio
        # =======================================================
        # ACUMULADOR DE TOTAIS DO EPISÓDIO
        # =======================================================
        metricas_ep = {}

        while env.agents:
            actions = {}
            for ts_id in env.agents:
                if ts_id in args.ts_selecionados:
                    state = np.array(observations[ts_id], dtype=np.float32)
                    actions[ts_id] = dqn_agents[ts_id].select_action(state)
                else:
                    num_phases = env.action_space(ts_id).n
                    actions[ts_id] = step_count % num_phases
            
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            #acumulacao a cada step
            for ts_id in env.agents:
                if ts_id not in metricas_ep:
                    metricas_ep[ts_id] = {'fila_sum': 0.0, 'co2_sum': 0.0, 'espera_sum': 0.0, 'delay_sum': 0.0, 'throughput': 0}

                base_sumo_env = env.unwrapped.env if hasattr(env.unwrapped, 'env') else env.unwrapped
                ts_obj = base_sumo_env.traffic_signals[ts_id]
                lanes_ts = ts_obj.lanes
                
                #soma instantanea co2 e fila
                fila_step = sum(ts_obj.sumo.lane.getLastStepHaltingNumber(l) for l in lanes_ts)
                co2_step = sum(ts_obj.sumo.lane.getCO2Emission(l) for l in lanes_ts)
                
                veiculos_step = []
                for l in lanes_ts:
                    veiculos_step.extend(ts_obj.sumo.lane.getLastStepVehicleIDs(l))
                
                #soma tempo de espera e delay
                if not veiculos_step:
                    espera_total_step = 0.0
                    delay_total_step = 0.0
                else:
                    espera_total_step = sum(ts_obj.sumo.vehicle.getWaitingTime(v) for v in veiculos_step)
                    delay_total_step = sum(ts_obj.sumo.vehicle.getTimeLoss(v) for v in veiculos_step)
                
                # throughput (Contamos a quantidade de carros "vistos" neste step nas ruas do semaforo)
                metricas_ep[ts_id]['throughput'] += len(veiculos_step)
                
                #acumula no episodio
                metricas_ep[ts_id]['fila_sum'] += fila_step
                metricas_ep[ts_id]['co2_sum'] += co2_step
                metricas_ep[ts_id]['espera_sum'] += espera_total_step
                metricas_ep[ts_id]['delay_sum'] += delay_total_step

            
            for ts_id in env.agents:
                if ts_id in args.ts_selecionados:
                    state = np.array(observations[ts_id], dtype=np.float32)
                    next_state = np.array(next_observations[ts_id], dtype=np.float32)
                    action = actions[ts_id]
                    reward = rewards[ts_id]
                    done = terminations[ts_id] or truncations[ts_id]
                    
                    dqn_agents[ts_id].remember(state, action, reward, next_state, done)
                    dqn_agents[ts_id].replay()
                    dqn_agents[ts_id].update_target()
                    
                    recompensa_ep_agentes[ts_id] += reward 
            
            observations = next_observations
            step_count += 1

        #salvar no csv apos episodio
        buffer_csv = []
        for ts_id, dados in metricas_ep.items():
            tipo_agente = "RL" if ts_id in args.ts_selecionados else "Fixo"
            reward_total = recompensa_ep_agentes.get(ts_id, 0.0) 
            
            # Sem divisão! Pegamos a soma direta que foi acumulada
            fila_total_ep = dados['fila_sum']
            espera_total_ep = dados['espera_sum']
            delay_total_ep = dados['delay_sum']
            co2_total_ep = dados['co2_sum'] 
            throughput_total = dados['throughput']
            
            buffer_csv.append([ep, ts_id, tipo_agente, reward_total, delay_total_ep, espera_total_ep, fila_total_ep, throughput_total, co2_total_ep])
            
        with open(caminho_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(buffer_csv)

        #atualizaçao decaimento, comentar p/ decaimento fixo ou alterar para FIXED_EPSILON
        for ts_id in dqn_agents:
            dqn_agents[ts_id].epsilon = max(EPSILON_END, dqn_agents[ts_id].epsilon * EPSILON_DECAY) #ou FIXED_EPSILON

        #graficos
        for ts_id in args.ts_selecionados:
            r_ep = recompensa_ep_agentes[ts_id]
            historico_recompensas[ts_id].append(r_ep)
            
            recompensas_acumuladas[ts_id] += r_ep
            media_atual = recompensas_acumuladas[ts_id] / (ep + 1)
            historico_media_movel[ts_id].append(media_atual)

        recompensas_outros = {t: round(v, 2) for t, v in recompensa_ep_agentes.items() if t != 'n11'}
        print(f"Proc {agent_id} Ep {ep+1}/{NUM_EPISODES} | Rec. Central (n11): {recompensa_ep_agentes.get('n11', 0):.2f}")
        print(f"Proc {agent_id} Ep {ep+1}/{NUM_EPISODES} | Rec. Outros: {recompensas_outros}")

        if (ep + 1) % 50 == 0:
            for ts_id, agent in dqn_agents.items():
                nome_arquivo = f"{checkpoint_dir}/dqn_proc_{agent_id}_ts_{ts_id}_checkpoint.pth"   
                torch.save(agent.policy_net.state_dict(), nome_arquivo)

        atualiza_graficos(historico_recompensas, historico_media_movel)

        if ep % 10 == 0:
            fig1.savefig(os.path.join(graficos_dir, f"grafico_recompensa_total_marl_{agent_id}.png"))
            fig2.savefig(os.path.join(graficos_dir, f"grafico_media_movel_marl_{agent_id}.png"))

    if env:
        env.close()

    for ts_id, agent in dqn_agents.items():
        torch.save(agent.policy_net.state_dict(), f"{checkpoint_dir}/dqn_proc_{agent_id}_ts_{ts_id}_final.pth")

    fig1.savefig(os.path.join(graficos_dir, f"grafico_recompensa_total_marl_{agent_id}.png"))
    fig2.savefig(os.path.join(graficos_dir, f"grafico_media_movel_marl_{agent_id}.png"))
    
    plt.close(fig1)
    plt.close(fig2)
    print(f"Processo {agent_id} finalizado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validacao", action="store_true")
    parser.add_argument("--episodios", type=int, default=100)
    parser.add_argument("--troca", type=int, default=10)
    parser.add_argument("--rotasdir", type=str, default="rotas_jtr_marl") 
    parser.add_argument("--net", type=str, default="baseSumo/grid.net.xml")
    parser.add_argument("--agentes", type=int, default=1, help="Quantidade de simulações em paralelo")
    parser.add_argument("--perfil", type=str, default="perfil_cruz", help="Qual perfil do JSON usar")
    args = parser.parse_args()
    
    with open("perfis_treinamento_marl_3x3.json", "r") as f:
        perfis = json.load(f)
    
    args.ts_selecionados = perfis[args.perfil]

    if args.agentes > 1:
        print(f"Iniciando {args.agentes} processos de simulação em paralelo (Perfil: {args.perfil})...")
        processos = [multiprocessing.Process(target=treinar_agente, args=(i, args)) for i in range(args.agentes)]
        for p in processos: p.start()
        for p in processos: p.join()
        print("Todos os processos finalizaram.")
    else:
        print(f"Iniciando treinamento único (Perfil: {args.perfil})...")
        treinar_agente(0, args)
