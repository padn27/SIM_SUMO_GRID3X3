# custom_sumo_rl_env_final.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import sumolib
import json
import os

STATE_SIZE = 8

class CustomSumoEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, sumo_cfg, tls_config, use_gui=False, max_steps=4200, steps_por_decision=10, sumo_binary=None):
        super().__init__()

        self.sumo_cfg = sumo_cfg
        self.tls_config_file = tls_config
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.steps_por_decision = steps_por_decision
        self.current_step = 0
        self.sumo_binary = sumo_binary or sumolib.checkBinary("sumo")

        # Ler configuração TLS
        try:
            with open(self.tls_config_file, "r") as f:
                tls_types = json.load(f)
        except FileNotFoundError:
            raise Exception(f"Arquivo TLS não encontrado: {self.tls_config_file}")

        self.all_tls_ids = list(tls_types.keys())
        self.adaptive_ids = [tls for tls, t in tls_types.items() if t == "A"]
        self.rl_ids = [tls for tls, t in tls_types.items() if t == "R"]
        if not self.rl_ids:
            raise Exception("Nenhum TLS 'R' encontrado para RL.")
        print(f"Agentes RL: {self.rl_ids}, Adaptativos: {self.adaptive_ids}")

        # Espaços de observação e ação
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        # Estado interno
        self.agent_phase_data = {tls: {"current_phase": 0, "time_in_phase": 0} for tls in self.rl_ids}
        self.tls_controlled_lanes = {}
        self.tls_ns_lanes = {}
        self.tls_ew_lanes = {}

        self._init_lane_maps()

    def _init_lane_maps(self):
        # Start temporário para ler as pistas
        port = sumolib.miscutils.getFreeSocketPort()
        traci.start([self.sumo_binary, "-c", self.sumo_cfg], port=port, label="init_lane_map")
        for tls_id in self.all_tls_ids:
            try:
                lanes = traci.trafficlight.getControlledLanes(tls_id)
                self.tls_controlled_lanes[tls_id] = lanes
                self.tls_ns_lanes[tls_id] = [l for i, l in enumerate(lanes) if i % 2 == 0]
                self.tls_ew_lanes[tls_id] = [l for i, l in enumerate(lanes) if i % 2 != 0]
            except traci.TraCIException:
                self.tls_controlled_lanes[tls_id] = []
                self.tls_ns_lanes[tls_id] = []
                self.tls_ew_lanes[tls_id] = []
        traci.close()

    def _start_traci(self):
        cmd = [self.sumo_binary, "-c", self.sumo_cfg,
               "--tripinfo-output", os.path.join(os.path.dirname(self.sumo_cfg), "tripinfo.xml")]
        if self.use_gui:
            cmd.append("--start")
        traci.start(cmd)

    def reset(self, seed=None, options=None):
        if traci.isLoaded():
            traci.close()
        self.current_step = 0
        self.agent_phase_data = {tls: {"current_phase": 0, "time_in_phase": 0} for tls in self.rl_ids}
        self._start_traci()

        # Inicializa fases RL
        for tls_id in self.rl_ids:
            traci.trafficlight.setPhase(tls_id, 0)

        obs = {tls: self._get_agent_state(tls) for tls in self.rl_ids}
        return obs, {}

    def step(self, actions):
        reward = {tls: 0.0 for tls in self.rl_ids}
        terminated = {tls: False for tls in self.rl_ids}
        truncated = {tls: False for tls in self.rl_ids}

        # Aplica ações RL
        for tls_id, action in actions.items():
            self._apply_rl_action(tls_id, action)

        for _ in range(self.steps_por_decision):
            self.current_step += 1
            traci.simulationStep()
            self._handle_adaptive_tls()

            for tls_id in self.rl_ids:
                reward[tls_id] += self._compute_agent_reward(tls_id)

            if traci.simulation.getMinExpectedNumber() <= 0 or self.current_step >= self.max_steps:
                for tls_id in self.rl_ids:
                    terminated[tls_id] = True
                    truncated[tls_id] = True
                break

        obs = {tls: self._get_agent_state(tls) for tls in self.rl_ids}
        return obs, reward, terminated, truncated, {}

    def _get_agent_state(self, tls_id):
        ns_lanes = self.tls_ns_lanes.get(tls_id, [])
        ew_lanes = self.tls_ew_lanes.get(tls_id, [])
        try:
            ns_queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in ns_lanes)
            ew_queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in ew_lanes)
            ns_wait = sum(traci.lane.getWaitingTime(l) for l in ns_lanes)
            ew_wait = sum(traci.lane.getWaitingTime(l) for l in ew_lanes)
            ns_veh = sum(traci.lane.getLastStepVehicleNumber(l) for l in ns_lanes)
            ew_veh = sum(traci.lane.getLastStepVehicleNumber(l) for l in ew_lanes)
        except traci.TraCIException:
            return np.zeros(STATE_SIZE, dtype=np.float32)

        phase = self.agent_phase_data[tls_id]["current_phase"]
        time_in_phase = self.agent_phase_data[tls_id]["time_in_phase"]
        return np.array([ns_queue, ew_queue, ns_wait, ew_wait, phase, time_in_phase, ns_veh, ew_veh], dtype=np.float32)

    def _apply_rl_action(self, tls_id, action):
        phase_data = self.agent_phase_data[tls_id]
        if action == 1:
            phase_data["current_phase"] = 1 - phase_data["current_phase"]
            phase_data["time_in_phase"] = 0
        idx = phase_data["current_phase"] * 2
        traci.trafficlight.setPhase(tls_id, idx)
        self.agent_phase_data[tls_id] = phase_data

    def _handle_adaptive_tls(self):
        for tls_id in self.adaptive_ids:
            lanes = self.tls_controlled_lanes.get(tls_id, [])
            if not lanes:
                continue
            try:
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
                green_phases = [i for i, p in enumerate(logic.phases) if 'g' in p.state.lower()]
                congestion = {}
                for idx in green_phases:
                    state = logic.phases[idx].state
                    queue = sum(traci.lane.getLastStepHaltingNumber(lanes[i]) for i, c in enumerate(state) if c.lower() == 'g' and i < len(lanes))
                    congestion[idx] = queue
                if congestion:
                    best = max(congestion, key=congestion.get)
                    next_switch = traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
                    if traci.trafficlight.getPhase(tls_id) != best and next_switch <= 2:
                        traci.trafficlight.setPhase(tls_id, best)
            except traci.TraCIException:
                continue

    def _compute_agent_reward(self, tls_id):
        lanes = self.tls_controlled_lanes.get(tls_id, [])
        if not lanes:
            return 0.0
        try:
            queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
            wait = sum(traci.lane.getWaitingTime(l) for l in lanes)
            return -(queue + wait)
        except traci.TraCIException:
            return 0.0

    def render(self):
        if not self.use_gui:
            return
        print(f"Step {self.current_step}")

    def close(self):
        if traci.isLoaded():
            traci.close()
        print("Ambiente fechado.")

