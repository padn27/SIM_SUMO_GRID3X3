import traci
import os

CFG_FILE = "grid.sumocfg"
SUMO_BINARY = "sumo"  # ou "sumo-gui" se quiser visualizar

def contar_estados_acoes():
    traci.start([SUMO_BINARY, "-c", CFG_FILE])
    results = []
    tls_ids = traci.trafficlight.getIDList()
    for tls_id in tls_ids:
        # lanes controladas
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        num_states = len(lanes) * 2  # fila + espera

        # ações = nº de fases do programa semafórico
        program = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
        num_actions = len(program.phases)

        results.append((tls_id, len(lanes), num_states, num_actions))
    traci.close()
    return results

if __name__ == "__main__":
    tabela = contar_estados_acoes()
    print("TLS | #Lanes | #Estados | #Ações")
    print("-" * 40)
    for tls_id, n_lanes, n_states, n_actions in tabela:
        print(f"{tls_id:3} | {n_lanes:7} | {n_states:8} | {n_actions:7}")

