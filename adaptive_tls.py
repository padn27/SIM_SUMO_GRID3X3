import traci
import csv

# -----------------------------
# Configuração SUMO
# -----------------------------
SUMO_BINARY = "sumo-gui"
SUMO_CONFIG = "grid.sumocfg"

tls_ids = ["n00","n01","n02","n10","n11","n12","n20","n21","n22"]

MAX_STEPS = 3600
MIN_GREEN = 10
MAX_GREEN = 90
BUS_PRIORITY = 17
DETECTOR_RANGE = 15  # metros antes do semáforo

traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])

metrics = {tls_id: {"total_queue": [], "total_wait": [], "phase_changes": 0, "vehicles_passed": 0} for tls_id in tls_ids}
last_green_phase = {tls_id: 0 for tls_id in tls_ids}

step = 0
while step < MAX_STEPS:
    traci.simulationStep()

    for tls_id in tls_ids:
        tls_program = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
        lanes = traci.trafficlight.getControlledLanes(tls_id)

        congestion_per_phase = [0]*len(tls_program.phases)
        green_phases = []

        for i, phase in enumerate(tls_program.phases):
            state = phase.state
            if "G" not in state:
                continue
            green_phases.append(i)
            count = 0
            for lane_index, lane in enumerate(lanes):
                if lane_index >= len(state):
                    continue
                if state[lane_index] == "G":
                    vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                    for v in vehicle_ids:
                        pos = traci.vehicle.getLanePosition(v)
                        if pos >= traci.lane.getLength(lane) - DETECTOR_RANGE:  # dentro do alcance
                            if traci.vehicle.getTypeID(v) == "car":
                                count += 1
                            elif traci.vehicle.getTypeID(v) == "bus":
                                count += BUS_PRIORITY
            congestion_per_phase[i] = count

        if green_phases:
            max_index = max(green_phases, key=lambda x: congestion_per_phase[x])
            duration = min(MAX_GREEN, max(MIN_GREEN, 10 + congestion_per_phase[max_index]*2))

            # Só troca quando a fase atual terminar
            time_remaining = traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
            if time_remaining <= 0:
                if last_green_phase[tls_id] != max_index:
                    traci.trafficlight.setPhase(tls_id, max_index)
                    metrics[tls_id]["phase_changes"] += 1
                    last_green_phase[tls_id] = max_index
                traci.trafficlight.setPhaseDuration(tls_id, duration)

        # Métricas
        ns_queue = traci.lane.getLastStepHaltingNumber(lanes[0])
        ew_queue = traci.lane.getLastStepHaltingNumber(lanes[1])
        ns_wait = traci.lane.getWaitingTime(lanes[0])
        ew_wait = traci.lane.getWaitingTime(lanes[1])

        metrics[tls_id]["total_queue"].append(ns_queue + ew_queue)
        metrics[tls_id]["total_wait"].append(ns_wait + ew_wait)
        metrics[tls_id]["vehicles_passed"] += traci.lane.getLastStepVehicleNumber(lanes[0]) + traci.lane.getLastStepVehicleNumber(lanes[1])

    step += 1

traci.close()
# -----------------------------
# Relatório de TLS ativos/inativos
# -----------------------------
ativos = []
inativos = []

for tls_id in tls_ids:
    if metrics[tls_id]["vehicles_passed"] > 0 or sum(metrics[tls_id]["total_queue"]) > 0:
        ativos.append(tls_id)
    else:
        inativos.append(tls_id)

print("\n=== Relatório de Atividade dos Semáforos ===")
print("Semáforos ATIVOS (houve tráfego detectado):")
for tls in ativos:
    print(f"  - {tls} (veículos: {metrics[tls]['vehicles_passed']}, trocas: {metrics[tls]['phase_changes']})")

print("\nSemáforos INATIVOS (nenhum veículo passou):")
for tls in inativos:
    print(f"  - {tls} (trocas de fase: {metrics[tls]['phase_changes']})")


# Salva métricas em CSV e imprime no terminal
with open("adaptive_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = ["Semáforo (TLS)", "Fila Média", "Fila Máxima", "Espera Média (s)", "Espera Máxima (s)", "Trocas de Fase", "Veículos Passaram"]
    writer.writerow(header)
    print("\t".join(header))
    for tls_id in tls_ids:
        avg_queue = sum(metrics[tls_id]["total_queue"]) / len(metrics[tls_id]["total_queue"])
        max_queue = max(metrics[tls_id]["total_queue"])
        avg_wait = sum(metrics[tls_id]["total_wait"]) / len(metrics[tls_id]["total_wait"])
        max_wait = max(metrics[tls_id]["total_wait"])
        phase_changes = metrics[tls_id]["phase_changes"]
        vehicles_passed = metrics[tls_id]["vehicles_passed"]
        row = [tls_id, round(avg_queue,2), max_queue, round(avg_wait,2), max_wait, phase_changes, vehicles_passed]
        writer.writerow(row)
        print("\t".join([str(x) for x in row]))

print("\nMétricas do adaptativo salvas em adaptive_metrics.csv")

