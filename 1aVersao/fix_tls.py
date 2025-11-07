import traci

SUMO_BINARY = "sumo-gui"
SUMO_CONFIG = "grid.sumocfg"

# inicia SUMO
traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])

# Lista dos TLS que têm 1 fase
tls_to_fix = ["n00", "n02", "n20", "n22"]

for tls_id in tls_to_fix:
    # Cria 4 fases corretamente
    phases = [
        traci.trafficlight.Phase(31, "GGrr"),
        traci.trafficlight.Phase(6,  "yyrr"),
        traci.trafficlight.Phase(31, "rrGG"),
        traci.trafficlight.Phase(6,  "rryy")
    ]

    # Cria a lógica completa
    logic = traci.trafficlight.Logic(
        programID="1",
        type="static",
        currentPhaseIndex=0,
        phases=phases
    )

    # Aplica a lógica antes de qualquer simulationStep()
    traci.trafficlight.setCompleteRedYellowGreenDefinition(tls_id, [logic])

# Confirma que todos têm 4 fases
tls_ids = ["n00","n01","n02","n10","n11","n12","n20","n21","n22"]
for tls_id in tls_ids:
    tls_program = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
    print(f"{tls_id} tem {len(tls_program.phases)} fases")

traci.close()
