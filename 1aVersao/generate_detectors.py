import sumolib

# Configurações
NET_FILE = "grid.net.xml"
DETECTOR_OFFSET = 15
FREQ = 1
COLOR = "255,105,180"
OUTPUT_FILE = "detectors.out.xml"  # único arquivo para todos os detectores

lanes_horizontais = [
    ("n00_n01", 2), ("n01_n00", 2),
    ("n01_n02", 2), ("n02_n01", 2),
    ("n10_n11", 2), ("n11_n10", 2),
    ("n11_n12", 2), ("n12_n11", 2),
    ("n20_n21", 2), ("n21_n20", 2),
    ("n21_n22", 2), ("n22_n21", 2)
]

lanes_verticais = [
    ("n00_n10", 2), ("n10_n00", 2),
    ("n01_n11", 2), ("n11_n01", 2),
    ("n02_n12", 2), ("n12_n02", 2),
    ("n10_n20", 2), ("n20_n10", 2),
    ("n11_n21", 2), ("n21_n11", 2),
    ("n12_n22", 2), ("n22_n12", 2)
]

# Junta horizontais e verticais
edges_com_duas_lanes = lanes_horizontais + lanes_verticais

# Carrega a rede diretamente com sumolib
net = sumolib.net.readNet(NET_FILE)

all_detectors = []

for edge_id, num_lanes in edges_com_duas_lanes:
    edge = net.getEdge(edge_id)
    for i in range(num_lanes):
        lane = edge.getLane(i)
        lane_id = lane.getID()
        lane_length = lane.getLength()
        detector_pos = max(0, lane_length - DETECTOR_OFFSET)
        detector_id = f"det_{lane_id}"
        all_detectors.append(
            f'<inductionLoop id="{detector_id}" lane="{lane_id}" pos="{detector_pos}" '
            f'freq="{FREQ}" file="{OUTPUT_FILE}" color="{COLOR}" />'
        )

# Gera o XML
add_xml_content = "<additional>\n"
for det in all_detectors:
    add_xml_content += "    " + det + "\n"
add_xml_content += "</additional>\n"

# Salva em arquivo
with open("grid.add.xml", "w") as f:
    f.write(add_xml_content)

print(f"grid.add.xml atualizado com detectores 15m antes das lanes selecionadas! Saída em {OUTPUT_FILE}")

