import os            # Para manipulação de diretórios e caminhos
import shutil        # Para copiar arquivos
import json          # Para salvar configurações de TLS em JSON
import xml.etree.ElementTree as ET  # Para manipular arquivos XML do SUMO
import random        # Para gerar números aleatórios (fluxos de veículos)

// ===========================
// Configurações básicas
// ===========================
BASE_DIR = os.path.dirname(__file__)  # Diretório do script
CENARIOS_DIR = os.path.join(BASE_DIR, "cenarios")  # Pasta onde os cenários serão gerados
TLS_IDS = ["n00","n01","n02","n10","n11","n12","n20","n21","n22"]  # IDs de todos os TLS (semaforos)

# Arquivos base do SUMO que serão copiados para cada cenário
BASE_CONFIG = os.path.join(BASE_DIR, "grid.sumocfg")
BASE_NET    = os.path.join(BASE_DIR, "grid.net.xml")
BASE_ADD    = os.path.join(BASE_DIR, "grid.add.xml")

os.makedirs(CENARIOS_DIR, exist_ok=True)

TLS_EDGES = {
    "n00": ["n01_n00","n10_n00"],
    "n01": ["n00_n01","n02_n01","n11_n01"],
    "n02": ["n01_n02","n12_n02"],
    "n10": ["n00_n10","n11_n10","n20_n10"],
    "n11": ["n01_n11","n10_n11","n12_n11","n21_n11"],
    "n12": ["n02_n12","n11_n12","n22_n12"],
    "n20": ["n10_n20","n21_n20"],
    "n21": ["n11_n21","n20_n21","n22_n21"],
    "n22": ["n12_n22","n21_n22"]
}

# Intervalos para geração aleatória dos cenários originais
INTERVALOS = {
    "baixo": (100, 300),
    "medio": (301, 700),
    "alto": (701, 1200)
}

# ===================================================
# Cenários originais (aleatórios)
# ===================================================
CENARIOS = [
    # Alto
    ("c1a_1RL_4A_4F","alto", ["n11"], ["n00","n01","n02","n10"], ["n12","n20","n21","n22"]),
    ("c1a_2RL_4A_3F","alto", ["n12"], ["n00","n01","n02","n10"], ["n11","n20","n21"]),
    ("c1a_3RL_4A_2F","alto", ["n11","n12"], ["n00","n01","n02","n10"], ["n20","n21"]),
    ("c1a_4RL_4A_1F","alto", ["n11"], ["n00","n01","n02","n10"], ["n12"]),
    ("c1a_5RL_3A_1F","alto", ["n11"], ["n00","n01","n02"], ["n12"]),
    ("c1a_6RL_2A_1F","alto", ["n12"], ["n01","n02"], ["n11"]),
    ("c1a_7RL_1A_1F","alto", ["n11"], ["n01"], ["n12"]),
    ("c1a_8RL_1A_0F","alto", ["n12"], ["n01"], []),
    ("c1a_9RL_0A_0F","alto", [], [], []),
    # Médio
    ("c1m_1RL_4A_4F","medio", ["n11"], ["n00","n01","n02","n10"], ["n12","n20","n21","n22"]),
    ("c1m_2RL_4A_3F","medio", ["n12"], ["n00","n01","n02","n10"], ["n11","n20","n21"]),
    ("c1m_3RL_4A_2F","medio", ["n11","n12"], ["n00","n01","n02","n10"], ["n20","n21"]),
    ("c1m_4RL_4A_1F","medio", ["n11"], ["n00","n01","n02","n10"], ["n12"]),
    ("c1m_5RL_3A_1F","medio", ["n11"], ["n00","n01","n02"], ["n12"]),
    ("c1m_6RL_2A_1F","medio", ["n12"], ["n01","n02"], ["n11"]),
    ("c1m_7RL_1A_1F","medio", ["n11"], ["n01"], ["n12"]),
    ("c1m_8RL_1A_0F","medio", ["n12"], ["n01"], []),
    ("c1m_9RL_0A_0F","medio", [], [], []),
    # Baixo
    ("c1b_1RL_4A_4F","baixo", ["n11"], ["n00","n01","n02","n10"], ["n12","n20","n21","n22"]),
    ("c1b_2RL_4A_3F","baixo", ["n12"], ["n00","n01","n02","n10"], ["n11","n20","n21"]),
    ("c1b_3RL_4A_2F","baixo", ["n11","n12"], ["n00","n01","n02","n10"], ["n20","n21"]),
    ("c1b_4RL_4A_1F","baixo", ["n11"], ["n00","n01","n02","n10"], ["n12"]),
    ("c1b_5RL_3A_1F","baixo", ["n11"], ["n00","n01","n02"], ["n12"]),
    ("c1b_6RL_2A_1F","baixo", ["n12"], ["n01","n02"], ["n11"]),
    ("c1b_7RL_1A_1F","baixo", ["n11"], ["n01"], ["n12"]),
    ("c1b_8RL_1A_0F","baixo", ["n12"], ["n01"], []),
    ("c1b_9RL_0A_0F","baixo", [], [], [])
]

# ===================================================
# Cenários fixos (mantêm quantidade)
# ===================================================
CENARIOS_FIXOS_BAIXO = [
    ("c1p_1RL_4A_4F", 280, ["n11"], ["n00","n01","n02","n10"], ["n12","n20","n21","n22"]),
    ("c1p_2RL_4A_3F", 280, ["n12"], ["n00","n01","n02","n10"], ["n11","n20","n21"]),
    ("c1p_3RL_4A_2F", 280, ["n11","n12"], ["n00","n01","n02","n10"], ["n20","n21"]),
    ("c1p_4RL_4A_1F", 280, ["n11"], ["n00","n01","n02","n10"], ["n12"]),
    ("c1p_5RL_3A_1F", 280, ["n11"], ["n00","n01","n02"], ["n12"]),
    ("c1p_6RL_2A_1F", 280, ["n12"], ["n01","n02"], ["n11"]),
    ("c1p_7RL_1A_1F", 280, ["n11"], ["n01"], ["n12"]),
    ("c1p_8RL_1A_0F", 280, ["n12"], ["n01"], []),
    ("c1p_9RL_0A_0F", 280, [], [], [])
]

CENARIOS_FIXOS_MEDIO = [
    ("c1i_1RL_4A_4F", 589, ["n11"], ["n00","n01","n02","n10"], ["n12","n20","n21","n22"]),
    ("c1i_2RL_4A_3F", 589, ["n12"], ["n00","n01","n02","n10"], ["n11","n20","n21"]),
    ("c1i_3RL_4A_2F", 589, ["n11","n12"], ["n00","n01","n02","n10"], ["n20","n21"]),
    ("c1i_4RL_4A_1F", 589, ["n11"], ["n00","n01","n02","n10"], ["n12"]),
    ("c1i_5RL_3A_1F", 589, ["n11"], ["n00","n01","n02"], ["n12"]),
    ("c1i_6RL_2A_1F", 589, ["n12"], ["n01","n02"], ["n11"]),
    ("c1i_7RL_1A_1F", 589, ["n11"], ["n01"], ["n12"]),
    ("c1i_8RL_1A_0F", 589, ["n12"], ["n01"], []),
    ("c1i_9RL_0A_0F", 589, [], [], [])
]

CENARIOS_FIXOS_ALTO = [
    ("c1g_1RL_4A_4F", 988, ["n11"], ["n00","n01","n02","n10"], ["n12","n20","n21","n22"]),
    ("c1g_2RL_4A_3F", 988, ["n12"], ["n00","n01","n02","n10"], ["n11","n20","n21"]),
    ("c1g_3RL_4A_2F", 988, ["n11","n12"], ["n00","n01","n02","n10"], ["n20","n21"]),
    ("c1g_4RL_4A_1F", 988, ["n11"], ["n00","n01","n02","n10"], ["n12"]),
    ("c1g_5RL_3A_1F", 988, ["n11"], ["n00","n01","n02"], ["n12"]),
    ("c1g_6RL_2A_1F", 988, ["n12"], ["n01","n02"], ["n11"]),
    ("c1g_7RL_1A_1F", 988, ["n11"], ["n01"], ["n12"]),
    ("c1g_8RL_1A_0F", 988, ["n12"], ["n01"], []),
    ("c1g_9RL_0A_0F", 988, [], [], [])
]

# ===================================================
# Cenários adicionais adaptativos e fixos
# ===================================================
CENARIOS_ADAPTATIVO_NOVO = [
    ("c1s_0RL_9A_0F", 280),
    ("c1y_0RL_9A_0F", 589),
    ("c1e_0RL_9A_0F", 988)
]

CENARIOS_FIXOS_NOVO = [
    ("c1s_0RL_0A_9F", 280),
    ("c1y_0RL_0A_9F", 589),
    ("c1e_0RL_0A_9F", 988)
]

// ===================================================
// Função para criar cenários
// ===================================================
def criar_cenario(nome, carros_total, RL, A, F, tipo=None):
    # Cria diretório do cenário
    scenario_path = os.path.join(CENARIOS_DIR, nome)
    os.makedirs(scenario_path, exist_ok=True)

    # Copia arquivos base do SUMO para cada cenário
    shutil.copy(BASE_CONFIG, os.path.join(scenario_path, "grid.sumocfg"))
    shutil.copy(BASE_NET, os.path.join(scenario_path, "grid.net.xml"))
    shutil.copy(BASE_ADD, os.path.join(scenario_path, "grid.add.xml"))

    # Define 10% do total de carros como ônibus
    onibus_total = int(0.1 * carros_total)

    # Lista com todos os TLS usados nesse cenário
    all_tls = RL + A + F
    if not all_tls:  # Se não houver TLS definido, usa n00 como padrão
        all_tls = ["n00"]

    # Divide carros e ônibus entre os TLS
    carros_por_tls = [carros_total//len(all_tls)]*len(all_tls)
    onibus_por_tls = [onibus_total//len(all_tls)]*len(all_tls)
    for i in range(carros_total%len(all_tls)):
        carros_por_tls[i] += 1
    for i in range(onibus_total%len(all_tls)):
        onibus_por_tls[i] += 1

    # Cria arquivo .rou.xml (rotas do SUMO)
    rou_file = os.path.join(scenario_path, f"{nome}.rou.xml")
    with open(rou_file, "w", encoding="utf-8") as f:
        f.write('<routes>\n')
        # Define tipos de veículo
        f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="70"/>\n')
        f.write('    <vType id="bus" accel="1.5" decel="3.5" sigma="0.5" length="12.0" maxSpeed="50"/>\n\n')
        # Cria fluxos de veículos por TLS
        for idx, tls in enumerate(all_tls):
            c = carros_por_tls[idx]
            o = onibus_por_tls[idx]
            edges = TLS_EDGES.get(tls, ["n00_n01"])
            from_edge = edges[0]
            to_edge = edges[-1]
            if c > 0: f.write(f'    <flow id="{tls}_car_{idx}" type="car" from="{from_edge}" to="{to_edge}" vehsPerHour="{c}" />\n')
            if o > 0: f.write(f'    <flow id="{tls}_bus_{idx}" type="bus" from="{from_edge}" to="{to_edge}" vehsPerHour="{o}" />\n')
        f.write('</routes>\n')

    # Atualiza arquivo .sumocfg para apontar para o novo .rou.xml
    tree = ET.parse(os.path.join(scenario_path, "grid.sumocfg"))
    root = tree.getroot()
    input_tag = root.find("input") or ET.SubElement(root, "input")
    for r in input_tag.findall("route-files"): input_tag.remove(r)
    route_tag = ET.SubElement(input_tag, "route-files")
    route_tag.set("value", f"{nome}.rou.xml")
    tree.write(os.path.join(scenario_path, "grid.sumocfg"), encoding="utf-8", xml_declaration=True)

    # Salva configuração de cada TLS (R, A ou F) em JSON
    tls_config = {tls_id: ("R" if tls_id in RL else "A" if tls_id in A else "F") for tls_id in TLS_IDS}
    with open(os.path.join(scenario_path, "tls_config.json"), "w") as jfile:
        json.dump(tls_config, jfile, indent=4)

    # Mensagem de sucesso
    if tipo:
        print(f"Cenário {nome} ({tipo}) criado: {carros_total} carros, {onibus_total} ônibus")
    else:
        print(f"Cenário {nome} criado: {carros_total} carros, {onibus_total} ônibus")

// ===================================================
// Geração de cenários
// ===================================================
# Cria cenários originais com número aleatório de veículos
for nome, tipo, RL, A, F in CENARIOS:
    carros_total = random.randint(*INTERVALOS[tipo])
    criar_cenario(nome, carros_total, RL, A, F, tipo=tipo)

# Cria cenários fixos (quantidade definida) – baixo, médio e alto
for nome, carros, RL, A, F in CENARIOS_FIXOS_BAIXO:
    criar_cenario(nome, carros, RL, A, F, tipo="pequeno")

for nome, carros, RL, A, F in CENARIOS_FIXOS_MEDIO:
    criar_cenario(nome, carros, RL, A, F, tipo="intermediario")

for nome, carros, RL, A, F in CENARIOS_FIXOS_ALTO:
    criar_cenario(nome, carros, RL, A, F, tipo="grande")

# Cria cenários adicionais adaptativos (todos os TLS adaptativos)
for nome, carros_total in CENARIOS_ADAPTATIVO_NOVO:
    criar_cenario(nome, carros_total, RL=[], A=TLS_IDS, F=[], tipo="adaptativo")

# Cria cenários adicionais fixos (todos os TLS fixos)
for nome, carros_total in CENARIOS_FIXOS_NOVO:
    criar_cenario(nome, carros_total, RL=[], A=[], F=TLS_IDS, tipo="fixo")

print("\nTodos os cenários criados com sucesso.")
