#!/usr/bin/env python3
import os
import itertools
import subprocess
from xml.etree import ElementTree as ET
from collections import deque

SUMO_HOME = os.getenv("SUMO_HOME")
if not SUMO_HOME:
    raise EnvironmentError("Defina SUMO_HOME (ex: export SUMO_HOME=/usr/share/sumo)")

JTRROUTER = os.path.join(SUMO_HOME, "bin", "jtrrouter")
NET_FILE = "baseSumo/grid.net.xml"
OUT_DIR = "rotas_jtr_marl"
os.makedirs(OUT_DIR, exist_ok=True)

# niveis de trafego
LEVEL_TO_NUM = {0: 0, 1: 150, 2: 250, 3: 350}


ROUTES_NS = [
    ("n00_n10", "n10_n20"), 
    ("n01_n11", "n11_n21"), 
    ("n02_n12", "n12_n22"), 
    ("n01_n00", "n00_n10"), 
    ("n01_n02", "n02_n12")  
]

ROUTES_SN = [
    ("n20_n10", "n10_n00"), 
    ("n21_n11", "n11_n01"), 
    ("n22_n12", "n12_n02"), 
    ("n21_n20", "n20_n10"),
    ("n21_n22", "n22_n12")  
]

ROUTES_LO = [ 
    ("n02_n01", "n01_n00"), 
    ("n12_n11", "n11_n10"), 
    ("n22_n21", "n21_n20"), 
    ("n12_n02", "n02_n01"), 
    ("n12_n22", "n22_n21")  
]

ROUTES_OL = [ 
    ("n00_n01", "n01_n02"), 
    ("n10_n11", "n11_n12"), 
    ("n20_n21", "n21_n22"), 
    ("n10_n00", "n00_n01"), 
    ("n10_n20", "n20_n21")  
]

#onibus e carros
V_TYPES = [
    {"id": "car", "attrs": 'accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="13.9"'},
    {"id": "bus", "attrs": 'accel="1.2" decel="3.5" sigma="0.5" length="12.0" maxSpeed="10.0"'},
]

def parse_net_connections(netfile):
    tree = ET.parse(netfile)
    root = tree.getroot()
    adj = {}
    for conn in root.findall("connection"):
        frm, to = conn.get("from"), conn.get("to")
        if frm and to:
            adj.setdefault(frm, set()).add(to)
    for edge in root.findall("edge"):
        eid = edge.get("id")
        if eid:
            adj.setdefault(eid, set())
    return adj

def reachable(adj, src, dst):
    if src == dst:
        return True
    if src not in adj or dst not in adj:
        return False
    q, seen = deque([src]), {src}
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v == dst:
                return True
            if v not in seen:
                seen.add(v)
                q.append(v)
    return False

def write_flows(flows, outpath):
    #escreve xml dos flows
    root = ET.Element("routes")
    for vt in V_TYPES:
        attrs = {k: v.strip('"') for k, v in [p.split("=") for p in vt["attrs"].split()]}
        attrs["id"] = vt["id"]
        ET.SubElement(root, "vType", attrs)
    for f in flows:
        ET.SubElement(root, "flow", {
            "id": f["id"], "type": f["type"],
            "from": f["from"], "to": f["to"],
            "vehsPerHour": str(int(f["vehsPerHour"])),
            "begin": "0", "end": "3200"
        })
    ET.ElementTree(root).write(outpath, encoding="utf-8", xml_declaration=True)

def run_jtrrouter(flows_file, rou_file, seed):
    cmd = [JTRROUTER, "-n", NET_FILE, "-r", flows_file, "-o", rou_file, "--seed", str(seed)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERRO] {rou_file} -> {res.stderr.strip()}")
        return False
    print(f"criado {rou_file}")
    return True

def build_flows(levels):
    flows = []
    flow_id_counter = 0
    directions = {"NS": ROUTES_NS, "SN": ROUTES_SN, "LO": ROUTES_LO, "OL": ROUTES_OL}

    #distribuição 20 40 20 10 10
    PESOS_ROTAS = [0.20, 0.40, 0.20, 0.10, 0.10] 

    for dir_key, lvl in levels.items():
        total_vehs = LEVEL_TO_NUM[lvl]
        if total_vehs <= 0:
            continue
            
        routes = directions[dir_key]

        for i, (orig, dest) in enumerate(routes):
            vehs_per_route = total_vehs * PESOS_ROTAS[i]
            if vehs_per_route < 1:
                continue

            # 80% carros e 20% onibus
            for vt in V_TYPES:
                percent = 0.8 if vt["id"] == "car" else 0.2
                flows.append({
                    "id": f"{dir_key}_{flow_id_counter}_{vt['id']}",
                    "type": vt["id"],
                    "from": orig,
                    "to": dest,
                    "vehsPerHour": int(vehs_per_route * percent)
                })
                flow_id_counter += 1
    return flows

def generate_all():
    adj = parse_net_connections(NET_FILE)
    dirs = ["NS", "SN", "LO", "OL"]
    total = 0

    for comb in itertools.product(range(1,4), repeat=4):
        levels = dict(zip(dirs, comb))
        name = "_".join(f"{k}{v}" for k, v in levels.items())

        flows = build_flows(levels)
        valid_flows = [f for f in flows if reachable(adj, f["from"], f["to"])]
        if not valid_flows:
            print(f"nao criou o flow")
            continue

        flow_file = os.path.join(OUT_DIR, f"flows_{name}.xml")
        rou_file  = os.path.join(OUT_DIR, f"routes_{name}.rou.xml")

        write_flows(valid_flows, flow_file)
        run_jtrrouter(flow_file, rou_file, seed=42 + total)
        total += 1

    print(f"fim sem erros")


if __name__ == "__main__":
    generate_all()
