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
OUT_DIR = "rotas_jtr"
os.makedirs(OUT_DIR, exist_ok=True)


ORIG_INS  = ["n01_n11", "n21_n11", "n10_n11", "n12_n11"]
DEST_OUTS = ["n11_n01", "n11_n21", "n11_n10", "n11_n12"]

LEVEL_TO_NUM = {0: 0, 1: 60, 2: 120, 3: 240}

DISTRIB = {
    "n01_n11": [0.05, 0.80, 0.10, 0.05],  # NS
    "n21_n11": [0.80, 0.05, 0.05, 0.10],  # SN
    "n10_n11": [0.10, 0.10, 0.60, 0.20],  # OL
    "n12_n11": [0.10, 0.20, 0.25, 0.45],  # LO
}

V_TYPES = [
    {"id": "car", "attrs": 'accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="70"'},
    {"id": "bus", "attrs": 'accel="1.5" decel="3.5" sigma="0.5" length="12.0" maxSpeed="50"'},
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
    """Escreve arquivo XML de flows"""
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
            "begin": "0", "end": "3600"
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
    dir_to_orig = {"NS": "n01_n11", "SN": "n21_n11", "LO": "n12_n11", "OL": "n10_n11"}
    flows = []
    for dir_key, lvl in levels.items():
        orig = dir_to_orig[dir_key]
        total = LEVEL_TO_NUM[lvl]
        if total <= 0:
            continue
        dist = DISTRIB.get(orig, [0.25]*4)
        for i, dest in enumerate(DEST_OUTS):
            if (orig == "n10_n11" and dest == "n11_n10") or \
               (orig == "n12_n11" and dest == "n11_n12") or \
               (orig == "n01_n11" and dest == "n11_n01") or \
               (orig == "n21_n11" and dest == "n11_n21"):
                continue
            num = total * dist[i]
            if num < 1:
                continue
            for vt in V_TYPES:
                flows.append({
                    "id": f"{dir_key}_{i}_{vt['id']}",
                    "type": vt["id"],
                    "from": orig,
                    "to": dest,
                    "vehsPerHour": int(num)
                })
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

