#!/usr/bin/env python3
import os
import itertools
import subprocess
import argparse
import csv
from xml.etree import ElementTree as ET
from collections import deque, defaultdict

# ============================================================
# Gerador de rotas MARL para grid 4x3 / nXY
#
# Versao corrigida:
#   - usa DUAROUTER em vez de JTRROUTER para fechar caminhos OD;
#   - gera candidatos origem-destino automaticamente pelas bordas;
#   - valida os pares OD com o proprio DUAROUTER antes de gerar cenarios;
#   - remove pares impossiveis para evitar erros como:
#       Mandatory edge ... not reachable
#       The vehicle ... has no valid route
# ============================================================

SUMO_HOME = os.getenv("SUMO_HOME")
if not SUMO_HOME:
    raise EnvironmentError("Defina SUMO_HOME. Exemplo: export SUMO_HOME=/usr/share/sumo")

DUAROUTER = os.path.join(SUMO_HOME, "bin", "duarouter")

DEFAULT_NET_FILE = "baseSumo4x3/grid.net.xml"
DEFAULT_OUT_DIR = "rotas_jtr_marl_4x3"

LEVEL_TO_NUM = {0: 0, 1: 150, 2: 250, 3: 350}

V_TYPES = [
    {
        "id": "car",
        "attrs": {
            "accel": "2.6",
            "decel": "4.5",
            "sigma": "0.5",
            "length": "5.0",
            "maxSpeed": "13.9",
        },
        "share": 0.80,
    },
    {
        "id": "bus",
        "attrs": {
            "accel": "1.2",
            "decel": "3.5",
            "sigma": "0.5",
            "length": "12.0",
            "maxSpeed": "10.0",
        },
        "share": 0.20,
    },
]

# N = norte/topo, S = sul/baixo, L = leste/direita, O = oeste/esquerda
SIDES = ["N", "S", "L", "O"]
OPPOSITE = {"N": "S", "S": "N", "O": "L", "L": "O"}
STRAIGHT_SHARE = 0.40
TURN_SHARE = 0.30


def node_to_rc(node_id):
    """Converte IDs do tipo n00, n11, n23 em (linha, coluna)."""
    if not node_id.startswith("n") or len(node_id) != 3:
        return None
    try:
        return int(node_id[1]), int(node_id[2])
    except ValueError:
        return None


def parse_net(netfile):
    tree = ET.parse(netfile)
    root = tree.getroot()

    edges = {}
    nodes = set()

    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if not edge_id or edge_id.startswith(":"):
            continue

        frm = edge.get("from")
        to = edge.get("to")
        if not frm or not to:
            continue

        if node_to_rc(frm) is None or node_to_rc(to) is None:
            continue

        edges[edge_id] = {"from": frm, "to": to}
        nodes.add(frm)
        nodes.add(to)

    if not edges:
        raise RuntimeError(f"Nenhuma aresta externa valida encontrada em {netfile}")

    rc_values = [node_to_rc(n) for n in nodes if node_to_rc(n) is not None]
    rows = sorted({r for r, _ in rc_values})
    cols = sorted({c for _, c in rc_values})

    adj = {eid: set() for eid in edges}
    for conn in root.findall("connection"):
        frm = conn.get("from")
        to = conn.get("to")
        if frm in edges and to in edges:
            adj.setdefault(frm, set()).add(to)

    return edges, adj, rows, cols


def reachable_by_connections(adj, src, dst):
    if src == dst:
        return True
    if src not in adj or dst not in adj:
        return False

    q = deque([src])
    seen = {src}
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v == dst:
                return True
            if v not in seen:
                seen.add(v)
                q.append(v)
    return False


def detect_entry_side(edge_data, rows, cols):
    """
    Detecta arestas que entram na rede a partir da borda.
    Exemplos:
      n00_n10 -> N
      n23_n13 -> S
      n00_n01 -> O
      n13_n12 -> L
    """
    r1, c1 = node_to_rc(edge_data["from"])
    r2, c2 = node_to_rc(edge_data["to"])
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    if r1 == min_r and r2 == min_r + 1 and c1 == c2:
        return "N"
    if r1 == max_r and r2 == max_r - 1 and c1 == c2:
        return "S"
    if c1 == min_c and c2 == min_c + 1 and r1 == r2:
        return "O"
    if c1 == max_c and c2 == max_c - 1 and r1 == r2:
        return "L"
    return None


def detect_exit_side(edge_data, rows, cols):
    """
    Detecta arestas que saem da rede por alguma borda.
    Exemplos:
      n10_n00 -> N
      n13_n23 -> S
      n01_n00 -> O
      n12_n13 -> L
    """
    r1, c1 = node_to_rc(edge_data["from"])
    r2, c2 = node_to_rc(edge_data["to"])
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    if r2 == min_r and r1 == min_r + 1 and c1 == c2:
        return "N"
    if r2 == max_r and r1 == max_r - 1 and c1 == c2:
        return "S"
    if c2 == min_c and c1 == min_c + 1 and r1 == r2:
        return "O"
    if c2 == max_c and c1 == max_c - 1 and r1 == r2:
        return "L"
    return None


def build_raw_catalog(edges, adj, rows, cols, allow_same_side=False):
    entry_edges = defaultdict(list)
    exit_edges = defaultdict(list)

    for edge_id, data in edges.items():
        entry_side = detect_entry_side(data, rows, cols)
        exit_side = detect_exit_side(data, rows, cols)
        if entry_side:
            entry_edges[entry_side].append(edge_id)
        if exit_side:
            exit_edges[exit_side].append(edge_id)

    for side in SIDES:
        entry_edges[side] = sorted(entry_edges[side])
        exit_edges[side] = sorted(exit_edges[side])

    catalog = {s: {t: [] for t in SIDES} for s in SIDES}
    for entry_side in SIDES:
        for exit_side in SIDES:
            if not allow_same_side and entry_side == exit_side:
                continue
            for orig in entry_edges[entry_side]:
                for dest in exit_edges[exit_side]:
                    if reachable_by_connections(adj, orig, dest):
                        catalog[entry_side][exit_side].append((orig, dest))

    return catalog, entry_edges, exit_edges


def write_vtypes(root):
    for vt in V_TYPES:
        attrs = dict(vt["attrs"])
        attrs["id"] = vt["id"]
        ET.SubElement(root, "vType", attrs)


def run_duarouter(net_file, input_file, output_file, seed=42, ignore_errors=False):
    cmd = [DUAROUTER, "-n", net_file, "-r", input_file, "-o", output_file, "--seed", str(seed)]
    if ignore_errors:
        cmd.extend(["--ignore-errors", "true"])
    return subprocess.run(cmd, capture_output=True, text=True)


def validate_catalog_with_duarouter(net_file, catalog, out_dir):
    candidates = []
    for entry_side in SIDES:
        for exit_side in SIDES:
            if entry_side == exit_side:
                continue
            for idx, (orig, dest) in enumerate(catalog[entry_side][exit_side]):
                cand_id = f"cand_{entry_side}_{exit_side}_{idx}"
                candidates.append((cand_id, entry_side, exit_side, orig, dest))

    if not candidates:
        raise RuntimeError("Nenhum candidato OD foi gerado antes da validacao.")

    tmp_dir = os.path.join(out_dir, "_validacao_od")
    os.makedirs(tmp_dir, exist_ok=True)
    trips_file = os.path.join(tmp_dir, "candidatos.trips.xml")
    routes_file = os.path.join(tmp_dir, "candidatos_validos.rou.xml")

    root = ET.Element("routes")
    write_vtypes(root)
    for depart, (cand_id, _, _, orig, dest) in enumerate(candidates):
        ET.SubElement(root, "trip", {
            "id": cand_id,
            "type": "car",
            "depart": str(depart),
            "from": orig,
            "to": dest,
        })

    ET.indent(root, space="    ")
    ET.ElementTree(root).write(trips_file, encoding="utf-8", xml_declaration=True)

    res = run_duarouter(net_file, trips_file, routes_file, seed=123, ignore_errors=True)
    if res.returncode != 0:
        print("[ERRO] DUAROUTER falhou na validacao OD.")
        print(res.stderr.strip())
        raise RuntimeError("Falha na validacao dos pares OD.")

    valid_ids = set()
    try:
        tree = ET.parse(routes_file)
        root_out = tree.getroot()
        for veh in root_out.findall("vehicle"):
            veh_id = veh.get("id")
            if veh_id:
                valid_ids.add(veh_id)
    except ET.ParseError as e:
        raise RuntimeError(f"Nao foi possivel ler {routes_file}: {e}")

    filtered = {s: {t: [] for t in SIDES} for s in SIDES}
    removed = []
    for cand_id, entry_side, exit_side, orig, dest in candidates:
        if cand_id in valid_ids:
            filtered[entry_side][exit_side].append((orig, dest))
        else:
            removed.append((entry_side, exit_side, orig, dest))

    print("\nValidacao OD com DUAROUTER:")
    print(f"  Candidatos testados: {len(candidates)}")
    print(f"  Pares validos:       {len(valid_ids)}")
    print(f"  Pares removidos:     {len(removed)}")

    removed_path = os.path.join(out_dir, "pares_od_removidos.csv")
    with open(removed_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entrada", "saida", "origem", "destino"])
        for row in removed:
            writer.writerow(row)
    print(f"  Pares removidos salvos em: {removed_path}\n")
    return filtered


def target_side_shares(entry_side, available_target_sides):
    raw = {}
    for target in available_target_sides:
        if target == entry_side:
            continue
        raw[target] = STRAIGHT_SHARE if target == OPPOSITE[entry_side] else TURN_SHARE

    total = sum(raw.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in raw.items()}


def build_flows(levels, catalog):
    flows = []
    flow_id_counter = 0

    for entry_side, level in levels.items():
        total_vehs = LEVEL_TO_NUM[level]
        if total_vehs <= 0:
            continue

        available_targets = [
            target_side
            for target_side, routes in catalog[entry_side].items()
            if target_side != entry_side and len(routes) > 0
        ]
        shares = target_side_shares(entry_side, available_targets)

        for target_side, side_share in shares.items():
            routes = catalog[entry_side][target_side]
            if not routes:
                continue

            vehs_for_movement = total_vehs * side_share
            vehs_per_route = vehs_for_movement / len(routes)

            for orig, dest in routes:
                for vt in V_TYPES:
                    vehs_type = vehs_per_route * vt["share"]
                    if vehs_type <= 0:
                        continue
                    flows.append({
                        "id": f"{entry_side}_{target_side}_{flow_id_counter}_{vt['id']}",
                        "type": vt["id"],
                        "from": orig,
                        "to": dest,
                        "vehsPerHour": vehs_type,
                    })
                    flow_id_counter += 1
    return flows


def write_flows(flows, outpath):
    root = ET.Element("routes")
    write_vtypes(root)
    for f in flows:
        ET.SubElement(root, "flow", {
            "id": f["id"],
            "type": f["type"],
            "from": f["from"],
            "to": f["to"],
            "vehsPerHour": f"{f['vehsPerHour']:.6f}",
            "begin": "0",
            "end": "2600",
        })
    ET.indent(root, space="    ")
    ET.ElementTree(root).write(outpath, encoding="utf-8", xml_declaration=True)


def write_catalog_summary(catalog, entry_edges, exit_edges, out_dir, filename):
    summary_path = os.path.join(out_dir, filename)
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entrada", "saida", "num_pares_od_validos"])
        for entry_side in SIDES:
            for target_side in SIDES:
                if entry_side == target_side:
                    continue
                writer.writerow([entry_side, target_side, len(catalog[entry_side][target_side])])
    print(f"Resumo salvo em: {summary_path}")


def print_catalog(catalog, entry_edges, exit_edges):
    print("\nArestas de entrada detectadas:")
    for side in SIDES:
        print(f"  {side}: {entry_edges[side]}")
    print("\nArestas de saida detectadas:")
    for side in SIDES:
        print(f"  {side}: {exit_edges[side]}")
    print("\nPares OD validos por movimento:")
    for entry_side in SIDES:
        for target_side in SIDES:
            if entry_side == target_side:
                continue
            print(f"  {entry_side}->{target_side}: {len(catalog[entry_side][target_side])}")


def generate_all(net_file, out_dir, include_zero_level=True, allow_same_side=False, dry_run=False):
    os.makedirs(out_dir, exist_ok=True)

    edges, adj, rows, cols = parse_net(net_file)
    print(f"Rede: {net_file}")
    print(f"Linhas detectadas: {rows}")
    print(f"Colunas detectadas: {cols}")
    print(f"Arestas externas validas: {len(edges)}")

    raw_catalog, entry_edges, exit_edges = build_raw_catalog(
        edges=edges,
        adj=adj,
        rows=rows,
        cols=cols,
        allow_same_side=allow_same_side,
    )

    print("\nCatalogo antes da validacao por DUAROUTER:")
    print_catalog(raw_catalog, entry_edges, exit_edges)
    write_catalog_summary(raw_catalog, entry_edges, exit_edges, out_dir, "resumo_catalogo_rotas_bruto.csv")

    catalog = validate_catalog_with_duarouter(net_file, raw_catalog, out_dir)

    print("Catalogo depois da validacao por DUAROUTER:")
    print_catalog(catalog, entry_edges, exit_edges)
    write_catalog_summary(catalog, entry_edges, exit_edges, out_dir, "resumo_catalogo_rotas_validado.csv")

    levels_to_test = [0, 1, 2, 3] if include_zero_level else [1, 2, 3]
    total_created = 0
    total_failed = 0

    for comb_idx, comb in enumerate(itertools.product(levels_to_test, repeat=len(SIDES))):
        levels = dict(zip(SIDES, comb))
        if all(level == 0 for level in levels.values()):
            continue

        name = "_".join(f"{side}{level}" for side, level in levels.items())
        flows = build_flows(levels, catalog)
        if not flows:
            print(f"[IGNORADO] {name}: nenhum flow gerado")
            continue

        flow_file = os.path.join(out_dir, f"flows_{name}.xml")
        rou_file = os.path.join(out_dir, f"routes_{name}.rou.xml")
        write_flows(flows, flow_file)

        if dry_run:
            print(f"[DRY-RUN] {flow_file}: {len(flows)} flows")
            total_created += 1
            continue

        res = run_duarouter(net_file, flow_file, rou_file, seed=42 + comb_idx, ignore_errors=False)
        if res.returncode == 0:
            print(f"[OK] criado {rou_file}")
            total_created += 1
        else:
            print(f"[ERRO] {rou_file}")
            print(res.stderr.strip())
            total_failed += 1

    print("\nFim da geracao.")
    print(f"Cenarios criados: {total_created}")
    print(f"Cenarios com erro: {total_failed}")
    print(f"Diretorio de saida: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Gerador automatico de rotas para grid MARL 4x3/3x3 usando DUAROUTER."
    )
    parser.add_argument("--net", type=str, default=DEFAULT_NET_FILE,
                        help="Arquivo .net.xml do SUMO. Default: baseSumo/grid.net.xml")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                        help="Diretorio de saida das rotas. Default: rotas_jtr_marl_4x3")
    parser.add_argument("--sem-zero", action="store_true",
                        help="Usa apenas niveis 1,2,3. Sem isso, usa 0,1,2,3 e exclui o cenario todo zero.")
    parser.add_argument("--permitir-mesmo-lado", action="store_true",
                        help="Tambem gera pares que entram e saem pelo mesmo lado. Normalmente nao recomendo.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Gera apenas arquivos de flow e resumos; nao gera .rou.xml final.")
    args = parser.parse_args()

    generate_all(
        net_file=args.net,
        out_dir=args.out_dir,
        include_zero_level=not args.sem_zero,
        allow_same_side=args.permitir_mesmo_lado,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

