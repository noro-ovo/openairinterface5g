#!/usr/bin/env python3
"""Generate the base-graph tables of 3GPP TS 38.212 Sec. 5.3.2 for the dec_std decoder.

Primary source : the spec itself: spec/bg1_Vij.csv, spec/bg2_Vij.csv (Tables 5.3.2-2/-3 extracted from the official
                 .docx) and Table 5.3.2-1 (spec/38212-j40/tables/table_5.3.2-1.csv).
Cross-check 0  : the raw per-release extracts 38212-{fd0,gf0,hd0,i80,j40}/tables/table_5.3.2-{1,2,3}.csv
                 (must be byte-identical across releases and equal to the flattened CSVs).
Cross-check 1  : srsRAN_Project (raw V_ij per set index iLS, lib/phy/upper/channel_coding/ldpc/ldpc_luts_impl.cpp)
Cross-check 2  : OAI 9313cbea decoder LUTs (nrLDPC_lut.h: circShift_BG*_Z*_CNG* = V_ij mod Z per check-node
                 group, posBnInCnProcBuf_BG*_CNG* = column of every edge), for every Z of Table 5.3.2-1.

Output: tables/ldpc_tables.c with, per base graph, the edge list in row-major order (row, column) and
the raw V_ij for the 8 set indices, plus the lifting-size table.

usage: gen_tables.py SPEC_DIR SRSRAN_LUTS_CPP OAI_NRLDPC_LUT_H OUT_C
"""
import csv, os, re, sys

def die(msg):
    print("ERROR:", msg); sys.exit(1)

def parse_spec_bg(path):
    """returns dict (row, col) -> [V for iLS 0..7]"""
    out = {}
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    for r in rows[3:]:
        if not r or not r[0].strip().isdigit():
            continue
        for half in (r[0:10], r[10:20]):
            if len(half) < 10 or not half[0].strip():
                continue
            i, j = int(half[0]), int(half[1])
            v = [int(x) for x in half[2:10]]
            if (i, j) in out:
                die(f"duplicate entry {i},{j} in {path}")
            out[(i, j)] = v
    return out

def parse_spec_ls(path):
    sets = {}
    with open(path, newline='') as f:
        for r in csv.reader(f):
            if r and r[0].strip().isdigit():
                sets[int(r[0])] = [int(x) for x in re.findall(r"\d+", r[1])]
    return sets

def parse_srsran(path):
    txt = open(path).read()
    res = {}
    for bg, nrow, ncol, name in ((1, 46, 68, "BG1_matrices"), (2, 42, 52, "BG2_matrices")):
        start = txt.index(f"std::array<BG_matrix_t, NOF_LIFTING_INDICES> {name}")
        body = txt[txt.index("{", start):]
        # cut at the end of this initializer: the next ';'
        body = body[:body.index(";")]
        body = re.sub(r"//[^\n]*", "", body)
        toks = re.findall(r"NO_EDGE|\d+|\{|\}", body)
        # structure: { { {{row}, {row}, ...} } x8 } ; collect rows as innermost brace groups
        rows, depth, cur = [], 0, None
        for t in toks:
            if t == "{":
                depth += 1; cur = []
            elif t == "}":
                if cur is not None:
                    rows.append(cur)
                cur = None; depth -= 1
            else:
                cur.append(None if t == "NO_EDGE" else int(t))
        # rows has the innermost groups only (row vectors); each matrix has MAX_BG_M=46 rows of MAX_BG_N_FULL=68
        rpm = len(rows) // 8  # rows stored per matrix (46 for BG1; BG2 is stored with 42 or padded to 46)
        if len(rows) % 8 or rpm < nrow:
            die(f"srsRAN {name}: got {len(rows)} rows")
        mats = [rows[k * rpm:(k + 1) * rpm] for k in range(8)]
        d = {}
        for ils in range(8):
            for i in range(nrow):
                row = mats[ils][i]
                for j in range(ncol):
                    v = row[j] if j < len(row) else None
                    if v is not None:
                        d.setdefault((i, j), [None] * 8)[ils] = v
            # padding rows/cols must be empty
            for i in range(nrow, rpm):
                if any(x is not None for x in mats[ils][i]):
                    die(f"srsRAN {name} iLS{ils} row {i} not empty")
        res[bg] = d
    return res

def parse_oai(path):
    txt = open(path).read()
    arr = {}
    for m in re.finditer(r"static const uint(?:8|16)_t (\w+)\[(\d+)\]\[(\d+)\]\s*=\s*\{(.*?)\};", txt, re.S):
        name, a, b, body = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        nums = [int(x) for x in re.findall(r"\d+", body)]
        if len(nums) != a * b:
            die(f"OAI {name}: size")
        arr[name] = [nums[k * b:(k + 1) * b] for k in range(a)]
    return arr

def main():
    spec_dir, srs_path, oai_path, out_path = sys.argv[1:5]
    versions = sorted(d for d in os.listdir(spec_dir)
                      if d.startswith("38212-") and os.path.isdir(os.path.join(spec_dir, d, "tables")))
    primary = "38212-j40" if "38212-j40" in versions else versions[-1]
    tdir = lambda v: os.path.join(spec_dir, v, "tables")
    for v in versions:  # all extracted releases must carry identical tables
        for t in ("1", "2", "3"):
            a = open(os.path.join(tdir(v), f"table_5.3.2-{t}.csv"), "rb").read()
            b = open(os.path.join(tdir(primary), f"table_5.3.2-{t}.csv"), "rb").read()
            if a != b:
                die(f"{v} table 5.3.2-{t} differs from {primary}")
    ls = parse_spec_ls(os.path.join(tdir(primary), "table_5.3.2-1.csv"))
    spec = {}
    for bg in (1, 2):  # primary: flattened V_ij tables shipped with the spec
        flat = {}
        with open(os.path.join(spec_dir, f"bg{bg}_Vij.csv"), newline='') as f:
            for r in csv.DictReader(f):
                if int(r["bg"]) != bg:
                    die("bg column")
                key = (int(r["row"]), int(r["col"]))
                if key in flat:
                    die(f"duplicate {key}")
                flat[key] = [int(r[f"V{k}"]) for k in range(8)]
        spec[bg] = flat
    raw = {1: parse_spec_bg(os.path.join(tdir(primary), "table_5.3.2-2.csv")),
           2: parse_spec_bg(os.path.join(tdir(primary), "table_5.3.2-3.csv"))}
    zs = sorted(z for s in ls.values() for z in s)
    if len(zs) != 51 or len(set(zs)) != 51:
        die("lifting sizes")
    ils_of = {z: i for i, s in ls.items() for z in s}
    dims = {1: (46, 68, 316), 2: (42, 52, 197)}
    for bg in (1, 2):
        nr, nc, ne = dims[bg]
        if len(spec[bg]) != ne:
            die(f"BG{bg}: {len(spec[bg])} edges, expected {ne}")
        if any(i >= nr or j >= nc for (i, j) in spec[bg]):
            die("index range")
    report = [f"spec CSV extracts {', '.join(versions)} identical"]
    for bg in (1, 2):
        if raw[bg] != spec[bg]:
            die(f"raw table 5.3.2-{bg + 1} extract differs from bg{bg}_Vij.csv")
        report.append(f"raw Table 5.3.2-{bg + 1} extract = bg{bg}_Vij.csv")
    # ---- cross-check 1: srsRAN raw V
    srs = parse_srsran(srs_path)
    for bg in (1, 2):
        if set(srs[bg]) != set(spec[bg]):
            die(f"srsRAN BG{bg}: edge set differs: {set(srs[bg]) ^ set(spec[bg])}")
        bad = [(e, spec[bg][e], srs[bg][e]) for e in spec[bg] if srs[bg][e] != spec[bg][e]]
        if bad:
            die(f"srsRAN BG{bg}: V differs: {bad[:5]}")
        report.append(f"srsRAN BG{bg}: {len(spec[bg])} edges x 8 iLS identical to spec")
    # ---- cross-check 2: OAI decoder LUTs, V mod Z for every Z
    oai = parse_oai(oai_path)
    groups = {1: [3, 4, 5, 6, 7, 8, 9, 10, 19], 2: [3, 4, 5, 6, 8, 10]}
    for bg in (1, 2):
        nr = dims[bg][0]
        rows = {i: sorted(j for (r, j) in spec[bg] if r == i) for i in range(nr)}
        nchk = 0
        for z in zs:
            ils = ils_of[z]
            seen = set()
            for g in groups[bg]:
                pos = oai[f"posBnInCnProcBuf_BG{bg}_CNG{g}"]
                sh = oai[f"circShift_BG{bg}_Z{z}_CNG{g}"]
                ncn = len(pos[0])
                for c in range(ncn):
                    cols = [pos[k][c] for k in range(g)]
                    cands = [i for i in range(nr) if rows[i] == sorted(cols) and i not in seen]
                    if len(cands) != 1:
                        die(f"OAI BG{bg} CNG{g} cn{c}: cannot identify row ({cands})")
                    i = cands[0]; seen.add(i)
                    for k in range(g):
                        if sh[k][c] != spec[bg][(i, cols[k])][ils] % z:
                            die(f"OAI BG{bg} Z{z} row {i} col {cols[k]}: OAI {sh[k][c]} spec {spec[bg][(i, cols[k])][ils]} mod {z}")
                        nchk += 1
            if len(seen) != nr:
                die(f"OAI BG{bg} Z{z}: {len(seen)} rows")
        report.append(f"OAI BG{bg}: {nchk} (edge, Z) shift values = V mod Z identical to spec (all 51 Z)")
    # ---- emit C
    with open(out_path, "w") as f:
        f.write("/* GENERATED by tables/gen_tables.py from 3GPP TS 38.212 (spec/bg1_Vij.csv, spec/bg2_Vij.csv, Table 5.3.2-1 of %s);\n" % primary)
        f.write(" * cross-checked: %s. Do not edit. */\n" % "; ".join(report))
        f.write('#include "ldpc_tables.h"\n\n')
        f.write("const uint16_t ldpc_lifting_sizes[51] = {%s};\n" % ", ".join(map(str, zs)))
        f.write("const uint8_t ldpc_ils_of_z[385] = {")
        f.write(", ".join(str(ils_of.get(z, 255)) for z in range(385)))
        f.write("};\n\n")
        for bg in (1, 2):
            edges = sorted(spec[bg])
            f.write(f"/* BG{bg}: {len(edges)} edges, row-major; V_ij for iLS 0..7 */\n")
            f.write(f"const ldpc_bg_edge_t ldpc_bg{bg}_edges[{len(edges)}] = {{\n")
            for (i, j) in edges:
                f.write("  {%d, %d, {%s}},\n" % (i, j, ", ".join(map(str, spec[bg][(i, j)]))))
            f.write("};\n\n")
    for r in report:
        print(r)
    print("wrote", out_path)

if __name__ == "__main__":
    main()
