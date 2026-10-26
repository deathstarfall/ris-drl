"""Channel generation & storage. Ayansh is the ONLY writer of data/*.npz."""
import numpy as np, yaml, pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]

def _cfg():
    return yaml.safe_load(open(_ROOT / "configs" / "env.yaml", encoding="utf-8"))

def pl_direct_db(d_m):
    """BS -> user, NLoS UMi: 128.1 + 37.6 log10(d_km)"""
    return 128.1 + 37.6 * np.log10(max(d_m, 1.0) / 1e3)

def pl_cascade_db(d_bt_m, d_tu_m):
    """Two-hop BS->RIS->user cascade path loss (Wu-Zhang form), d in meters."""
    return 61.4 + 35.8 * np.log10(max(max(d_bt_m, d_tu_m), 1.0))

def noise_sigma2(p_cfg):
    """linear noise power (W) from  -174 dBm/Hz + 10log10(BW) + NF"""
    bw_hz = p_cfg["bandwidth_mhz"] * 1e6
    return 10 ** ((-174 + 10 * np.log10(bw_hz) + p_cfg["noise_figure_db"] - 30) / 10)

def gen_channel(rng, cfg, zone=None):
    g = cfg["geometry"]
    z = zone or g["user_zone"]
    u = np.array([rng.uniform(z["x1"], z["x2"]), rng.uniform(z["y1"], z["y2"]), z["z"]])
    bs, ris = np.array(g["bs_pos"]), np.array(g["ris_pos"])
    d_bt = float(np.linalg.norm(ris - bs))
    d_tu = float(np.linalg.norm(u - ris))
    d_bu = float(np.linalg.norm(u - bs))

    # ---- direct link: UMi NLoS formula + behind-the-wall blockage (DD-011b) ----
    blockage_db = cfg["channel"].get("direct_blockage_db", 0.0)
    g_direct = 10 ** (-(pl_direct_db(d_bu) + blockage_db) / 10)

    # ---- cascade link: two-hop Wu-Zhang path loss ----
    # DD-042 channel-model disclosure: the two hops share ONE product attenuation.
    # pl_cascade_db() already covers the WORSE of the two hops' distances (Wu-Zhang
    # form for the larger of d_bt/d_tu), so a separate per-hop scaling of G would
    # double-count the shared loss. G consequently carries unit magnitude (i.i.d.
    # CN(0,1)) and the full cascade attenuation profiles the RIS -> user hop h_r.
    # carrier_ghz in env.yaml is documentation-only: the 3.5 GHz mid-band point is
    # fixed by these published UMi coefficients; mmWave/sub-THz propagation is NOT
    # modelled by this generator (v1.1 scope).
    g_casc = 10 ** (-pl_cascade_db(d_bt, d_tu) / 10)

    M, N, K = cfg["dims"]["M"], cfg["dims"]["N"], cfg["dims"]["K"]
    cn = lambda s: (rng.normal(size=s) + 1j * rng.normal(size=s)) / np.sqrt(2)

    ch = dict(
        u_pos=u,
        h_d=np.sqrt(g_direct) * cn((K, M)),
        G=cn((M, N)),                              # BS -> RIS (unprofiled)
        h_r=np.sqrt(g_casc) * cn((K, N)),          # RIS -> user k, per-element profile
        sigma2=noise_sigma2(cfg["power"]),
        P=10 ** ((cfg["power"]["p_tx_dbm"] - 30.0) / 10.0),   # 30 dBm -> 1.0 W
    )
    return ch

def make_dataset(n, seed, zone=None):
    cfg = _cfg()
    rng = np.random.default_rng(seed)
    return [gen_channel(rng, cfg, zone) for _ in range(n)]

def save_dataset(path, n, seed, zone=None):
    chs = make_dataset(n, seed, zone)
    keys = chs[0].keys()
    np.savez_compressed(_ROOT / path,
        u_pos  = np.stack([c["u_pos"]  for c in chs]),
        h_d    = np.stack([c["h_d"]    for c in chs]),
        G      = np.stack([c["G"]      for c in chs]),
        h_r    = np.stack([c["h_r"]    for c in chs]),
        sigma2 = np.array([c["sigma2"] for c in chs]),
        P      = np.array([c["P"]      for c in chs]),
        seed=seed, keys="|".join(keys))
    return len(chs)

def load_dataset(path):
    d = np.load(_ROOT / path, allow_pickle=True)
    keys = str(d["keys"]).split("|")
    
    # FIX: Load arrays into memory FIRST to avoid millions of slow zip decompressions
    arrays = {k: d[k] for k in keys}
    
    return [dict(zip(keys, [arrays[k][i] for k in keys])) for i in range(len(arrays["sigma2"]))]