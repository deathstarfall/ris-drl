"""Generates the three frozen datasets. Run ONCE with the D2 command; the npz files are then
   committed to the repo and NEVER regenerated (DD-006)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.channel import save_dataset, _cfg

d = _cfg()["data"]
if __name__ == "__main__":
    print("train:", save_dataset("data/train_channels.npz", d["train_n"], d["train_seed"]))
    print("test :", save_dataset("data/test_channels.npz",  d["test_n"],  d["test_seed"]))
    gen = _cfg()["geometry"]["gen_user_zone"]                       # unseen positions zone
    print("gen  :", save_dataset("data/gen_channels.npz",   d["gen_n"],   d["gen_seed"], zone=gen))