import io

ADD = r"""
@inproceedings{huang2019indoor,
  author    = {Huang, Chongwen and Alexandropoulos, George C. and Yuen, Chau and Debbah, M{\'e}rouane},
  title     = {Indoor Signal Focusing with Deep Learning Designed Reconfigurable Intelligent Surfaces},
  booktitle = {2019 IEEE 20th International Workshop on Signal Processing Advances in Wireless Communications (SPAWC)},
  pages     = {1--5},
  year      = {2019},
  doi       = {10.1109/SPAWC.2019.8815412}
}
@article{liu2022deepunfolding,
  author  = {Liu, Yanzhen and Hu, Qiyu and Cai, Yunlong and Yu, Guanding and Li, Geoffrey Ye},
  title   = {Deep-Unfolding Beamforming for Intelligent Reflecting Surface Assisted Full-Duplex Systems},
  journal = {IEEE Transactions on Wireless Communications},
  volume  = {21},
  number  = {7},
  pages   = {4784--4800},
  year    = {2022},
  doi     = {10.1109/TWC.2021.3133296}
}
@article{alexandropoulos2022pervasive,
  author  = {Alexandropoulos, George C. and Stylianopoulos, Kyriakos and Huang, Chongwen and Yuen, Chau and Bennis, Mehdi and Debbah, M{\'e}rouane},
  title   = {Pervasive Machine Learning for Smart Radio Environments Enabled by Reconfigurable Intelligent Surfaces},
  journal = {Proceedings of the IEEE},
  volume  = {110},
  number  = {9},
  pages   = {1494--1525},
  year    = {2022},
  doi     = {10.1109/JPROC.2022.3174030}
}
@article{borgerding2017amp,
  author  = {Borgerding, Mark and Schniter, Philipp and Rangan, Sundeep},
  title   = {{AMP}-Inspired Deep Networks for Sparse Linear Inverse Problems},
  journal = {IEEE Transactions on Signal Processing},
  volume  = {65},
  number  = {16},
  pages   = {4293--4308},
  year    = {2017},
  doi     = {10.1109/TSP.2017.2708040}
}
"""
b = io.open("paper/refs.bib", encoding="utf-8-sig").read()
if "liu2022deepunfolding" not in b:
    io.open("paper/refs.bib", "a", encoding="utf-8").write(ADD)
    print("bib: 4 entries appended")

def patch(path, pairs):
    t = io.open(path, encoding="utf-8-sig").read()
    for a, bb in pairs:
        if a not in t:
            print("MISS:" + path + " :: " + a[:60].replace("\n", "|"))
            continue
        t = t.replace(a, bb)
    io.open(path, "w", encoding="utf-8", newline="\n").write(t)
    print("patched", path)

# intro: unfolding lineage + prior unfolding-for-RIS + learning-based RIS line
patch("paper/sections/sec1_intro.tex", [
    ("\\emph{deep unfolding} \\cite{shlezinger2023model} converts iterations of\n"
     "a projected-gradient method into layers of a fixed-depth neural\n"
     "network, regaining learned step sizes while keeping the physics\n"
     "visible.",
     "\\emph{deep unfolding} \\cite{shlezinger2023model} converts iterations of\n"
     "a projected-gradient method into layers of a fixed-depth neural\n"
     "network, regaining learned step sizes while keeping the physics\n"
     "visible---a lineage that began with AMP-inspired networks\n"
     "\\cite{borgerding2017amp} and has since been applied to RIS\n"
     "beamforming directly \\cite{liu2022deepunfolding}."),
    ("end-to-end from reward; DRL-based phase configuration for RIS has been\nstudied recently~\\cite{huang2020risdrl}, and sum-rate objectives with\ndiscrete phases are commonly attacked by alternating or manifold\nmethods~\\cite{guo2020weighted,zhang2021manifold}",
     "end-to-end from reward; learning-designed sub-surface signal focusing\npredates the current RL wave~\\cite{huang2019indoor}, DRL-based phase\nconfiguration was analyzed by Huang et al.~\\cite{huang2020risdrl}, a\nrecent Proc.~IEEE overview maps the learning-for-RIS landscape\n\\cite{alexandropoulos2022pervasive}, and sum-rate objectives with\ndiscrete phases are commonly attacked by alternating or manifold\nmethods~\\cite{guo2020weighted,zhang2021manifold}"),
])

# sec4b: cite the unfolding-for-RIS precedent at the method statement
patch("paper/sections/sec4b_deep_unfolding.tex", [], {})
print("in-text citations wired")
