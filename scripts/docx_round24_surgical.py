# Round-24 (DD-065): surgical bug-fix pass on the TEAM'S HUMANIZED docx.
# Hard rules:
#  - replace ONLY buggy sentences/paragraphs; all other text untouched
#  - never collapse runs on a paragraph containing an inline image
#  - python-docx API only; Word-COM verify afterwards
import docx
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

LIVE = r'C:\Users\user\OneDrive\Desktop\All Assignments\CN project\TeamNo_6_24BDS0166_RIS_6G_Report.docx'
FIGS = r'C:\code\ris-drl\paper\figs'
d = docx.Document(LIVE)
ps = d.paragraphs

def by_idx(i):
    return ps[i]

def replace_text(p, old, new):
    """Guarded replacement: no images inside, old must be present."""
    assert len(p._p.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip')) == 0, \
        'attempted text-collapse on IMAGE paragraph'
    full = p.text
    assert old in full, 'MISSING: ' + old[:70]
    first = p.runs[0]
    for r in list(p.runs)[1:]:
        r._r.getparent().remove(r._r)
    first.text = full.replace(old, new)

def append_sentence(p, sent):
    assert len(p._p.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip')) == 0
    p.runs[-1].text = p.runs[-1].text + sent

# ---- fix 4: quantization numbers (para 158) -----------------------------------
replace_text(by_idx(158),
    "The quantization losses are 2.936 / 0.730 / 0.182 bps/Hz for AO at B = 2/3/4 "
    "against the unquantized 64.193: monotone degradation in which a 2 bit "
    "codebook already retains over 70% of the AO gain over random. Deep "
    "unfolding's own losses are 0.965 / 0.233 / 0.060 bps/Hz against 62.532 "
    "smaller than AO's at every B. (Table 11)",
    "The quantization losses are 0.976 / 0.243 / 0.061 bps/Hz for AO at B = 2/3/4 "
    "against the unquantized 64.193: monotone degradation in which a 2 bit "
    "codebook already retains over 90% of the AO gain over random (91.1%, "
    "computed against the 11.02 bps/Hz AO-over-random gain). Deep "
    "unfolding's own losses are 0.965 / 0.233 / 0.060 bps/Hz against 62.532 - "
    "essentially the same relative quantization loss as AO (1.54% vs 1.52% of "
    "the respective unquantized rates at B = 2). (Table 11)")
print('fix4 ok')

# ---- fix 3 wording: unfolding rename (para 174) -------------------------------
replace_text(by_idx(174),
    'Deep unfolding is the operating principle of this result:',
    'Deep unfolding - an L = 6 unrolled projected-gradient network whose only '
    'trainable parameters are six learned step sizes (6 parameters in code) - '
    'is the operating principle of this result:')
print('fix3 rename ok')

# ---- latency ratio variance + equal-compute + FLOPs (paras 26, 174, 187) ------
replace_text(by_idx(174),
    'at a same machine latency ratio of roughly 51 times (AO 0.424 s vs. '
    'unfolding 8.26 ms per decision, serialized single solve wall clocks on '
    'one process running on one idle desktop CPU).',
    'at a same machine latency ratio of roughly 40-50x (frozen protocol run: '
    'AO 0.424 s vs. unfolding 8.26 ms per decision, ratio 51.4x; an independent '
    're-measure of the same protocol gave AO 0.340 s vs. 8.39 ms, ratio 40.5x, '
    'so the ratio carries roughly +/-25% session variance; serialized single '
    'solve wall clocks on one process running on one idle desktop CPU).')
add_eq = (" A compute-matched counterpoint (Table 13) bounds how much of the "
          "gap is architecture versus budget: at six iterations for every "
          "method, truncated AO reaches 61.58 +/- 0.75 and a grid-searched "
          "constant-step projected gradient (step 3.0, selected on a "
          "300-channel validation holdout) reaches 62.41 +/- 0.74, while the "
          "learned step sizes reach 62.53 +/- 0.85 - the trained network is "
          "the best six-iteration method, though the margin over a tuned "
          "constant step is only about +0.12 bps/Hz, so the headline ratio is "
          "dominated by the iteration budget, not the architecture. In "
          "floating-point terms the gap is about 48x: one AO decision is "
          "roughly 1.37 million complex multiply-accumulate operations (193 "
          "iterations x ~7.1 k each) against roughly 28.5 thousand for the "
          "network (6 layers x ~4.7 k each). Gradient computation is "
          "intrinsic to the network: each layer's update is itself a "
          "gradient step, computed with the same torch autograd engine the "
          "AO reference uses, so the per-decision latency legitimately "
          "includes one autograd pass per layer.")
if 'compute-matched counterpoint' not in by_idx(174).text:
    append_sentence(by_idx(174), add_eq)
print('latency/equal-compute ok')

# ratio numbers in abstract-bullet and conclusion
p26 = by_idx(26)
if 'only 51 time' in p26.text:
    i = p26.text.find('with only 51 time')
    frag = p26.text[i:i+60]
    replace_text(p26, frag.split(' with')[0], '')  # no-op guard; real replace below
# safer: locate exact phrasing
t26 = p26.text
import re as _re
m = _re.search(r'with only 51 times?[^.]*\.', t26)
if m:
    replace_text(p26, m.group(0), 'with roughly 40-50 times lower decision '
                 'latency under the same measurement protocol.')
replace_text(by_idx(187),
    'with 51 times lower decision latency (0.424 s vs 8.26 ms)',
    'with roughly 40-50 times lower decision latency (0.424 s/8.26 ms frozen '
    'protocol; 0.340 s/8.39 ms re-measure)')
print('ratio consistency ok')

# ---- fix 7: factored DQN limitation (para 178) --------------------------------
append_sentence(by_idx(178),
    ' Sixth is that the discrete DQN arms use a factored head that scores each '
    'RIS element codebook entry independently rather than one joint action '
    'over the codebook; a joint-action DQN over a small codebook is the fair '
    'discrete comparator and remains future work, so the discrete-arm null is '
    'bounded by this design choice.')
print('fix7 ok')

# ---- fix 8: Huang-Mo-Yuen representation note (para 59) -----------------------
append_sentence(by_idx(59),
    ' Note that the JSAC 2020 study obtained its positive outcome with a '
    'different input/output representation and reward scale from ours '
    '(state-augmented real/imaginary observations with cosine/sine phase '
    'parameterization versus our raw complex observation with raw-phase or '
    'factored-codebook outputs); representation-matched DRL controls are '
    'therefore flagged as required future work.')
print('fix8 ok')

# ---- jargon strips ------------------------------------------------------------
replace_text(by_idx(114),
    'This is an explicit simplification in the project decision log which '
    'implies that the generator does not consider',
    'This is an explicitly documented simplification: the generator does not '
    'consider')
replace_text(by_idx(116),
    'are all fixed and versioned in the released decision log.',
    'are all fixed and versioned in the released code repository.')
replace_text(by_idx(133),
    'bootstrap intervals at 95% are saved in '
    'results/baselines/zf_rescore/certified_table_det.json.',
    'bootstrap intervals at 95% are archived per architecture in the released '
    'artifact set.')
replace_text(by_idx(134),
    'selected based on validation rate alone (DD 041 protocol; selection '
    'artifact results/unfold/val_protocol_selection.json).',
    'selected based on validation rate alone.')
replace_text(by_idx(134),
    'Hyperparameters per run configurations and the full decision log are '
    'committed in configs/ and docs/.',
    'Hyperparameters and per-run configurations are committed in the released '
    'code repository.')
replace_text(by_idx(144),
    'Acceptance Criteria Disclosure (Auditor Round 3):',
    'Acceptance Criteria Disclosure:')
replace_text(by_idx(144),
    'the retired cross machine pair (1.295 ms / 1.352 s) is revealed',
    'an earlier cross machine pair (1.295 ms / 1.352 s) is disclosed for '
    'completeness')
replace_text(by_idx(144),
    'which was noted in the project decision log.',
    'which was documented in the released record.')
replace_text(by_idx(145),
    'and the retired cross machine pair 1.295 ms / 1.352 s is disclosed in '
    'section 5)',
    'and an earlier cross machine pair 1.295 ms / 1.352 s is disclosed in '
    'section 5)')
replace_text(by_idx(174),
    'It was a cross machine ratio and is now retired from the spotlight. It '
    'is disclosed here.',
    'It is a cross machine ratio, is not used in any headline claim, and is '
    'disclosed for completeness.')
replace_text(by_idx(201),
    '(all reported numbers derive from the released raw JSON artifacts in '
    'results/ and the full decision log is provided in docs/design_decisions.md)',
    '(all reported numbers derive from the released raw result artifacts, '
    'with the full per-run configuration record provided alongside the code)')
print('jargon strips ok')

# ---- fix 5 wording in para 151 (v1.1 only; Figure 2 ref stays valid until
#      the new figures force renumbering, handled below) ------------------------
replace_text(by_idx(151),
    '(a robustness sweep using the trained DRL checkpoint is left to the '
    'v1.1 run). (Table 10; Figure 2)',
    '(including the trained DRL checkpoints in the robustness sweep is left '
    'as future work). (Table 10; Figure 2)')
print('fix5 wording ok (pre-renumber)')

d.save(LIVE)
print('PHASE 1 SAVED')
