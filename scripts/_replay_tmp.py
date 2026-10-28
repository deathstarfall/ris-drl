# Round-20 (DD-062): auditor must-fix lane C (non-computing edits), all
# artifact-verified before wording:
#  - Fix 4: sec 4.2 quantization losses corrected from stale 2.936/0.730/0.182
#    (those are the paper's old unexplained "dB" column) to the bits_sweep.json
#    truth 0.976/0.243/0.061 bps/Hz; retained share 70% -> 91%.
#  - Fix 3 (wording): unfolding renamed "L = 6 unrolled projected-gradient
#    network whose only trainable parameters are six learned step sizes"
#    (verified: UnrolledPG(L=6).parameters == 6 alphas).
#  - Fix 7: factored DQN head limitation appended.
#  - Fix 8: Huang-Mo-Yuen representation sentence in discussion.
#  - Fix 6: training-curve composite + AO convergence figures inserted in 4.1;
#    CSI figure renumbered 2 -> 4 (captions only, no in-text numeric refs).
# Hard rules (DD-059): python-docx API only, schema-safe, Word-COM verify after.
import docx
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import itertools

DOC = r'C:\Users\user\AppData\Local\Temp\opencode\replay_test.docx'
FIGS = r'C:\code\ris-drl\paper\figs'

d = docx.Document(DOC)
paras = d.paragraphs

def replace_in_para(p, old, new):
    """Replace text in a paragraph run-safely (only on text runs, no p.text)."""
    full = p.text
    assert old in full, 'missing: ' + old[:60]
    # single-shot: put whole new text in first run, clear the rest
    done = False
    for r in p.runs:
        if not done:
            r.text = new if False else r.text  # placeholder no-op
            done = True
    # safer generic: rebuild as one run preserving first-run formatting
    first = p.runs[0]
    for r in list(p.runs)[1:]:
        r._r.getparent().remove(r._r)
    first.text = full.replace(old, new)

# --- Fix 4: paragraph 158 quantization numbers --------------------------------
p158 = paras[158]
replace_in_para(
    p158,
    'The quantization losses are 2.936 / 0.730 / 0.182 bps/Hz for AO at B = 2/3/4 against the unquantized 64.193: monotone degradation in which a 2 bit codebook already retains over 70% of the AO gain over random. Deep unfolding\u2019s own losses are 0.965 / 0.233 / 0.060 bps/Hz against 62.532 smaller than AO\u2019s at every B. (Table 11)'.replace('\u2019', "'"),
    'The quantization losses are 0.976 / 0.243 / 0.061 bps/Hz for AO at B = 2/3/4 against the unquantized 64.193: monotone degradation in which a 2-bit codebook already retains over 90% of the AO gain over random (91.1%). Deep unfolding\u2019s own losses are 0.965 / 0.233 / 0.060 bps/Hz against 62.532 - smaller than AO\u2019s at every B, though only by a hair (0.011/0.010/0.001 bps/Hz). (Table 11)'.replace('\u2019', "'"))
print('fix4 done')

# --- Fix 3 wording: paragraph 174 rename --------------------------------------
p174 = paras[174]
replace_in_para(
    p174,
    'Deep unfolding is the operating principle of this result:',
    'Deep unfolding - an L = 6 unrolled projected-gradient network whose only trainable parameters are six learned step sizes (verified: 6 parameters in code) - is the operating principle of this result:')
print('fix3 wording done')

# --- Fix 7: factored-DQN limitation -------------------------------------------
p178 = paras[178]
add = (' Sixth is that the discrete DQN arms use a factored head that scores each RIS '
       'element\u2019s codebook entry independently instead of one joint action over the '
       'codebook; a joint-action DQN over a small codebook is the fair discrete '
       'comparator and is identified as required future work, so the discrete-arm null '
       'is bounded by this design choice.'.replace('\u2019', "'"))
if 'factored head' not in p178.text:
    last = p178.runs[-1]
    last.text = last.text + add
print('fix7 done')

# --- Fix 8: Huang-Mo-Yuen representation sentence in discussion ---------------
target = None
for p in paras:
    if 'Huang, Mo and Yuen' in p.text and 'JSAC 2020' in p.text:
        target = p
        break
sent = (' Their positive outcome was obtained with a different input/output '
        'representation and reward scale from ours (real/imaginary observations with '
        'cosine/sine phase parameterization versus our raw complex observation and raw '
        'phase or factored codebook outputs), which is precisely the representation '
        'question our positive-control probe addresses; representation-matched DRL '
        'controls are flagged as required future work.')
if 'representation question' not in target.text:
    target.runs[-1].text = target.runs[-1].text + sent
print('fix8 done')

# --- Fix 6: composite training-curve image, insert Figure 2 + Figure 3 ---------
from PIL import Image
imgs = [Image.open(rf'{FIGS}\\fig_band_{f}.png') for f in
        ('sac', 'td3', 'ddpg', 'dqn_b2', 'dqn_b3', 'dqn_b4')]
w = min(im.width for im in imgs)
imgs = [im.resize((w, int(im.height * w / im.width))) for im in imgs]
rows, cols = 3, 2
cell_h = max(im.height for im in imgs)
canvas = Image.new('RGB', (cols * w, rows * cell_h), 'white')
for k, im in enumerate(imgs):
    canvas.paste(im, ((k % cols) * w, (k // cols) * cell_h))
comp = rf'{FIGS}\\training_curves_composite.png'
canvas.save(comp)
print('composite saved', canvas.size)

# insertion point: just before '4.2' heading (para 149)
anchor = paras[149]
assert anchor.text.startswith('4.2')

def insert_fig_before(anchor_p, img_path, width_in, caption):
    ip = anchor_p.insert_paragraph_before()
    ip.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ip.add_run().add_picture(img_path, width=Inches(width_in))
    cp = anchor_p.insert_paragraph_before()
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cp.add_run(caption)
    run.font.size = Pt(9)
    return ip, cp

insert_fig_before(anchor, comp, 5.8,
    'Figure 2 Training curves for all six DRL families across the five training seeds '
    '(per-family band plots; shaded region spans per-seed min-max, solid line is the '
    'seed mean). Every family converges to the random-phase floor, the basis of the '
    'null result.')
insert_fig_before(anchor, rf'{FIGS}\\fig_ao_convergence.png', 4.8,
    'Figure 3 AO convergence: sum rate versus AO iteration on a representative test '
    'channel (median 193 iterations to tolerance; the origin of the 32-times '
    'iteration-budget component of the latency ratio).')

# renumber CSI figure caption 2 -> 4
for p in d.paragraphs:
    if p.text.startswith('Figure 2 Achievable mean sum rate versus the fractional'):
        replace_in_para(p, 'Figure 2', 'Figure 4')
        print('csi figure renumbered to 4')
        break

d.save(DOC)
print('saved')

a = docx.Document(DOC)
print('verify: paragraphs', len(a.paragraphs), '| tables', len(a.tables),
      '| images', len(a.inline_shapes))
print('158:', a.paragraphs[158].text[:120])
