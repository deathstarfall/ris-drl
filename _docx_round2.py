# one-shot docx sync for external-review round 2 (logged as DD-044). Idempotent-ish.
import docx, json, shutil, os
from docx.shared import Inches

SRC = r'C:\Users\user\OneDrive\Desktop\All Assignments\CN project\TeamNo_6_24BDS0166_RIS_6G_Report.docx'
BAK = r'C:\Users\user\OneDrive\Desktop\All Assignments\CN project\TeamNo_6_24BDS0166_RIS_6G_Report_pre_round2_sync.docx'
FIGS = r'C:\code\ris-drl\paper\figs'
shutil.copyfile(SRC, BAK)
d = docx.Document(SRC)
LOG = []

def find_para(sub):
    for i, p in enumerate(d.paragraphs):
        if sub in p.text: return i, p
    return None, None

def set_text(p, new): p.text = new

def sub_in_para(p, old, new):
    if old not in p.text: return False
    if any(old in r.text for r in p.runs):
        for r in p.runs:
            if old in r.text: r.text = r.text.replace(old, new)
    else:
        p.text = p.text.replace(old, new)
    return True

def append(p, txt): p.add_run(txt)

# --- typos ---------------------------------------------------------------------
i, p = find_para('107 devices')
if p: sub_in_para(p, '107 devices', '10^7 devices')
i, p = find_para('tb per second')
if p: sub_in_para(p, 'tb per second', 'terabits per second')

# --- condense world-initiatives paragraph ---------------------------------------
i, p = find_para('IMT 2030 Promotion Group')
if p and 'programmable-surface' not in p.text:
    set_text(p, 'Reconfigurable intelligent surfaces are also known as intelligent reflecting surfaces. '
        'The structure of an RIS is a two-dimensional array of low-cost passive elements whose reflection '
        'coefficients can be programmed in software, which allows the propagation environment itself to be '
        'shaped rather than merely compensated for at the receiver. Programme bodies such as the ITU-R and '
        'the IMT-2030 Promotion Group list programmable-surface capabilities among candidate 6G technologies, '
        'and early hardware testbeds have demonstrated rate gains in blocking-rich indoor links, but these '
        'trials stop short of the frozen-protocol, equal-compute comparison this report performs.')

# --- MDP / bandit leftover -------------------------------------------------------
i, p = find_para('posed as a Markov decision process')
if p and 'contextual bandit: the' not in p.text:
    set_text(p, 'Phrased in MDP vocabulary, the problem degenerates to a one-step contextual bandit: the '
        '"state" is the current channel observation alone (each episode starts from a fresh channel, so there '
        'is no previous configuration to carry), the action is the M=64 dimensional phase vector (continuous '
        'or quantized to a 2, 3 or 4-bit codebook), and the episode terminates after a single decision whose '
        'reward is the zero-forcing sum rate. There is no discounting; the gamma reported in Table 6 is inert '
        'by construction.')

# --- SAC action space ------------------------------------------------------------
i, p = find_para('3 bit codebook was selected')
if p and 'DDPG, TD3 and SAC' not in p.text:
    set_text(p, 'In the published runs DDPG, TD3 and SAC all act in the continuous M=64 phase space; discrete '
        'codebook variants of DQN act over 2, 3 and 4-bit quantizations of each phase. The reward is the '
        'zero-forcing sum rate of Eq (2).')

# --- observation-stack correction (full sentence) --------------------------------
i, p = find_para('800 complex coefficients')
if p:
    sub_in_para(p, 'The state input is the flattened real and imaginary channel state (800 complex coefficients '
        'over BS-RIS, RIS-user and BS-user links), and the only optimization variable is the vector of 64 RIS '
        'reflection phases; the sum rate of the zero-forcing precoder is the reward.',
        'Each policy observes the four-part stack concat(|h_r|, angle h_r, |G|, |h_d|) flattened to 256 real '
        'values per decision; crucially this stack contains the magnitudes and the angle of h_r but NOT the '
        'angles of G or h_d, on which the optimal phases depend (quantified in Section 5). The only '
        'optimization variable is the vector of 64 RIS reflection phases; the zero-forcing sum rate is the reward.')

# --- ablation numbers -------------------------------------------------------------
i, p = find_para('headline model')
if p and '59.327' not in p.text:
    append(p, ' The loss ablation on the same frozen test set quantifies the cost of each shortcut: the '
        'rate-trained network scores 62.532 bps/Hz, the circular-loss variant 61.860, and the old '
        'MSE-on-wrapped-angles variant 59.327 (close to an untrained forward pass, 59.390).')

# --- Figure 1 image ---------------------------------------------------------------
i, p = find_para('Figure 1 Unified RIS evaluation framework')
if p and 'unified_framework.png' not in p._p.xml:
    pic = p.insert_paragraph_before(); pic.alignment = 1
    pic.add_run().add_picture(os.path.join(FIGS, 'unified_framework.png'), width=Inches(5.6))

# --- SNR statement ----------------------------------------------------------------
i, p = find_para('64.193 +/- 0.582')
if p and '39.9' not in p.text:
    append(p, ' In link-budget terms the post-zero-forcing channel is noise-limited: the mean per-user SNR is '
        '39.9 dB at random phases and 48.3 dB under the AO phases (noise power -107 dBm; under ZF, SINR = SNR), '
        'so the phase vector genuinely is the controllable degree of freedom.')

# --- robustness scope note ---------------------------------------------------------
i, p = find_para('fractional channel-state error magnitude')
if p:
    for j in range(i, max(i - 6, 0), -1):
        b = d.paragraphs[j]
        if 'degrades' in b.text or 'sweep' in b.text.lower():
            if 'DRL agents' not in b.text:
                append(b, ' Scope note: this sweep covers the AO reference, deep unfolding and the random '
                    'control; the DRL agents are evaluated under clean CSI only, as in the paper (a '
                    'robustness sweep over trained DRL checkpoints is left to the v1.1 campaign).')
            break

# --- matched-baseline fairness ------------------------------------------------------
i, p = find_para('1044x speed-up')
if p and '59.390' not in p.text:
    append(p, ' Two matched-discipline facts complete this comparison (added after external review): an '
        'untrained forward pass of the same six-layer network already reaches 59.390 bps/Hz, so the fixed '
        'six-step gradient structure alone supplies +6.21 over random and training adds a further +3.14; and '
        'the 1044x latency ratio is six unrolled layers against a 200-iteration AO solver - a per-step budget '
        'ratio of roughly 175x per solver step, not a matched-compute claim.')

# --- "slower than random" -------------------------------------------------------------
i, p = find_para('slower than random')
if p: sub_in_para(p, 'slower than random', 'below random in sum rate')

# --- DD range -------------------------------------------------------------------------
for paras in d.paragraphs:
    if 'DD-043' in paras.text and 'DD-044' not in paras.text:
        sub_in_para(paras, 'DD-043', 'DD-044')

# --- Table 6 true configs ---------------------------------------------------------------
for t in d.tables:
    txt = ' '.join(c.text for r in t.rows for c in r.cells)
    if 'OU' in txt and 'Huber' in txt:
        for r in t.rows:
            for c in r.cells:
                c.text = (c.text
                    .replace('critic 3\u00d710\u207b\u2074', 'critic 1\u00d710\u207b\u00b3')
                    .replace('OU noise, \u03c3 = 0.15', 'Gaussian noise, \u03c3 = 0.1')
                    .replace('Smooth L1 / Huber critic loss', 'MSE critic loss')
                    .replace('Automatic entropy tuning, Gumbel\u2013Softmax',
                             'Continuous Gaussian policy; entropy auto-tuned (\u03b1 init 0.2)'))
        for r in t.rows:
            cs = r.cells
            if cs[0].text.strip().lower().startswith('batch'): cs[1].text = '256'
            if cs[0].text.strip().lower().startswith('replay'): cs[1].text = '300,000 transitions'
        LOG.append('table6')

# --- affiliation + corresponding author before Keywords ---------------------------------
i, p = find_para('Keywords:')
if p and 'Vellore Institute of Technology' not in '\n'.join(q.text for q in d.paragraphs[:i]):
    aff = p.insert_paragraph_before('School of Electronic Engineering, Vellore Institute of Technology, '
                                   'Vellore, Tamil Nadu, India.')
    corr = p.insert_paragraph_before('Corresponding author: 24BDS0166 (see author list), e-mail available via '
                                     'the project repository.')

# --- Table 10 quantization --------------------------------------------------------------
i, decl = find_para('Declarations')
if decl and not any('Table 10' in q.text for q in d.paragraphs[:i] + d.paragraphs[i:i+3]):
    bs = json.load(open(r'C:\code\ris-drl\results\bits_sweep.json'))
    m = bs['mean_rate_by_method']; f3 = lambda v: '%.3f' % v
    cap = decl.insert_paragraph_before('Table 10  Post-hoc B-bit phase quantization on the frozen test set '
        '(500 channels); the dB-equivalent per-sublink penalty is 10 log10 2.')
    tbl = d.add_table(rows=4, cols=4)
    tbl.style = 'Table Grid1'
    for j, h in enumerate(['Method', '2-bit (4 levels)', '3-bit (8 levels)', '4-bit (16 levels)']):
        tbl.rows[0].cells[j].text = h
    for ri, key in enumerate(('ao', 'unfolding', 'random'), start=1):
        vals = m[key]
        for j, v in enumerate(['AO reference', 'Deep unfolding', 'Random phases'][ri-1:ri] + [f3(x) for x in vals]):
            tbl.rows[ri].cells[j].text = v
    decl._p.addprevious(tbl._element)
    decl.insert_paragraph_before('The quantization losses are 0.98 / 0.24 / 0.06 dB-equivalent for AO at '
        'B = 2/3/4 (gaps of 2.936 / 0.730 / 0.182 bps/Hz against the unquantized 64.193): graceful, monotone '
        'degradation in which a 2-bit codebook already retains over 70% of the AO gain over random. Deep '
        'unfolding tracks AO within 0.30 dB-equivalent at every B.')
    LOG.append('table10')

d.save(SRC)
print('saved; log:', LOG)
check = '\n'.join(q.text for q in docx.Document(SRC).paragraphs)
tblall = '\n'.join(' | '.join(c.text for c in r.cells) for t in docx.Document(SRC).tables for r in t.rows)
for x in ('10^7', 'terabits', 'below random', '59.390', 'Table 10', '300,000', 'Vellore Institute',
          '256-value observation stack', 'angles of G or h_d', '39.9 dB', 'DD-044', 'critic 1\u00d710\u207b\u00b3',
          'Gaussian noise', '256 | batches'.replace(' | batches', '')):
    where = 'paras' if x in check else ('tables' if x in tblall else 'MISSING')
    print(x, '->', where)
