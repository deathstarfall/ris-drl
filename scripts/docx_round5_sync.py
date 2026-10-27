# round-5 docx sync (DD-047): Table-7 header label, related-work layout, editorial cleanup,
# full-CSI stress-test summary table (new Table 11).
import docx, shutil, json, os

SRC = r'C:\Users\user\OneDrive\Desktop\All Assignments\CN project\TeamNo_6_24BDS0166_RIS_6G_Report.docx'
BAK = r'C:\Users\user\OneDrive\Desktop\All Assignments\CN project\TeamNo_6_24BDS0166_RIS_6G_Report_pre_round5_sync.docx'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
shutil.copyfile(SRC, BAK)
d = docx.Document(SRC)

def sub_in(p, old, new):
    for r in p.runs:
        if old in r.text:
            r.text = r.text.replace(old, new); return True
    if old in p.text:
        p.text = p.text.replace(old, new); return True
    return False

def find(sub, start=0):
    for i, p in enumerate(d.paragraphs[start:], start):
        if sub in p.text: return i, p
    return None, None

# 1) Table 7 header label: SAC is continuous; the codebook arms are DQN ----------
for t in d.tables:
    h = [c.text for c in t.rows[0].cells]
    if any('M2: SAC + codebook' in x for x in h):
        for j, c in enumerate(t.rows[0].cells):
            if 'M2: SAC + codebook' in c.text:
                c.text = 'M2: SAC (continuous); DQN codebook arms'
        print('table 7 header corrected')

# 2) related-work layout: caption below its table + strays ------------------------
body = d.element.body
tabs = [ch for ch in body.iterchildren() if ch.tag == W + 'tbl']
cap = None  # locate caption paragraph element by text via wrapped Paragraph
paras = {id(p._p): p for p in d.paragraphs}
# caption "Table 6: Summary..." paragraph found in d.paragraphs; move it BEFORE its table
capP = find('Table 6: Summary of the related-work groups')[1]
if capP is not None:
    tbl6 = tabs[5]  # 6th table (tables[5]) = related-work table sits at body pos right before caption
    # confirm adjacency: previous sibling of caption should be the table
    prev = capP._p.getprevious()
    if prev is not None and prev.tag == W + 'tbl' and prev is tbl6:
        capP._p.addprevious(capP._p)  # no-op safeguard
        # move: remove caption element and insert before tbl6
        el = capP._p
        el.getparent().remove(el)
        tbl6.addprevious(el)
        print('Table 6 caption moved above its table')
# strip broken "()" citation placeholders in captions
for p in d.paragraphs:
    if p.text.strip().lower().startswith('table') and '()' in p.text:
        sub_in(p, ' ()', '')
        sub_in(p, '()', '')
        print('caption placeholder cleaned:', p.text[:46])

# 3) editorial cleanup: abstract tail + intro --------------------------------------
i, p = find('never published, never published')
if p:
    # rewrite the final sentences of the abstract cleanly
    tail_start = p.text.find('Latency protocol note:')
    if tail_start > 0:
        clean = (p.text[:tail_start] +
            'All per-channel latency figures are serialized single-solve wall clocks on one desktop CPU '
            '(the AO run\'s total-wall line is a parallel-elapsed figure across workers, not a per-channel '
            'number); a retired ~9.6 ms development-stage float32 benchmark is disclosed and was never part '
            'of any published number. Every reported value derives from committed machine-readable artifacts '
            '(results/ and ckpt/ in the open repository).')
        p.text = clean
        print('abstract tail cleaned')

i, p = find('After the worldwide rollout of 5G, attention has shifted toward the next')
if p:
    p.text = ('After the worldwide rollout of 5G, attention has shifted toward next-generation (broadly "6G") '
              'systems whose candidate feature lists include programmable radio environments. This report rests '
              'none of its performance claims on future-generation targets: it evaluates RIS phase control '
              'within one fixed mid-band regime under a frozen evaluation protocol.')
    print('intro P16 completed')

i, p = find('RIS implementation is a hard task')
if p:
    p.text = ('RIS deployment is a non-trivial optimization problem: near-optimal phase configurations must be '
        'computed for every user position and channel state. Classical methods - alternating optimization, '
        'semidefinite relaxation and manifold optimization - attain good solutions but require accurate, fresh '
        'channel state information, which is costly at high frequencies. Model-free deep reinforcement learning '
        'learns the configuration strategy from interaction alone, without a channel model; DQN and its variants '
        'handle finite phase states, while DDPG, PPO, SAC and TD3 handle continuous ones. Deep unfolding takes a '
        'third route: it reverses the iterations of a model-based optimizer into a fixed-depth network that '
        'executes in a single fast pass, keeping inductive bias that pure exploration lacks.')
    print('intro P18 rewritten')

# 4) full-CSI stress-test summary table (Table 11) ---------------------------------
from docx.oxml.shared import OxmlElement
from docx.text.paragraph import Paragraph

fam = {"SAC (continuous)": (53.164, 0.129), "TD3 (continuous)": (53.086, 0.011),
       "DDPG (continuous)": (53.065, 0.035), "DQN 2-bit codebook": (53.153, 0.083),
       "DQN 3-bit codebook": (53.164, 0.071), "DQN 4-bit codebook": (53.204, 0.037)}
i, anchor = find('best full-CSI family 53.204')
if anchor is None:
    i, anchor = find('full-observation stress test')
if anchor is not None:
    tbl = d.add_table(rows=7, cols=3)
    tbl.style = 'Table Grid1'
    for j, hcell in enumerate(['Full-CSI architecture (5 seeds)', 'Mean sum rate (bps/Hz)', 'Delta vs random floor']):
        tbl.rows[0].cells[j].text = hcell
    for ri, (k, (m, s)) in enumerate(fam.items(), start=1):
        tbl.rows[ri].cells[0].text = k
        tbl.rows[ri].cells[1].text = '%.3f +/- %.3f' % (m, s)
        tbl.rows[ri].cells[2].text = '%+.3f (p >= 0.94)' % (m - 53.178)
    # table right AFTER the discussion paragraph; caption directly above the table
    anchor._p.addnext(tbl._element)
    capel = OxmlElement('w:p')
    capPara = Paragraph(capel, anchor._parent)
    capPara.text = ('Table 11  Full-observation stress test (30 checkpoints): every agent retrained with the '
                    'complete observation stack (|h_r|, angle h_r, |G|, angle G, |h_d|, angle h_d; 256 -> 1600 real '
                    'values); 6 architectures x 5 seeds x 200,000 steps; deterministic fixed-order evaluation on '
                    'the same frozen 500 channels; random floor 53.178. Paired per-channel Wilcoxon vs the floor: '
                    'p >= 0.94 for every family; best family delta +0.03 (t ~ 0.3); no family exceeds random.')
    tbl._element.addprevious(capel)
    print('Table 11 inserted after paragraph', i)
d.save(SRC)
print('saved')
a = docx.Document(SRC)
print('tables now:', len(a.tables))
print('never dup gone:', 'never published, never published' not in '\n'.join(q.text for q in a.paragraphs))
for ti, t in enumerate(a.tables):
    h = [c.text for c in t.rows[0].cells]
    if any('M2' in x for x in h): print('params header:', h)
print('Table 11 present:', any('full-observation stress test' in ' '.join(c.text for c in r.cells) for t in a.tables for r in t.rows) or 'Table 11' in '\n'.join(q.text for q in a.paragraphs))
