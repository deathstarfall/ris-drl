# Round-6 docx sync, part A (DD-048): obs-size correction (1056), narrow null wording,
# Table 7 repair (SAC cell fixes + new TD3/DQN columns), Table 11 anchors + borders,
# editorial residue (title 'xx', duplicate abstract sentences, corpus claim, drafts note),
# survey references [17]-[19], affiliation + corresponding e-mail.
import copy
import docx

SRC = r'C:\Users\user\OneDrive\Desktop\All Assignments\CN project\TeamNo_6_24BDS0166_RIS_6G_Report.docx'
d = docx.Document(SRC)

# ---------------------------------------------------------------- title fix
p0 = d.paragraphs[0]
if p0.text.startswith('xx'):
    p0.runs[0].text = p0.text[2:]
    print('title cleaned:', p0.text[:60])

def setp(i, text):
    d.paragraphs[i].text = text

def subp(i, old, new):
    p = d.paragraphs[i]
    if old in p.text:
        p.text = p.text.replace(old, new)
        return True
    print('MISS', i, old[:40])
    return False

# ---------------------------------------------------------------- abstract P11 dedupe
setp(11, 'Reconfigurable intelligent surfaces (RIS) are arrays of passive reflecting devices that reshape the '
         'propagation environment to improve signal throughput, coverage and energy efficiency. Deep reinforcement '
         'learning (DRL) has been proposed as a model-free way to configure the surface phases, but reported benefits '
         'have rarely been compared under a common, frozen evaluation protocol.')

# ---------------------------------------------------------------- abstract P12 lean rewrite
setp(12, 'This report evaluates direct DRL phase policies (continuous-policy SAC, TD3 and DDPG; discrete codebook DQN '
         'at 2, 3 and 4-bit phase resolution), a model-driven deep unfolding baseline, and a Riemannian '
         'alternating-optimization (AO) reference in one simulated multi-user downlink. Under a frozen 500-channel '
         'test protocol, no trained architecture achieved a positive mean gain over random phase selection '
         '(53.18 bps/Hz): test-set deltas range from -0.10 to +0.04, every 95% bootstrap interval covers zero and '
         'every paired Wilcoxon test gives p >= 0.37; given these budgets, observation setup and reward structure, '
         'the DRL configurations fail to learn a useful phase policy. Deep unfolding (L=6 unrolled projected-gradient '
         'steps trained end-to-end on the sum rate) reaches 62.53 +/- 0.85 bps/Hz, 97.41% of the AO reference '
         '(64.19 +/- 0.58), at 1.295 ms median decision latency on a desktop CPU - a 1044x speed-up over the '
         'committed 1.352 s per-channel AO figure; an untrained forward pass of the same network already reaches '
         '59.390 bps/Hz, so the six-step gradient structure alone supplies +6.21 over random and training adds a '
         'further +3.14. A full-observation stress test (all thirty checkpoints retrained with complete '
         'channel-phase access, 1056 -> 1600 real inputs) reproduces the null: the best family gains only +0.03 '
         'bps/Hz over random, while a no-learning closed-form full-CSI policy reaches 56.944 (+3.77). Every reported '
         'value derives from committed machine-readable artifacts in the open repository.')

# ---------------------------------------------------------------- obs size: 1056 in four places
subp(76, 'Each policy observes the four-part stack concat(|h_r|, angle h_r, |G|, |h_d|) flattened to 256 real values '
         'per decision',
        'Each policy observes a four-part stack flattened to 1,056 real values per decision (2 x 256 from the '
        'magnitude and angle of h_r, 512 magnitudes from G, and 32 magnitudes from h_d)')
subp(81, 'fully connected actor and critic over the 256-value channel observation stack',
        'fully connected actor and critic over the 1,056-value channel observation stack')
subp(116, '(256 -> 1600 real values)', '(1056 -> 1600 real values)')

# ---------------------------------------------------------------- methods paragraph: drafts note out
subp(82, ' - not a Gumbel-Softmax codebook sampler as described in earlier drafts', '')
subp(82, ' - not a Gumbel-Softmax codebook sampler', '')

# ---------------------------------------------------------------- Section 2 corpus claim removed
subp(46, 'Tables 1 to 5 are the verified corpus. Four gaps remain.',
        'Tables 1 to 5 span five research directions that motivate this study. Four gaps remain.')

# ---------------------------------------------------------------- Table 1 caption
subp(28, '(all entries )', '(all entries checked against the original papers)')

# ---------------------------------------------------------------- P116 null wording narrowed
setp(116, 'The measured outcome is asymmetric rather than a photo-finish comparison. Under the frozen protocol no '
          'DRL architecture (six architectures, five seeds each, 200,000-step budgets) achieved a positive mean gain '
          'over random phase selection, and training plateaus across every seed show behaviour at random-level '
          'reward under episodic one-step bandit feedback. With 1,500 training channels, 200,000 steps per run, one '
          'scalar reward per 64-dimensional action and the default hyperparameters, these DRL configurations fail to '
          'learn rather than fail for lack of information. Three anchors make that concrete. First, a no-learning '
          'closed-form full-CSI heuristic committed in the repository reaches 56.944 +/- 1.884 bps/Hz (+3.77 over '
          'random, Wilcoxon p < 0.001), so a better policy exists when the full channel is available - yet complete '
          'input access did not teach the agents to find it in the stress test. Second, TD3 and DDPG sit at '
          '53.07-53.09 bps/Hz, essentially the all-zero-phase value 53.076, i.e. they collapse toward a static '
          'policy. Third, the Wilcoxon values in Table 11 use seed-averaged per-channel rates, which is why they are '
          'not significant; at the individual-seed level the weakest continuous families sit below random.')

# ---------------------------------------------------------------- Table 11: anchors + borders
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
t11 = None
for t in d.tables:
    if 'Full-CSI architecture' in ' '.join(c.text for c in t.rows[0].cells):
        t11 = t
if t11 is not None:
    r = t11.add_row()
    r.cells[0].text = 'All-zero phases (no learning)'
    r.cells[1].text = '53.076'
    r.cells[2].text = '-0.102 (Wilcoxon p = 0.51)'
    r = t11.add_row()
    r.cells[0].text = 'Closed-form full-CSI (no learning)'
    r.cells[1].text = '56.944 +/- 1.884'
    r.cells[2].text = '+3.767 (Wilcoxon p < 0.001)'
    # title the two new rows offshore of the "5 seeds" header semantics
    hdr = t11.rows[0]
    hdr.cells[0].text = 'Policy (per-channel rates; DRL families: 5 seeds)'
    # borders
    tblPr = t11._tbl.tblPr
    for tag in tblPr.findall(W + 'tblBorders'):
        tblPr.remove(tag)
    from docx.oxml.shared import OxmlElement
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement('w:' + edge)
        el.set(W + 'val', 'single')
        el.set(W + 'sz', '4')
        el.set(W + 'color', 'auto')
        borders.append(el)
    tblPr.append(borders)
    print('Table 11: anchor rows added, borders applied')
# caption update
for p in d.paragraphs:
    if p.text.startswith('Table 11'):
        p.text = ('Table 11  Full-observation stress test (30 checkpoints): every agent retrained with the complete '
                  'observation stack (|h_r|, angle h_r, |G|, angle G, |h_d|, angle h_d; 1056 -> 1600 real values); '
                  '6 architectures x 5 seeds x 200,000 env steps; deterministic fixed-order evaluation on the same '
                  'frozen 500 channels; random floor 53.178. Paired per-channel Wilcoxon versus the floor (computed '
                  'on seed-averaged rates): p = 0.16 to 0.53 across the six DRL families, none significant; best '
                  'family +0.026 bps/Hz (t ~ 0.3). Two no-learning anchors ground the comparison: all-zero phases '
                  '(53.076, below random) and the closed-form full-CSI policy (56.944 +/- 1.884, +3.77 over random, '
                  'p < 0.001). The stress-test null is therefore about what these DRL configurations learn, not '
                  'about how much channel information they see.')

# ---------------------------------------------------------------- Table 7 repair + new columns
params = None
for t in d.tables:
    if 'M1' in ' '.join(c.text for c in t.rows[0].cells):
        params = t
        break
ROWS_TD3 = {'Parameter': 'TD3 (continuous)',
            'Optimizer': 'Adam',
            'Learning rate': 'Actor 3×10⁻⁴, critic 1×10⁻³',
            'Batch size': '256',
            'Replay buffer': '300,000 transitions',
            'Training budget': '200,000 environment steps',
            'Exploration / regularisation': 'Gaussian noise, σ = 0.1; target policy smoothing',
            'Loss': 'Twin Q critics, delayed policy updates, smoothing 0.2'}
ROWS_DQN = {'Parameter': 'DQN (codebook, 2/3/4-bit)',
            'Optimizer': 'Adam',
            'Learning rate': '3×10⁻⁴',
            'Batch size': '256',
            'Replay buffer': '300,000 transitions',
            'Training budget': '200,000 env steps; target sync 1,000',
            'Exploration / regularisation': 'ε-greedy 1.0 → 0.05 (decay 100k steps)',
            'Loss': 'Huber Q-learning (δ = 1.0)'}
FIX = {'Replay buffer': {2: '300,000 transitions'},
       'Learning rate': {2: 'Actor 3×10⁻⁴; critic 1×10⁻³'}}
if params is not None:
    header = params.rows[0]
    for row in params.rows:
        c0 = row.cells[0].text.strip()
        if c0 == 'Learning rate':
            row.cells[2].text = 'Actor 3×10⁻⁴; critic 1×10⁻³'
        elif c0 == 'Replay buffer':
            row.cells[2].text = '300,000 transitions'
    # insert TD3 column after DDPG (index 1) and DQN column after SAC (now index 3)
    def insert_col_after(tbl, col_idx, new_texts, header_text):
        # col_idx counted BEFORE insertion
        for row in tbl.rows:
            src = row.cells[col_idx]
            new_tc = copy.deepcopy(src._tc)
            src._tc.addnext(new_tc)
        for row in tbl.rows:
            cells = row.cells  # re-query; new cell is at col_idx+1
            key = row.cells[0].text.strip()
            text = header_text if row._index == 0 else new_texts.get(key, '')
            row.cells[col_idx + 1].text = text
    insert_col_after(params, 1, ROWS_TD3, 'TD3 (continuous)')
    header_cells = params.rows[0]
    header_cells.cells[2].text = 'M2: SAC (continuous)'   # old M2 label, now 3rd data col
    insert_col_after(params, 3, ROWS_DQN, 'DQN (codebook, 2/3/4-bit)')
    print('Table 7: SAC cells fixed; TD3 + DQN columns added')
    for row in params.rows:
        print('  ', ' | '.join(c.text[:38] for c in row.cells))

# ---------------------------------------------------------------- intro reference for testbeds
p = d.paragraphs[17]
if '[8], [17]' not in p.text:
    p.text = p.text.replace('demonstrated rate gains in blocking-rich indoor links, but',
                            'demonstrated rate gains in blocking-rich indoor links [8], [17], but')

# ---------------------------------------------------------------- affiliation + email
subp(5, 'School of Electronics Engineering, Vellore Institute of Technology, Vellore, Tamil Nadu, India.',
        'Vellore Institute of Technology, Vellore, Tamil Nadu, India.')
subp(6, 'Contact: github.com/deathstarfall/ris-drl (issues); institutional e-mail finalized at journal submission.',
        'Contact: ayanshmanan2007@gmail.com (corresponding author); code at github.com/deathstarfall/ris-drl.')

# ---------------------------------------------------------------- references [17]-[19]
newrefs = [
    '[17] M. Tariq, S. Ahmad, M. A. Jan, and H. Song, "Deep learning aided intelligent reflective surfaces for 6G: '
    'A survey," ACM Computing Surveys, vol. 57, no. 3, 2024, doi: 10.1145/3696414.',
    '[18] K. M. Faisal and W. Choi, "Machine learning approaches for reconfigurable intelligent surfaces: A survey," '
    'IEEE Access, vol. 10, 2022, doi: 10.1109/ACCESS.2022.3157651.',
    '[19] A. A. Puspitasari and B. M. Lee, "A survey on reinforcement learning for reconfigurable intelligent '
    'surfaces in wireless communications," Sensors, vol. 23, no. 5, 2554, 2023, doi: 10.3390/s23052554.',
]
last = None
for pp in d.paragraphs:
    if pp.text.strip().startswith('[16]'):
        last = pp
        break
if last is not None:
    for txt in newrefs:
        el = copy.deepcopy(last._p)
        last._p.addnext(el)
        last = docx.text.paragraph.Paragraph(el, d)
        last.text = txt
    print('refs [17]-[19] inserted after [16]')

d.save(SRC)
print('saved')
