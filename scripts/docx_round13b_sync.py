# Round-13 stage 2 (DD-055 cont.): abstract to 200-230 words, references
# [20]-[27] into the docx bibliography, prose citations in the literature review.
import docx

SRC = r'C:\Users\user\OneDrive\Desktop\All Assignments\CN project\TeamNo_6_24BDS0166_RIS_6G_Report.docx'
d = docx.Document(SRC)

def paros(fn):
    for i, p in enumerate(d.paragraphs):
        if fn(p.text):
            return i, p
    return None, None

# -- abstract pad --
ABSTRACT_ADD = (
    ' Rationale, retired measurements and every protocol decision are disclosed in the '
    'accompanying decision log; the study operates at 3.5 GHz with simplified i.i.d.-Gaussian '
    'fading and zero-forcing reception, so none of the conclusions extends outside this regime.')
i, p = paros(lambda t: t.startswith('This report evaluates direct deep-reinforcement-learning'))
p.text = p.text.rstrip() + ABSTRACT_ADD
print('abstract now %d words' % len(p.text.split()))

# -- 2.2 / 2.3 prose citations for the new sources --
i22, p22 = paros(lambda t: 'Learning-designed RIS configurations predate' in t)
p22.text = p22.text.rstrip().removesuffix('.') + (
    '. Recent single-family studies continue this pattern: DRL phase control in mmWave '
    'high-speed-rail links [22], DRL optimization of RIS for 6G multi-user NLOS connectivity '
    '[24], and deep-reinforcement learning for discrete-phase beam-pattern synthesis of mmWave '
    'surfaces [27] each evaluate one method family in its own setup; the channel-acquisition '
    'problem that motivates partial-observation designs is likewise addressed inside dedicated '
    'estimation frameworks, from LMMSE-to-deep-learning solutions to super-resolution estimators '
    '[20], [21].')
i23, p23 = paros(lambda t: 'Beyond policies that pick the phases' in t)
p23.text = p23.text.rstrip().removesuffix('.') + (
    '. Robust-DRL designs for energy-harvesting RIS terminals [23], fast hybrid-learning '
    'beamforming [25] and machine-learning beamforming overviews [26] similarly build on their '
    'own simulators and reward definitions, which reinforces the cross-paper comparability '
    'problem summarized in Section 2.6.')

# -- reference paragraphs [20]-[27] after [19] --
NEWREFS = [
 "[20] N. K. Kundu and M. R. McKay, \u201cChannel estimation for reconfigurable intelligent "
 "surface aided MISO communications: From LMMSE to deep learning solutions,\u201d IEEE Open "
 "Journal of the Communications Society, vol. 2, pp. 471-487, 2021, doi: 10.1109/OJCOMS.2021.3063171.",
 "[21] W. Shen, Z. Qin, and A. Nallanathan, \u201cDeep learning for super-resolution channel "
 "estimation in reconfigurable intelligent surface aided systems,\u201d IEEE Transactions on "
 "Communications, vol. 71, no. 3, pp. 1491-1503, 2023, doi: 10.1109/TCOMM.2023.3239621.",
 "[22] J. Xu and B. Ai, \u201cWhen mmWave high-speed railway networks meet reconfigurable "
 "intelligent surface: A deep reinforcement learning method,\u201d IEEE Wireless Communications "
 "Letters, vol. 11, no. 3, pp. 533-537, 2022, doi: 10.1109/LWC.2021.3135602.",
 "[23] H. Peng and L.-C. Wang, \u201cEnergy harvesting reconfigurable intelligent surface for "
 "UAV based on robust deep reinforcement learning,\u201d IEEE Transactions on Wireless "
 "Communications, vol. 22, no. 10, pp. 6826-6838, 2023, doi: 10.1109/TWC.2023.3245820.",
 "[24] Z. Aasa, \u201cDeep reinforcement learning-based optimization of reconfigurable "
 "intelligent surfaces (RIS) for 6G multi-user connectivity in NLOS environments,\u201d "
 "Scientific Reports, 2026, doi: 10.1038/s41598-026-69294-9.",
 "[25] S. Sivasankar and S. Markkandan, \u201cFast and scalable intelligent reflecting surface "
 "beamforming using hybrid learning approaches,\u201d IEEE Access, vol. 13, pp. 152826-152842, "
 "2025, doi: 10.1109/ACCESS.2025.3603206.",
 "[26] A. Ahmadinejad and S. Talebi, \u201cBeamforming design via machine learning in "
 "intelligent reflecting surface-aided wireless communication,\u201d Physical Communication, "
 "vol. 68, 102586, 2025, doi: 10.1016/j.phycom.2024.102586.",
 "[27] P. K. Nishad and G. Prasad, \u201cDeep reinforcement learning-based arbitrary beam "
 "pattern synthesis for millimeter-wave reconfigurable intelligent surfaces with discrete phase "
 "shifts,\u201d Engineering Research Express, vol. 8, no. 20, 205339, 2026, "
 "doi: 10.1088/2631-8695/aeb46e.",
]
i, p = paros(lambda t: '10.3390/s23052554' in t)
for txt in NEWREFS:
    p.insert_paragraph_before(txt)  # after [19], in order: insert before next-sibling trick
# insert_paragraph_before inserted them before the [19] paragraph in order; fix by moving:
# simplest correction: gather and re-apply after the [19] paragraph using addnext chain
arts = [q for q in d.paragraphs if any(q.text.startswith(r[:20]) and r[:8] in q.text[:22]
                                      for r in NEWREFS)]
print('inserted-in-wrong-place count:', len(arts))
ip, p19 = paros(lambda t: '10.3390/s23052554' in t)
for q in arts[::-1]:
    el = q._p
    el.getparent().remove(el)
    p19._p.addnext(el)
print('references repositioned after [19] (P%d)' % ip)

d.save(SRC)

# verify
a = docx.Document(SRC)
adt = [q.text for q in a.paragraphs]
refs = [t[:8] for t in adt if t.strip().startswith('[2')]
print('tail refs:', refs)
print('abstract wc final:', len(adt[12].split()))
