# Round-16 (DD-058): fit literature tables 1-6 (docx tables[0..5]) inside the
# 6.63" text column. Word sizes tables by tblGrid, so the grid columns, every
# cell's tcW, the table width and the fixed layout are all rewritten at XML
# level (round-15's tcW-only assignment didn't move the grid: the real bug).
import docx
import docx.shared as sh
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

DOC = r'C:\Users\user\OneDrive\Desktop\All Assignments\CN project\TeamNo_6_24BDS0166_RIS_6G_Report.docx'
d = docx.Document(DOC)

def set_table_width(t, new_in):
    tbl = t._tbl
    total = sh.Inches(new_in)
    # 1) tblGrid gridCols, with a floor so no column collapses
    cols = tbl.tblGrid.findall(qn('w:gridCol'))
    cur = [int(c.get(qn('w:w')) or 0) for c in cols]
    tot_cur = sum(cur) or 1
    raw = [max(w * total / tot_cur, sh.Inches(0.50)) for w in cur]
    over = sum(raw) - total
    if over > 0:                       # shave the surplus off the widest column
        k = max(range(len(raw)), key=lambda i: raw[i])
        raw[k] -= over
    for g, w in zip(cols, raw):
        g.set(qn('w:w'), str(int(w)))
    # 2) every cell's tcW
    for row in t.rows:
        tc_ws = []
        for g, c in zip(cols, row.cells):
            tcPr = c._tc.get_or_add_tcPr()
            for old in tcPr.findall(qn('w:tcW')):
                tcPr.remove(old)
            tcW = OxmlElement('w:tcW')
            tcW.set(qn('w:w'), str(int(g.get(qn('w:w')))))
            tcW.set(qn('w:type'), 'dxa')
            tcPr.append(tcW)
            tc_ws.append(c)
    return raw

for ti in range(6):                       # tables[0..5] = literature Tables 1-6
    t = d.tables[ti]
    ws = set_table_width(t, 6.58)
    # fixed layout + explicit table width
    tblPr = t._tbl.tblPr
    for tag in ('w:tblLayout', 'w:tblW'):
        for old in tblPr.findall(qn(tag)):
            tblPr.remove(old)
    layout = OxmlElement('w:tblLayout')
    layout.set(qn('w:type'), 'fixed')
    tblPr.append(layout)
    tblW = OxmlElement('w:tblW')
    tblW.set(qn('w:w'), str(int(sum(int(x) for x in ws))))
    tblW.set(qn('w:type'), 'dxa')
    tblPr.append(tblW)
    # 3) font 8 pt everywhere in the table
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = sh.Pt(8)
    print('table %d scaled to %.2f in: %s' % (ti, sum(int(x) for x in ws) / 914400,
                                              [round(int(x) / 914400, 2) for x in ws]))

d.save(DOC)

# ---- verify by re-opening and summing grid widths ----
a = docx.Document(DOC)
for ti in range(6):
    t = a.tables[ti]
    grid = t._tbl.tblGrid.findall(qn('w:gridCol'))
    tot = sum(int(g.get(qn('w:w'))) for g in grid) / 914400
    print('verify table %d grid total = %.2f in' % (ti, tot))
