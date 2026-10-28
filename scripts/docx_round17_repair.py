# Round-17 / REPAIR (DD-059): undo the schema-invalid manual XML from round-16
# (hand-appended w:tblLayout/w:tblW in tblPr and w:tcW at wrong tcPr positions
# triggered Word's repair-on-open, which visually drops content - the user's
# "you vanished everything"). Re-implement the Table 1-6 width fix purely via
# python-docx's own property API, which inserts children in correct schema
# order: table.autofit=False (w:tblLayout), table.columns[i].width (w:gridCol),
# cell.width (w:tcW via get_or_add_tcPr), and remove stray tcPr leftovers.
import docx
import docx.shared as sh
from docx.oxml.ns import qn

DOC = r'C:\Users\user\OneDrive\Desktop\All Assignments\CN project\TeamNo_6_24BDS0166_RIS_6G_Report.docx'
d = docx.Document(DOC)

TARGET_TOTALS = {0: 6.58, 1: 6.58, 2: 6.58, 3: 6.58, 4: 6.58, 5: 6.58}

for ti in range(6):
    t = d.tables[ti]
    tbl = t._tbl
    tblPr = tbl.tblPr
    # 1) remove the manually appended (wrong-order) elements
    for tag in ('w:tblLayout', 'w:tblW'):
        for el in tblPr.findall(qn(tag)):
            tblPr.remove(el)
    # 2) compute proportional grid from current grid widths
    cols = list(t.columns)
    cur = [int(g.get(qn('w:w')) or 0) for g in [c._gridCol for c in cols]]
    tot = sum(cur) or 1
    total_emu = sh.Inches(TARGET_TOTALS[ti])
    raw = [max(w * total_emu / tot, sh.Inches(0.5)) for w in cur]
    over = sum(raw) - total_emu
    if over > 0:
        k = max(range(len(raw)), key=lambda i: raw[i])
        raw[k] -= over
    # 3) set via python-docx API only
    t.autofit = False                       # writes w:tblLayout fixed, ordered correctly
    for c, w in zip(cols, raw):
        c.width = int(w)                    # writes w:gridCol/@w
    for row in t.rows:
        for c, w in zip(row.cells, raw):
            c.width = int(w)                # get_or_add_tcW via python-docx (correct order)
    # 4) font 8pt
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = sh.Pt(8)
    got = [x.inches if hasattr(x, 'inches') else None for x in [col.width for col in cols]]
    print('table %d grid now: %s (total %.2f in)' % (
        ti, [round((col.width or 0) / 914400, 2) for col in cols],
        sum((col.width or 0) for col in cols) / 914400))

d.save(DOC)

# verify round-trip
a = docx.Document(DOC)
for ti in range(6):
    t = a.tables[ti]
    tot = sum(int(g.get(qn('w:w'))) for g in t._tbl.tblGrid.findall(qn('w:gridCol'))) / 914400
    print('verify tbl%d grid=%.2f in | autofit=%s' % (ti, tot, t.autofit))
print('paragraphs:', len(a.paragraphs), '| tables:', len(a.tables),
      '| images:', len(a.inline_shapes))
