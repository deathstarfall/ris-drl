# Round-18 (DD-060): restyle Table 7 (comparison of prior studies, docx
# tables[6]) to match the report's other tables: 'Table Grid1' style, fixed
# layout, table-level borders, shaded+bold+centred header row, vAlign top.
import docx
import docx.shared as sh
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

DOC = r'C:\Users\user\OneDrive\Desktop\All Assignments\CN project\TeamNo_6_24BDS0166_RIS_6G_Report.docx'
d = docx.Document(DOC)
t = d.tables[6]

# 1) style like the results tables (Table Grid1)
style = d.tables[7].style
t.style = style

tblPr = t._tbl.tblPr

def ensure(tag):
    for old in tblPr.findall(qn(tag)):
        tblPr.remove(old)
    return OxmlElement(tag)

# 2) table-level borders (single, 4pt = 0.5pt) exactly like tbl11's
borders = ensure('w:tblBorders')
for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
    e = OxmlElement('w:' + edge)
    e.set(qn('w:val'), 'single')
    e.set(qn('w:sz'), '4')
    e.set(qn('w:space'), '0')
    e.set(qn('w:color'), 'auto')
    borders.append(e)
# schema order: tblStyle, tblW, jc, tblInd, tblBorders, tblLook
look = tblPr.find(qn('w:tblLook'))
if look is not None:
    look.addprevious(borders)          # borders before tblLook

# 3) fixed layout + autofit off via python-docx
t.autofit = False
layout = tblPr.find(qn('w:tblLayout'))

# 4) header row: bold, shaded, vAlign centre like the other table headers
for c in t.rows[0].cells:
    for p in c.paragraphs:
        for run in p.runs:
            run.bold = True
    tcPr = c._tc.get_or_add_tcPr()
    for old in tcPr.findall(qn('w:shd')):
        tcPr.remove(old)
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), 'D9D9D9')
    tcW = tcPr.find(qn('w:tcW'))
    if tcW is not None:
        tcW.addnext(shd)
    else:
        tcPr.insert(0, shd)
    va = OxmlElement('w:vAlign')
    va.set(qn('w:val'), 'top')
    tcPr.append(va)

# 5) keep grid at the 6.63" full-text width it already has; widths are per-cell
# already set (python-docx tcW, schema-ordered).

d.save(DOC)

a = docx.Document(DOC)
t = a.tables[6]
print('style:', t.style.name, '| autofit:', t.autofit)
print('tblPr:', [e.tag.split('}')[1] for e in t._tbl.tblPr])
tc = t.rows[0].cells[0]._tc.find(qn('w:tcPr'))
print('hdr tcPr:', [e.tag.split('}')[1] for e in tc])
print('hdr bold:', any(r.bold for p in t.rows[0].cells[0].paragraphs for r in p.runs))
grid = sum(int(g.get(qn('w:w'))) for g in t._tbl.tblGrid.findall(qn('w:gridCol')))
print('grid twips total:', grid, '=', round(grid / 1440, 2), 'in')
