# -*- coding: utf-8 -*-
"""Injects the report body into the university template, keeping its styles intact.

Adapted from the pipeline in `D:\\Final year project\\Report\\` for this project.
Two things changed beyond paths:

  * citations are written in the prose as ``[key]`` and resolved to IEEE numbers
    in order of first appearance, so the reference list is always correctly
    ordered no matter how much the text is edited;
  * tables may be supplied inline as lists of rows, since this project's tables
    live in markdown reports rather than in result CSVs.
"""
import re, shutil, zipfile, os, sys
from xml.sax.saxutils import escape
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_content as C

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = r'D:\Final year project\Report\MSc Project Report Template.docx'
OUT = os.path.join(HERE, 'MSc_Project_Report_Skin_Fairness.docx')

FIGDIR = os.path.join(os.path.dirname(HERE), 'reports', 'figures')
EMU_PER_IN = 914400
MAX_W_IN = 5.9          # usable width inside the template's margins

_media = []             # (filename, bytes) queued for word/media/
_fig_n = [0]
_tab_n = [0]
_rels = []              # (rId, target) for images

# ---------------------------------------------------------------- citations

_cite_order = []        # citation keys, in order of first appearance

def _resolve_citations(text):
    """Rewrite [key] and [key1, key2] markers into IEEE numbers, registering
    each key the first time it is seen so the reference list can be emitted in
    citation order."""
    def repl(m):
        keys = [k.strip() for k in m.group(1).split(',')]
        if not all(k in C.CITATIONS for k in keys):
            return m.group(0)                      # not a citation, leave alone
        nums = []
        for k in keys:
            if k not in _cite_order:
                _cite_order.append(k)
            nums.append(_cite_order.index(k) + 1)
        return '[' + ', '.join(str(n) for n in sorted(nums)) + ']'
    return re.sub(r'\[([A-Za-z0-9_\-]+(?:\s*,\s*[A-Za-z0-9_\-]+)*)\]', repl, text)

# ---------------------------------------------------------------- primitives

def para(text, style='Normal'):
    return (f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
            f'<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>')

def figure(fname, caption):
    """Inline image + numbered caption. Caption style feeds the List of Figures."""
    from PIL import Image
    src = os.path.join(FIGDIR, fname)
    with Image.open(src) as im:
        pw, ph = im.size
    w_in = MAX_W_IN
    h_in = MAX_W_IN * ph / pw
    if h_in > 7.0:                       # keep tall figures on one page
        h_in = 7.0; w_in = 7.0 * pw / ph
    cx, cy = int(w_in * EMU_PER_IN), int(h_in * EMU_PER_IN)

    idx = len(_media) + 2                # image1.png already exists in the template
    newname = f'image{idx}.png'
    _media.append((newname, open(src, 'rb').read()))
    rid = f'rIdFig{idx}'
    _fig_n[0] += 1

    drawing = (
      '<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing>'
      f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
      f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
      f'<wp:docPr id="{100+idx}" name="Figure {_fig_n[0]}"/>'
      '<wp:cNvGraphicFramePr><a:graphicFrameLocks '
      'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/>'
      '</wp:cNvGraphicFramePr>'
      '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
      '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
      '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
      f'<pic:nvPicPr><pic:cNvPr id="{100+idx}" name="{newname}"/><pic:cNvPicPr/></pic:nvPicPr>'
      f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
      '<pic:spPr><a:xfrm><a:off x="0" y="0"/>'
      f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
      '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
      '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
    )
    # Caption built with a real SEQ field so Word numbers it and the List of
    # Figures can pick it up. A plain text caption would not appear in that list.
    cap = ('<w:p><w:pPr><w:pStyle w:val="Caption"/><w:jc w:val="center"/></w:pPr>'
           '<w:r><w:t xml:space="preserve">Figure </w:t></w:r>'
           '<w:fldSimple w:instr=" SEQ Figure \\* ARABIC ">'
           f'<w:r><w:t>{_fig_n[0]}</w:t></w:r></w:fldSimple>'
           f'<w:r><w:t xml:space="preserve">: {escape(_resolve_citations(caption))}</w:t></w:r></w:p>')
    return drawing + cap, rid, newname

def table(rows, caption):
    """Render a list of rows as a Word table, with the numbered caption above."""
    rows = [[str(c) for c in r] for r in rows]
    ncol = max(len(r) for r in rows)
    total = int(MAX_W_IN * 1440)
    colw = total // ncol
    _tab_n[0] += 1

    def cell(text, header=False):
        shade = '<w:shd w:val="clear" w:color="auto" w:fill="D9E2F3"/>' if header else ''
        bold = '<w:rPr><w:b/></w:rPr>' if header else ''
        return (f'<w:tc><w:tcPr><w:tcW w:w="{colw}" w:type="dxa"/>{shade}</w:tcPr>'
                f'<w:p><w:pPr><w:spacing w:after="0"/></w:pPr>'
                f'<w:r>{bold}<w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p></w:tc>')

    xml = ['<w:p><w:pPr><w:pStyle w:val="Caption"/></w:pPr>'
           '<w:r><w:t xml:space="preserve">Table </w:t></w:r>'
           '<w:fldSimple w:instr=" SEQ Table \\* ARABIC ">'
           f'<w:r><w:t>{_tab_n[0]}</w:t></w:r></w:fldSimple>'
           f'<w:r><w:t xml:space="preserve">: {escape(_resolve_citations(caption))}</w:t></w:r></w:p>']
    xml.append('<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
               f'<w:tblW w:w="{total}" w:type="dxa"/>'
               '<w:tblBorders>'
               + ''.join(f'<w:{e} w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
                         for e in ('top','left','bottom','right','insideH','insideV'))
               + '</w:tblBorders></w:tblPr>')
    xml.append('<w:tblGrid>' + ''.join(f'<w:gridCol w:w="{colw}"/>' for _ in range(ncol)) + '</w:tblGrid>')
    for i, r in enumerate(rows):
        r = list(r) + [''] * (ncol - len(r))
        trPr = '<w:trPr><w:tblHeader/></w:trPr>' if i == 0 else ''
        xml.append('<w:tr>' + trPr + ''.join(cell(c, i == 0) for c in r) + '</w:tr>')
    xml.append('</w:tbl>')
    xml.append(para(''))     # spacer so the next paragraph is not glued to the table
    return ''.join(xml)

def block(items):
    """Turn a list of strings into paragraphs. '3.1 Title' becomes a Heading2,
    [[FIG:file|caption]] an image, and a ('TABLE', caption, rows) tuple a table."""
    out = []
    for t in items:
        if isinstance(t, tuple) and t and t[0] == 'TABLE':
            out.append(table(t[2], t[1]))
            continue
        mf = re.match(r'^\[\[FIG:([^|]+)\|(.+)\]\]$', t)
        if mf:
            xml, rid, name = figure(mf.group(1), mf.group(2))
            _rels.append((rid, name))
            out.append(xml)
        elif re.match(r'^\d\.\d\s+[A-Z]', t) and len(t) < 90:
            out.append(para(re.sub(r'^\d\.\d\s+', '', t), 'Heading2'))
        else:
            out.append(para(_resolve_citations(t)))
    return ''.join(out)

# ------------------------------------------------------- content assembly
# Built in document order so that citation numbering comes out ascending.

SIGN_OFF = (para(f'Name: {C.AUTHOR}')
            + para('Date: ______________________')
            + para('Signature: ______________________')
            + para(''))

AFTER = {}
for heading, items in [
    ('Acknowledgements', C.ACKNOWLEDGEMENTS),
    ('Abstract', C.ABSTRACT),
    ('Introduction', C.CH1_INTRO),
    ('Problem Description, Context and Motivation', C.CH1_PROBLEM),
    ('Objectives', C.CH1_OBJECTIVES),
    ('Methodology', C.CH1_METHODOLOGY),
    ('Legal, Social, Ethical and Professional Considerations', C.CH1_LEGAL),
    ('Background', C.CH1_BACKGROUND),
    ('Structure of Report', C.CH1_STRUCTURE),
    ('Literature - Technology Review', C.CH2_INTRO),
    ('Literature Review', C.CH2_LIT),
    ('Technology Review', C.CH2_TECH),
    ('Summary', C.CH2_SUMMARY),
    ('Implementation', C.CH3_IMPL),
    ('Evaluation and Results', C.CH4_EVAL),
    ('Related Works', C.CH4_RELATED),
    ('Conclusion', C.CH5_CONC),
    ('Future Work', C.CH5_FUTURE),
    ('Reflection', C.CH5_REFLECT),
]:
    AFTER[heading] = block(items)

for k, v in C.APPENDICES.items():
    AFTER[k] = block(v)

def _reference_list():
    """Emitted last, so every citation has been seen and numbered."""
    missing = [k for k in C.CITATIONS if k not in _cite_order]
    if missing:
        print('UNCITED (dropped from the list):', ', '.join(sorted(missing)))
    return ''.join(para(f'[{i+1}] {C.CITATIONS[k]}')
                   for i, k in enumerate(_cite_order))

def toc_field(kind):
    """A real Table of Figures / Tables field. The template leaves these headings
    empty, so without this there is nothing for Word to populate on refresh."""
    return ('<w:p><w:r><w:fldChar w:fldCharType="begin"/></w:r>'
            f'<w:r><w:instrText xml:space="preserve"> TOC \\h \\z \\c "{kind}" </w:instrText></w:r>'
            '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
            '<w:r><w:t xml:space="preserve">Select this list and press F9 to populate it.</w:t></w:r>'
            '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>')

AFTER['List of Figures'] = toc_field('Figure')
AFTER['List of Tables'] = toc_field('Table')

BEFORE = {'Acknowledgements': SIGN_OFF}

FILL = [
    ('Project Report Title', C.TITLE),
    ('Your Name', C.AUTHOR),
    ('Computing /Data Science /Web Development', 'Data Science'),
    ('Subtitle if required', C.SUBTITLE),
]

def main():
    AFTER['References'] = _reference_list()

    zin = zipfile.ZipFile(TEMPLATE)
    doc = zin.read('word/document.xml').decode('utf-8')

    # Fill the title-page fields. Word splits text across several <w:r> runs, so a
    # phrase visible in the document often is not contiguous in the XML. Match on the
    # paragraph's combined text instead, then rewrite that paragraph's runs as one.
    def fill_titles(xml):
        def repl(m):
            p = m.group(0)
            text = ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p)).strip()
            for placeholder, value in FILL:
                if text == placeholder:
                    pPr = re.search(r'<w:pPr>.*?</w:pPr>', p, re.S)
                    rPr = re.search(r'<w:rPr>.*?</w:rPr>', p, re.S)
                    return ('<w:p>' + (pPr.group(0) if pPr else '') +
                            '<w:r>' + (rPr.group(0) if rPr else '') +
                            f'<w:t xml:space="preserve">{escape(value)}</w:t></w:r></w:p>')
            return p
        return re.sub(r'<w:p[ >].*?</w:p>', repl, xml, flags=re.S)

    doc = fill_titles(doc)

    # split the body into paragraph-level chunks, keeping non-paragraph xml intact
    parts = re.split(r'(<w:p[ >].*?</w:p>)', doc, flags=re.S)

    used, out = set(), []
    for p in parts:
        if not p.startswith('<w:p'):
            out.append(p); continue

        style_m = re.search(r'w:pStyle w:val="([^"]+)"', p)
        style = style_m.group(1) if style_m else 'Normal'
        text = ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p)).strip()

        # drop the template's blue guidance notes and every placeholder variant
        # (the template uses both 'Insert your text here' and 'Insert text here')
        if style == 'Guidancenotes' or re.fullmatch(
                r'insert\s+(your\s+)?text\s+here[.:]?', text, re.I):
            continue

        key0 = text.replace('\u2013', '-').strip()
        if style in ('Heading1', 'Heading2', 'Headingbackmatter') and key0 in BEFORE:
            out.append(BEFORE.pop(key0))

        out.append(p)

        if style in ('Heading1', 'Heading2', 'Headingbackmatter'):
            key = text.replace('\u2013', '-').replace('\u2014', '-').strip()
            for k in AFTER:
                if k not in used and (key == k or key.replace('-', '\u2013') == k):
                    out.append(AFTER[k]); used.add(k); break

    newdoc = ''.join(out)

    # The template's guidance boxes were tables. Deleting the guidance paragraphs
    # empties them but leaves the borders, which render as blank boxes - strip any
    # table left with no text at all.
    def drop_empty_tables(xml):
        def repl(m):
            body = m.group(0)
            text = ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', body)).strip()
            return '' if not text else body
        return re.sub(r'<w:tbl>.*?</w:tbl>', repl, xml, flags=re.S)

    newdoc = drop_empty_tables(newdoc)

    zout = zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED)
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == 'word/document.xml':
            data = newdoc.encode('utf-8')
        elif item.filename == 'word/_rels/document.xml.rels' and _rels:
            rels = data.decode('utf-8')
            add = ''.join(
                f'<Relationship Id="{rid}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                f'Target="media/{name}"/>' for rid, name in _rels)
            rels = rels.replace('</Relationships>', add + '</Relationships>')
            data = rels.encode('utf-8')
        zout.writestr(item, data)
    for name, blob in _media:
        zout.writestr(f'word/media/{name}', blob)
    zout.close(); zin.close()

    missing = [k for k in AFTER if k not in used]
    print(f'inserted {len(used)}/{len(AFTER)} sections; '
          f'{_fig_n[0]} figures, {_tab_n[0]} tables, {len(_cite_order)} references')
    if missing:
        print('NOT MATCHED:', missing)
    print('written:', OUT)

if __name__ == '__main__':
    main()
