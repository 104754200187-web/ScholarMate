"""Markdown answer -> .docx (Word), pure stdlib.

The generated answer is what ends up pasted into a manuscript, and Word users
cannot open a Markdown file without a converter. A .docx is just a zip of XML
parts, so the subset that matters here -- headings, paragraphs, bullet lines,
bold spans, tables flattened to rows -- is written directly instead of adding a
converter dependency.
"""

import io
import re
import zipfile
from xml.sax.saxutils import escape

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

_DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

# Word needs the styles it is asked to use to exist; a serif default matches the
# manuscript-oriented exports (GB/T 7714 / CSL) rather than a UI font.
_STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr>
<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="SimSun"/>
<w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault>
</w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>
<w:pPr><w:outlineLvl w:val="0"/><w:spacing w:before="240" w:after="120"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/>
<w:pPr><w:outlineLvl w:val="1"/><w:spacing w:before="200" w:after="100"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/>
<w:pPr><w:outlineLvl w:val="2"/><w:spacing w:before="160" w:after="80"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Quote"><w:name w:val="Quote"/>
<w:rPr><w:i/></w:rPr></w:style>
</w:styles>"""

_SECTION = ('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>')

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^[-*+]\s+(.*)$")
_ORDERED = re.compile(r"^(\d+)[.)]\s+(.*)$")
_QUOTE = re.compile(r"^>\s?(.*)$")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_TABLE_RULE = re.compile(r"^\|?[\s:|-]+\|[\s:|-]*$")


def _clean(text):
    """XML 1.0 forbids most control characters; publishers do ship them."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text or "")


def _runs(text):
    """Inline markup -> [(text, flags)] for **bold** and `code`."""
    out = []
    for chunk in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", _LINK.sub(r"\1 (\2)", text)):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**") and len(chunk) > 4:
            out.append((chunk[2:-2], {"b": True}))
        elif chunk.startswith("`") and chunk.endswith("`") and len(chunk) > 2:
            out.append((chunk[1:-1], {"mono": True}))
        else:
            out.append((chunk, {}))
    return out


def _run_xml(text, flags):
    props = "<w:b/>" if flags.get("b") else ""
    if flags.get("mono"):
        props += '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>'
    props = f"<w:rPr>{props}</w:rPr>" if props else ""
    return f'<w:r>{props}<w:t xml:space="preserve">{escape(_clean(text))}</w:t></w:r>'


def _para_xml(text, style=None, indent=0):
    props = f'<w:pStyle w:val="{style}"/>' if style else ""
    if indent:
        props += f'<w:ind w:left="{indent}"/>'
    props = f"<w:pPr>{props}</w:pPr>" if props else ""
    runs = "".join(_run_xml(t, f) for t, f in _runs(text))
    return f"<w:p>{props}{runs}</w:p>"


def document_xml(markdown, title=""):
    """Body XML for a markdown answer (soft-wrapped lines join into paragraphs)."""
    body, buf = [], []

    def flush():
        if buf:
            body.append(_para_xml(" ".join(buf)))
            buf.clear()

    if title:
        body.append(_para_xml(title, "Heading1"))
    for raw in (markdown or "").splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if _TABLE_RULE.match(line):
            continue        # markdown separator row
        if line.startswith("|"):
            flush()
            cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
            body.append(_para_xml("  ｜  ".join(cells), indent=360))
            continue
        m = _HEADING.match(line)
        if m:
            flush()
            body.append(_para_xml(m.group(2).strip(), f"Heading{min(len(m.group(1)), 3)}"))
            continue
        m = _BULLET.match(line)
        if m:
            flush()
            body.append(_para_xml("• " + m.group(1).strip(), indent=360))
            continue
        m = _ORDERED.match(line)
        if m:
            flush()
            body.append(_para_xml(f"{m.group(1)}. {m.group(2).strip()}", indent=360))
            continue
        m = _QUOTE.match(line)
        if m:
            flush()
            body.append(_para_xml(m.group(1).strip(), "Quote", indent=360))
            continue
        buf.append(line)
    flush()
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{''.join(body)}{_SECTION}</w:body></w:document>")


def markdown_to_docx(markdown, title=""):
    """Whole .docx package for a markdown answer (bytes)."""
    parts = {
        "[Content_Types].xml": _CONTENT_TYPES,
        "_rels/.rels": _RELS,
        "word/_rels/document.xml.rels": _DOC_RELS,
        "word/styles.xml": _STYLES,
        "word/document.xml": document_xml(markdown, title),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    return buf.getvalue()


if __name__ == "__main__":
    import sys
    md = sys.stdin.read() if not sys.argv[1:] else open(sys.argv[1], encoding="utf-8").read()
    data = markdown_to_docx(md, "ScholarMate")
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/answer.docx"
    with open(out, "wb") as f:
        f.write(data)
    print(f"wrote {out} ({len(data)} bytes)")
