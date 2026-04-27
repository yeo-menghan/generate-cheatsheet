#!/usr/bin/env python3
"""
Generate a printable, editable multi-column dual-page cheatsheet from the
markdown files in ./markdowns.

Outputs (next to this script):
  - cheatsheet.md   : a single combined markdown (editable source of truth)
  - cheatsheet.html : {columns}-column, double-sided, print-ready (Cmd/Ctrl-P -> Save as PDF)

The HTML uses CSS multi-column layout so content reflows naturally across
columns and pages. Math is rendered with MathJax. Tweak FONT_PT, COLUMNS,
and PAGE settings below to fit more/less content per page.

Usage:
    python3 generate-cheatsheet.py
    # then open cheatsheet.html in a browser and print (landscape, 2-sided).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent
MD_DIR = ROOT / "markdowns"
OUT_MD = ROOT / "cheatsheet.md"
OUT_HTML = ROOT / "cheatsheet.html"

# ---- Layout knobs ---------------------------------------------------------
PAGES = 2              # force fit into exactly this many pages
COLUMNS = 6            # columns per page
FONT_PT_START = 5.0    # starting body font size; auto-fit will shrink as needed
FONT_PT_MIN = 3.5      # don't shrink below this
LINE_HEIGHT = 1.0
PAGE = "A4 landscape"   # or "A4 landscape"
MARGIN_IN = 0.1
COL_GAP_IN = 0.1
TITLE = "DSA5103 Cheatsheet"
# -------------------------------------------------------------------


def natural_key(p: Path):
    m = re.search(r"(\d+)", p.stem)
    return (int(m.group(1)) if m else 0, p.stem)


def load_markdowns() -> str:
    files = sorted(MD_DIR.glob("*.md"), key=natural_key)
    chunks = []
    for f in files:
        chunks.append(f.read_text(encoding="utf-8").strip())
    return "\n\n---\n\n".join(chunks) + "\n"


def write_combined_md(md: str) -> None:
    header = f"# {TITLE}\n\n_Combined from `markdowns/`. Edit this file freely; re-run the script to refresh the HTML._\n\n---\n\n"
    OUT_MD.write_text(header + md, encoding="utf-8")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script>
  window.MathJax = {{
    tex: {{
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }},
    svg: {{ fontCache: 'global' }},
    startup: {{ typeset: false }}
  }};
</script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  :root {{
    --font-pt: {font_pt}pt;
    --line-height: {line_height};
    --col-gap: {col_gap}in;
    --margin: {margin}in;
    /* page-content size (landscape letter = 11in x 8.5in by default) */
    --page-w: {page_w}in;
    --page-h: {page_h}in;
    --content-w: calc(var(--page-w) - 2 * var(--margin));
    --content-h: calc(var(--page-h) - 2 * var(--margin));
  }}
  @page {{ size: {page}; margin: var(--margin); }}

  html, body {{ margin: 0; padding: 0; }}
  body {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: var(--font-pt);
    line-height: var(--line-height);
    color: #111;
  }}

  /* Toolbar (hidden on print) */
  .toolbar {{
    position: fixed; top: 0; left: 0; right: 0; background: #fafafa;
    border-bottom: 1px solid #ddd; padding: 6px 10px; font-size: 11pt; z-index: 10;
  }}
  .toolbar button {{ font-size: 11pt; margin-right: 8px; }}
  body {{ padding-top: 38px; }}
  @media print {{ .toolbar {{ display: none; }} body {{ padding-top: 0; }} }}

  /* Each .page is exactly one printed page; columns flow within it. */
  .page {{
    width: var(--content-w);
    height: var(--content-h);
    margin: var(--margin) auto;
    column-count: {columns};
    column-gap: var(--col-gap);
    column-fill: auto;
    column-rule: 0.5pt solid #bbb;
    overflow: hidden;            /* clip anything beyond the 2 pages */
    box-sizing: border-box;
    background: white;
  }}
  @media screen {{
    .page {{ box-shadow: 0 0 6px rgba(0,0,0,0.15); margin-bottom: 14px; }}
    body {{ background: #eee; }}
  }}
  @media print {{
    .page {{ margin: 0; box-shadow: none; page-break-after: always; }}
    .page:last-child {{ page-break-after: auto; }}
  }}

  .page > * {{ break-inside: avoid-column; }}
  h1, h2, h3, h4 {{ break-after: avoid; margin: 0.30em 0 0.15em; line-height: 1.12; }}
  h1 {{ font-size: 1.40em; border-bottom: 1px solid #333; padding-bottom: 1px; }}
  h2 {{ font-size: 1.18em; border-bottom: 1px solid #888; padding-bottom: 1px; }}
  h3 {{ font-size: 1.04em; }}
  h4 {{ font-size: 1.00em; font-style: italic; }}
  p  {{ margin: 0.14em 0; }}
  ul, ol {{ margin: 0.14em 0 0.14em 1.0em; padding: 0; }}
  li {{ margin: 0.03em 0; }}
  hr {{ border: none; border-top: 0.5pt dashed #888; margin: 0.30em 0; }}
  code {{ font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 0.92em; background: #f3f3f3; padding: 0 2px; border-radius: 2px; }}
  pre {{ background: #f3f3f3; padding: 3px 5px; border-radius: 3px; overflow: hidden; font-size: 0.85em; line-height: 1.15; break-inside: avoid; }}
  pre code {{ background: none; padding: 0; }}
  blockquote {{ margin: 0.18em 0; padding: 0.1em 0.5em; border-left: 2px solid #888; color: #333; }}
  table {{ border-collapse: collapse; font-size: 0.92em; margin: 0.20em 0; }}
  th, td {{ border: 0.4pt solid #888; padding: 1px 4px; text-align: left; }}

  mjx-container {{ font-size: 92% !important; }}
  mjx-container[display="true"] {{ margin: 0.15em 0 !important; overflow-x: hidden; }}

  #status {{ font-size: 10pt; color: #555; margin-left: 8px; }}
</style>
</head>
<body>
  <div class="toolbar">
    <strong>{title}</strong> &mdash;
    <button onclick="window.print()">Print / Save PDF</button>
    <span style="color:#666;">{pages} pages × {columns} cols. Auto-fit shrinks the font until it fits.</span>
    <span id="status"></span>
  </div>

  <script id="raw" type="text/markdown">{raw_md}</script>
  <div id="measure" style="position:absolute; left:-99999px; top:0;"></div>
  <div id="pages"></div>

<script>
  const PAGES = {pages};
  const FONT_START = {font_pt};
  const FONT_MIN   = {font_min};

  const raw = document.getElementById('raw').textContent;
  marked.setOptions({{ gfm: true, breaks: false }});

  // Protect $$...$$ and $...$ math from markdown processing (which mangles _, *, \).
  // Replace each math span with a placeholder, parse markdown, then restore.
  function protectMath(src) {{
    const store = [];
    const stash = (tex, display) => {{
      const idx = store.length;
      store.push({{ tex, display }});
      return `@@MATH${{idx}}@@`;
    }};
    // display first ($$...$$), then inline ($...$). Avoid escaped \$.
    src = src.replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => stash(t, true));
    src = src.replace(/(^|[^\\])\$([^\n$]+?)\$/g, (m, pre, t) => pre + stash(t, false));
    // \[ ... \] and \( ... \)
    src = src.replace(/\\\[([\s\S]+?)\\\]/g, (_, t) => stash(t, true));
    src = src.replace(/\\\(([\s\S]+?)\\\)/g, (_, t) => stash(t, false));
    return {{ src, store }};
  }}
  function restoreMath(html, store) {{
    return html.replace(/@@MATH(\d+)@@/g, (_, i) => {{
      const m = store[+i];
      // Use \( \) / \[ \] so MathJax picks them up regardless of $ delimiters.
      return m.display ? `\\[${{m.tex}}\\]` : `\\(${{m.tex}}\\)`;
    }});
  }}

  const protectedSrc = protectMath(raw);
  let renderedHTML = marked.parse(protectedSrc.src);
  renderedHTML = restoreMath(renderedHTML, protectedSrc.store);

  // Build N page containers and dump all content into the first; overflow hidden
  // means anything past page N is clipped — auto-fit avoids that by shrinking text.
  const pagesEl = document.getElementById('pages');
  for (let i = 0; i < PAGES; i++) {{
    const p = document.createElement('div');
    p.className = 'page';
    pagesEl.appendChild(p);
  }}

  // Put the entire rendered HTML in a flat list of block children inside page 1.
  // We then move overflow children into subsequent pages until they fit.
  const allHtml = document.createElement('div');
  allHtml.innerHTML = renderedHTML;
  const blocks = Array.from(allHtml.children);
  const firstPage = pagesEl.firstElementChild;
  blocks.forEach(b => firstPage.appendChild(b));

  function distributeAcrossPages() {{
    // Reset: move every block back into page 1
    const pages = Array.from(pagesEl.children);
    for (let i = 1; i < pages.length; i++) {{
      while (pages[i].firstChild) pages[0].appendChild(pages[i].firstChild);
    }}
    // For pages 1..N-1, while content overflows, push trailing blocks to next page
    for (let i = 0; i < pages.length - 1; i++) {{
      const cur = pages[i];
      const nxt = pages[i + 1];
      // multi-column: overflow when scrollHeight > clientHeight OR scrollWidth > clientWidth
      let guard = 0;
      while ((cur.scrollHeight > cur.clientHeight + 1 || cur.scrollWidth > cur.clientWidth + 1) && cur.children.length > 1 && guard++ < 5000) {{
        nxt.insertBefore(cur.lastElementChild, nxt.firstChild);
      }}
    }}
    const last = pages[pages.length - 1];
    return !(last.scrollHeight > last.clientHeight + 1 || last.scrollWidth > last.clientWidth + 1);
  }}

  async function waitForMathJax() {{
    // MathJax is loaded async; poll until typesetPromise is available.
    for (let i = 0; i < 200; i++) {{
      if (window.MathJax && window.MathJax.typesetPromise) return true;
      await new Promise(r => setTimeout(r, 50));
    }}
    return false;
  }}

  async function autofit() {{
    const status = document.getElementById('status');
    const ready = await waitForMathJax();
    if (ready) {{
      try {{ await MathJax.typesetPromise([pagesEl]); }}
      catch (e) {{ console.error('MathJax typeset failed', e); }}
    }} else {{
      console.warn('MathJax did not load in time');
    }}
    let pt = FONT_START;
    document.documentElement.style.setProperty('--font-pt', pt + 'pt');
    let fits = false;
    for (let step = 0; step < 40; step++) {{
      // Re-typeset is unnecessary for size changes (em-based), just redistribute.
      fits = distributeAcrossPages();
      if (fits) break;
      pt = Math.max(FONT_MIN, pt - 0.2);
      document.documentElement.style.setProperty('--font-pt', pt + 'pt');
      if (pt <= FONT_MIN) {{
        fits = distributeAcrossPages();
        break;
      }}
      // allow layout to settle
      await new Promise(r => requestAnimationFrame(r));
    }}
    status.textContent = fits
      ? `fit at ${{pt.toFixed(1)}}pt`
      : `still overflowing at ${{FONT_MIN}}pt — content trimmed; reduce material or raise FONT_MIN floor.`;
  }}

  autofit();
</script>
</body>
</html>
"""


def write_html(md: str) -> None:
    # In a <script type="text/markdown"> block the only sequence we must neutralize
    # is a closing </script> tag. Leave $, \, _, *, < etc. intact so MathJax/marked
    # can process them on the client side.
    safe = re.sub(r"</(script)", r"<\\/\1", md, flags=re.IGNORECASE)
    # Page dimensions for the @page size keyword we use (letter/a4 landscape).
    if "a4" in PAGE.lower():
        page_w, page_h = (11.69, 8.27)
    else:  # letter
        page_w, page_h = (11.0, 8.5)
    html = HTML_TEMPLATE.format(
        title=TITLE,
        raw_md=safe,
        columns=COLUMNS,
        pages=PAGES,
        font_pt=FONT_PT_START,
        font_min=FONT_PT_MIN,
        line_height=LINE_HEIGHT,
        col_gap=COL_GAP_IN,
        margin=MARGIN_IN,
        page=PAGE,
        page_w=page_w,
        page_h=page_h,
    )
    OUT_HTML.write_text(html, encoding="utf-8")


def main() -> None:
    if not MD_DIR.is_dir():
        raise SystemExit(f"markdowns folder not found: {MD_DIR}")
    md = load_markdowns()
    write_combined_md(md)
    write_html(md)
    print(f"Wrote {OUT_MD.relative_to(ROOT)} and {OUT_HTML.relative_to(ROOT)}")
    print("Open cheatsheet.html in a browser, then Print -> Save as PDF (landscape, 2-sided).")


if __name__ == "__main__":
    main()
