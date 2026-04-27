# generate-cheatsheet

Turn a folder of markdown notes into a printable, multi-column, dual-page cheatsheet — editable as markdown, viewable as HTML, printable to PDF. Math is rendered with MathJax.

Originally built for DSA5103 lecture notes, but works for any set of markdown files.

## What it does

- Concatenates every `.md` in `markdowns/` (sorted naturally: `lec1`, `lec2`, …, `lec10`) into a single `cheatsheet.md`.
- Generates a `cheatsheet.html` that lays the content out in a fixed number of pages × columns (default: 2 pages × 6 columns, A4 landscape).
- Renders LaTeX math (`$...$`, `$$...$$`, `\(...\)`, `\[...\]`) via MathJax, protecting it from markdown's `_`/`*`/`\` mangling.
- Auto-shrinks the body font until everything fits within the page budget; the toolbar shows the chosen size or warns if the content overflows the floor.
- Designed to be printed double-sided (flip on long edge) for a true two-sided crib sheet.

## Usage

```bash
python3 generate-cheatsheet.py
```

This writes:

- `cheatsheet.md` — combined markdown source (edit freely; re-run to refresh).
- `cheatsheet.html` — open in a browser, click **Print / Save PDF** (landscape, two-sided).

No dependencies beyond Python 3 and a modern browser. `marked` and `MathJax` are loaded from a CDN at view time, so the HTML needs network access on first load.

## Tuning

The layout knobs live at the top of `generate-cheatsheet.py`:

| Knob | Default | Notes |
|---|---|---|
| `PAGES` | `2` | Hard page budget. Content beyond it is clipped (with a warning). |
| `COLUMNS` | `6` | Columns per page. |
| `FONT_PT_START` | `5.0` | Auto-fit starts here and shrinks. |
| `FONT_PT_MIN` | `3.5` | Floor for auto-fit. Raise if 3.5pt is unreadable. |
| `LINE_HEIGHT` | `1.0` | |
| `PAGE` | `"A4 landscape"` | Or `"letter landscape"`. |
| `MARGIN_IN` | `0.1` | Page margin in inches. |
| `COL_GAP_IN` | `0.1` | Gap between columns. |
| `TITLE` | `"DSA5103 Cheatsheet"` | Shown in the toolbar and combined `.md`. |

## Repo layout

```
generate-cheatsheet.py    # the generator
markdowns/                # your source notes (one .md per topic)
cheatsheet.md             # generated; combined source
cheatsheet.html           # generated; print this
```

## License

[MIT](LICENSE) © 2026 Yeo Meng Han.
