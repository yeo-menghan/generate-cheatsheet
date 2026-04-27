# generate-cheatsheet

Turn a folder of markdown notes into a printable, multi-column, dual-page cheatsheet — editable as markdown, viewable as HTML, printable to PDF. Math is rendered with MathJax.

Originally built for DSA5103 lecture notes, but works for any set of markdown files.

## What it does

- Concatenates every `.md` in `markdowns/` into a single `cheatsheet.md` (see [Ordering](#ordering) below).
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

## Ordering

You don't need `lec1.md`, `lec2.md`, … — files are sorted with a *natural sort* on the first number found in each filename, falling back to the filename itself. So all of these work:

- `lec1.md, lec2.md, …, lec10.md` → numeric order (`lec10` after `lec9`, not after `lec1`).
- `01-intro.md, 02-gradients.md, …` → numeric order via the prefix. Zero-pad if you have more than 9 files.
- `intro.md, gradients.md, optimization.md` → plain alphabetical (no numbers found).

A file with no digits in its name is treated as `0` and sorts to the top. If you want a strict custom order, the simplest fix is to prefix filenames with `01-`, `02-`, etc.

## Repo layout

```
generate-cheatsheet.py    # the generator
markdowns/                # your source notes (one .md per topic)
cheatsheet.md             # generated; combined source
cheatsheet.html           # generated; print this
```

## License

[MIT](LICENSE) © 2026 Yeo Meng Han.
