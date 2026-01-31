# Reinforcement Learning: From Foundations to Frontiers

Lecture notes on reinforcement learning that can be compiled to both LaTeX (PDF) and distill.pub (HTML) formats.

## Quick Start

### Prerequisites

Install the required tools:

```bash
# Install Pandoc (macOS)
brew install pandoc

# Install Python dependencies
pip install pandocfilters

# For LaTeX output, ensure you have a LaTeX distribution installed
# (TeX Live, MacTeX, etc.)
```

### Building

Build both formats:
```bash
make all
```

Build LaTeX only:
```bash
make latex
```

Build distill.pub only:
```bash
make distill
```

## Project Structure

```
.
├── chapters/           # Markdown source files
│   ├── 00_preface.md
│   ├── 00_intro_rl.md
│   └── A_notation_checklist.md
├── figures/           # Figure files (TikZ, SVG, etc.)
├── templates/         # Pandoc templates
├── scripts/          # Pandoc filters
├── build/            # Generated output (created during build)
│   ├── latex/        # LaTeX/PDF output
│   └── distill/      # Distill.pub HTML output
├── references.bib     # Bibliography
├── preamble.tex      # LaTeX preamble
├── math_commands.tex  # LaTeX math commands
└── Makefile          # Build automation
```

## Writing Guidelines

### Math

Use standard LaTeX math syntax:
- Inline: `$x = y$`
- Display: `$$x = y$$`

### Margin Notes

Use fenced div syntax:
```markdown
::: {.marginnote}
This is a margin note.
:::
```

### Figures

For figures that need different handling in LaTeX vs HTML:

```markdown
```{=latex}
\input{figures/figure-name}
```

```{=html}
<figure>
  <img src="figures/figure.svg">
  <figcaption>Caption text</figcaption>
</figure>
```
```

### Citations

Use standard Pandoc citation syntax:
```markdown
[@citation-key]
```

## Output Formats

### LaTeX/PDF

The LaTeX output produces a book-style PDF with:
- Professional typography
- Margin notes
- TikZ figures
- Full bibliography support

Output: `build/latex/main.pdf`

### Distill.pub

The distill.pub output produces a modern, interactive webpage with:
- Beautiful typography
- Side notes (converted from margin notes)
- Interactive figures
- MathJax rendering

Output: `build/distill/` (serve with a web server)

## Customization

- **LaTeX styling**: Edit `preamble.tex` and `templates/latex-main.tex`
- **Distill styling**: Edit `templates/distill-post.html` and `templates/distill-index.html`
- **Conversion logic**: Modify filters in `scripts/filter-latex.py` and `scripts/filter-distill.py`

## Development

The build system uses:
- **Pandoc** for markdown to LaTeX/HTML conversion
- **Python filters** for format-specific transformations
- **Make** for build automation

See `README_BUILD.md` for detailed build instructions.
