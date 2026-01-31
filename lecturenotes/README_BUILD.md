# Building the Lecture Notes

This project uses markdown as the source format and can compile to both LaTeX (PDF) and distill.pub (HTML) formats using **Pandoc**.

## Prerequisites

- [Pandoc](https://pandoc.org/) (version 2.19 or later)
- Python 3 with `pandocfilters` package: `pip install pandocfilters`
- For LaTeX output: A LaTeX distribution (TeX Live, MacTeX, etc.)
- For distill.pub output: A web server (for local viewing)

## Building

### Build both formats
```bash
make all
```

### Build LaTeX only
```bash
make latex
# or
bash build-latex.sh
```

### Build distill.pub only
```bash
make distill
# or
bash build-distill.sh
```

## Output

- LaTeX output: `build/latex/main.pdf`
- Distill.pub output: `build/distill/` (serve with a web server)

## How It Works

The build system uses **Pandoc** to convert markdown to LaTeX and HTML:

1. **LaTeX Build** (`build-latex.sh`):
   - Combines all markdown chapters into a single file
   - Uses Pandoc's `book.latex` template
   - Includes custom preamble and math commands
   - Compiles to PDF using `pdflatex`

2. **Distill Build** (`build-distill.sh`):
   - Converts each chapter to HTML
   - Uses distill.pub template structure
   - Creates an interactive webpage

## Project Structure

- `chapters/*.md` - Markdown source files for each chapter
- `templates/book.latex` - Pandoc LaTeX template for book format
- `templates/distill-*.html` - Distill.pub HTML templates
- `scripts/` - Pandoc filters for format-specific conversions
- `preamble.tex` - LaTeX preamble (packages, styles)
- `math_commands.tex` - LaTeX math command definitions
- `build/` - Generated output (created during build)

## Writing in Markdown

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

## Customization

- **LaTeX styling**: Edit `preamble.tex` and `templates/book.latex`
- **Distill styling**: Edit `templates/distill-post.html` and `templates/distill-index.html`
- **Conversion logic**: Modify filters in `scripts/filter-latex.py` and `scripts/filter-distill.py`

## Troubleshooting

### LaTeX compilation errors

- Check that all required packages are installed
- Ensure `preamble.tex` and `math_commands.tex` are in `build/latex/`
- Check that figures are in `build/latex/figures/`

### Pandoc filter errors

- Ensure `pandocfilters` is installed: `pip install pandocfilters`
- Check Python version (requires Python 3)
