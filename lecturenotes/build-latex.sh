#!/bin/bash
# Build script for LaTeX output from markdown using Pandoc

set -e

# Create output directory
mkdir -p build/latex

echo "Building LaTeX PDF from markdown..."

# Copy necessary files
cp preamble.tex build/latex/ 2>/dev/null || true
cp math_commands.tex build/latex/ 2>/dev/null || true
cp references.bib build/latex/ 2>/dev/null || true

# Copy figures directory
mkdir -p build/latex/figures
cp -r figures/* build/latex/figures/ 2>/dev/null || true

# Combine all chapters into a single markdown file
echo "Combining chapters..."
{
    # Add YAML header
    cat <<'YAML_EOF'
---
title: "Reinforcement Learning: From Foundations to Frontiers"
author: "Peter Henderson"
date: \today
documentclass: book
fontsize: 11pt
geometry: margin=1in
toc: true
toc-depth: 2
numbersections: false
colorlinks: true
linkcolor: mymauve
citecolor: echodrk
bibliography: references.bib
---

YAML_EOF

    # Add preface
    if [ -f "chapters/00_preface.md" ]; then
        echo ""
        echo "\\section*{Preface}"
        echo "\\addcontentsline{toc}{section}{Preface}"
        echo "\\markboth{Preface}{Preface}"
        echo ""
        cat chapters/00_preface.md
        echo ""
    fi

    # Add notation
    if [ -f "chapters/A_notation_checklist.md" ]; then
        echo ""
        echo "\\section*{Notation}"
        echo "\\addcontentsline{toc}{section}{Notation}"
        echo "\\markboth{Notation}{Notation}"
        echo ""
        cat chapters/A_notation_checklist.md
        echo ""
    fi

    # Add intro
    if [ -f "chapters/00_intro_rl.md" ]; then
        echo ""
        echo "\\newpage"
        echo ""
        cat chapters/00_intro_rl.md
        echo ""
    fi

} > build/latex/combined.md

# Convert to LaTeX using Pandoc
echo "Converting to LaTeX..."

# Get absolute path to project root (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Verify template exists
if [ ! -f "$PROJECT_ROOT/templates/book.latex" ]; then
    echo "Error: Template not found at $PROJECT_ROOT/templates/book.latex"
    exit 1
fi

cd build/latex

# Build pandoc command with proper argument handling
PANDOC_ARGS=(
    "combined.md"
    "--from=markdown+tex_math_dollars+raw_tex+yaml_metadata_block"
    "--to=latex"
    "--template=$PROJECT_ROOT/templates/book.latex"
    "--standalone"
    "--output=main.tex"
    "--bibliography=references.bib"
    "--citeproc"
    "--wrap=none"
)

# Add filter if pandocfilters is available
if python3 -c "import pandocfilters" 2>/dev/null; then
    PANDOC_ARGS+=("--filter=$PROJECT_ROOT/scripts/filter-latex.py")
    echo "Using Pandoc filter..."
else
    echo "Warning: pandocfilters not found. Some features may not work correctly."
    echo "Install with: pip install pandocfilters"
fi

# Execute pandoc command
pandoc "${PANDOC_ARGS[@]}"

# Compile LaTeX
echo "Compiling LaTeX..."
if command -v pdflatex &> /dev/null; then
    pdflatex -interaction=nonstopmode main.tex || true
    if [ -f main.aux ]; then
        bibtex main || true
    fi
    pdflatex -interaction=nonstopmode main.tex || true
    pdflatex -interaction=nonstopmode main.tex || true
    echo ""
    echo "✓ LaTeX build complete! Output: build/latex/main.pdf"
else
    echo "Warning: pdflatex not found. LaTeX file created but not compiled."
    echo "Output: build/latex/main.tex"
fi
