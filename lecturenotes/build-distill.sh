#!/bin/bash
# Build script for distill.pub output from markdown

set -e

# Create output directory
mkdir -p build/distill

echo "Building distill.pub website..."

# Create distill site structure
mkdir -p build/distill/{_posts,assets,figures}

# Convert each chapter to a distill post
for md_file in chapters/*.md; do
    if [ -f "$md_file" ]; then
        base_name=$(basename "$md_file" .md)
        
        # Skip notation checklist for now (can be added as appendix)
        if [[ "$base_name" == "A_notation_checklist" ]]; then
            continue
        fi
        
        echo "Processing $md_file..."
        
        # Extract title from first heading
        title=$(grep '^# ' "$md_file" | head -1 | sed 's/^# //' || echo "Untitled")
        
        # Build pandoc command for individual posts
        PANDOC_POST_ARGS=(
            "$md_file"
            "--from=markdown+tex_math_dollars+raw_html"
            "--to=html"
            "--output=build/distill/_posts/${base_name}.html"
            "--template=templates/distill-post.html"
            "--mathjax"
            "--standalone"
            "--metadata=title:$title"
            "--metadata=date:$(date +%Y-%m-%d)"
            "--metadata=author:Peter Henderson"
        )
        
        # Add filter if available
        if python3 -c "import pandocfilters" 2>/dev/null; then
            PANDOC_POST_ARGS+=("--filter=scripts/filter-distill.py")
        fi
        
        pandoc "${PANDOC_POST_ARGS[@]}" || true
    fi
done

# Create main index page with distill template
cat > build/distill/index.html <<HTML_EOF
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Reinforcement Learning: From Foundations to Frontiers</title>
    <script src="https://distill.pub/template.v2.js"></script>
    <link rel="stylesheet" href="https://distill.pub/template.v2.css">
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
      window.MathJax = {
        tex: {
          inlineMath: [["$", "$"], ["\\(", "\\)"]],
          displayMath: [["$$", "$$"], ["\\[", "\\]"]]
        }
      };
    </script>
</head>
<body>
    <d-front-matter>
        <script type="text/json">{
            "title": "Reinforcement Learning: From Foundations to Frontiers",
            "authors": [{
                "author": "Peter Henderson",
                "authorURL": "#"
            }],
            "katex": {
                "delimiters": [
                    {"left": "$$", "right": "$$", "display": true},
                    {"left": "$", "right": "$", "display": false}
                ]
            }
        }</script>
    </d-front-matter>
    
    <d-title>
        <h1>Reinforcement Learning: From Foundations to Frontiers</h1>
        <p>Lecture notes on reinforcement learning</p>
    </d-title>
    
    <d-article>
HTML_EOF

# Convert and append chapters
for md_file in chapters/00_preface.md chapters/00_intro_rl.md; do
    if [ -f "$md_file" ]; then
        echo "Adding $md_file to index..."
        
        # Build pandoc command
        PANDOC_ARGS=(
            "$md_file"
            "--from=markdown+tex_math_dollars+raw_html"
            "--to=html"
            "--mathjax"
        )
        
        # Add filter if pandocfilters is available
        if python3 -c "import pandocfilters" 2>/dev/null; then
            PANDOC_ARGS+=("--filter=scripts/filter-distill.py")
        fi
        
        pandoc "${PANDOC_ARGS[@]}" >> build/distill/index.html || true
    fi
done

# Post-process to convert standard footnotes to distill side notes
# This handles cases where the filter wasn't available
if ! python3 -c "import pandocfilters" 2>/dev/null; then
    echo "Converting footnotes to side notes..."
    # Use sed/perl to convert <sup><a href="#fn1">1</a></sup> and <div class="footnotes"> to distill format
    python3 <<'PYTHON_EOF'
import re
import sys

html_file = 'build/distill/index.html'
try:
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Convert footnote references: <sup><a href="#fn1" class="footnote-ref" id="fnref1">1</a></sup>
    # to: <d-footnote>content</d-footnote>
    
    # Find all footnotes section
    footnote_pattern = r'<div class="footnotes">\s*<hr\s*/?>\s*<ol>(.*?)</ol>\s*</div>'
    footnote_section = re.search(footnote_pattern, content, re.DOTALL)
    
    if footnote_section:
        # Extract individual footnotes
        footnote_items = re.findall(r'<li id="fn(\d+)"[^>]*>(.*?)</li>', footnote_section.group(1), re.DOTALL)
        
        # Create a mapping of footnote IDs to content
        footnotes = {}
        for fn_id, fn_content in footnote_items:
            # Clean up the footnote content
            fn_content = re.sub(r'<p>|</p>', '', fn_content)
            fn_content = re.sub(r'\s+', ' ', fn_content).strip()
            # Remove backlink
            fn_content = re.sub(r'<a href="#fnref\d+"[^>]*>.*?</a>', '', fn_content)
            footnotes[fn_id] = fn_content
        
        # Replace footnote references in text with <d-footnote>
        for fn_id, fn_content in footnotes.items():
            # Find references like <sup><a href="#fn1" ...>1</a></sup>
            ref_pattern = rf'<sup><a href="#fn{fn_id}"[^>]*id="fnref{fn_id}"[^>]*>{fn_id}</a></sup>'
            replacement = f'<d-footnote>{fn_content}</d-footnote>'
            content = re.sub(ref_pattern, replacement, content)
        
        # Remove the footnotes section
        content = re.sub(footnote_pattern, '', content, flags=re.DOTALL)
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ Converted footnotes to side notes")
except Exception as e:
    print(f"Warning: Could not convert footnotes: {e}")
PYTHON_EOF
fi

# Close HTML
cat >> build/distill/index.html <<HTML_EOF
    </d-article>
    
    <d-appendix>
        <h3>Notation</h3>
HTML_EOF

# Add notation checklist
if [ -f "chapters/A_notation_checklist.md" ]; then
    pandoc chapters/A_notation_checklist.md \
        --from=markdown+tex_math_dollars+raw_html \
        --to=html \
        --filter=scripts/filter-distill.py \
        --mathjax >> build/distill/index.html || true
fi

cat >> build/distill/index.html <<HTML_EOF
    </d-appendix>
</body>
</html>
HTML_EOF

# Copy figures
if [ -d "figures" ]; then
    cp -r figures/* build/distill/figures/ 2>/dev/null || true
fi

# Create a simple CSS file for custom styling
{
    echo "/* Custom styles for distill.pub output */"
    echo "d-article {"
    echo "    max-width: 65em;"
    echo "}"
    echo ""
    echo "d-footnote {"
    echo "    font-size: 0.9em;"
    echo "}"
} > build/distill/assets/style.css

echo "Distill build complete! Output: build/distill/"
echo "To view, serve build/distill/ with a web server:"
echo "  cd build/distill && python3 -m http.server 8000"
