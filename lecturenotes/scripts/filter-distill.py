#!/usr/bin/env python3
"""
Pandoc filter to handle distill.pub-specific conversions
"""
import pandocfilters as pf
import re

def distill_filter(key, value, format, meta):
    """Filter to convert markdown to distill.pub HTML"""
    
    # Convert footnotes to distill side notes (d-footnote)
    if key == "Note":
        # Convert footnote content to HTML
        def to_html(elem):
            if isinstance(elem, dict):
                t = elem.get("t")
                c = elem.get("c", [])
                if t == "Math":
                    math_type, math_content = c[0], c[1]
                    # MathJax will handle this
                    return f"${math_content}$" if math_type == "InlineMath" else f"$${math_content}$$"
                elif t == "Str":
                    return c
                elif t == "Space":
                    return " "
                elif t == "SoftBreak":
                    return " "
                elif t == "Emph":
                    return f"<em>{''.join(to_html(x) for x in c)}</em>"
                elif t == "Strong":
                    return f"<strong>{''.join(to_html(x) for x in c)}</strong>"
                elif t == "Para":
                    return ''.join(to_html(x) for x in c)
                else:
                    return ''.join(to_html(x) for x in c) if isinstance(c, list) else str(c)
            elif isinstance(elem, list):
                return ''.join(to_html(x) for x in elem)
            return str(elem)
        
        # Get footnote content - Note structure is [note_id, content]
        if isinstance(value, list) and len(value) >= 2:
            footnote_content = value[1]  # Content is second element
        else:
            footnote_content = value
        
        text = ''.join(to_html(x) for x in footnote_content) if isinstance(footnote_content, list) else to_html(footnote_content)
        return pf.RawInline("html", f'<d-footnote>{text}</d-footnote>')
    
    # Handle margin notes - convert to distill side notes
    if key == "Div" and isinstance(value, list) and len(value) >= 2:
        attrs, content = value[0], value[1]
        classes = attrs[1] if len(attrs) >= 2 else []
        
        if "marginnote" in classes:
            text = pf.stringify({"t": "Para", "c": content})
            return pf.RawBlock("html", f'<d-footnote>{text}</d-footnote>')
    
    # Handle raw LaTeX blocks - skip for HTML
    if key == "CodeBlock":
        attrs, code = value[0], value[1]
        if attrs and len(attrs) >= 2:
            classes = attrs[1]
            if "latex" in classes:
                # Skip LaTeX-only content for HTML
                return []
    
    # Handle images - convert to distill figure format
    if key == "Image":
        attrs, caption, target = value[0], value[1], value[2]
        url, title = target[0], target[1] if len(target) > 1 else ""
        caption_text = pf.stringify(caption) if caption else ""
        
        return pf.RawBlock("html",
            f'<figure>\n'
            f'  <img src="{url}" alt="{caption_text}">\n'
            f'  <figcaption>{caption_text}</figcaption>\n'
            f'</figure>')

if __name__ == "__main__":
    pf.toJSONFilter(distill_filter)
