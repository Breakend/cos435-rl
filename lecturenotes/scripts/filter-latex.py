#!/usr/bin/env python3
"""
Pandoc filter to handle LaTeX-specific conversions
"""
import pandocfilters as pf
import re

def latex_filter(key, value, format, meta):
    """Filter to convert markdown to LaTeX with proper handling"""
    
    # Convert footnotes to margin notes
    if key == "Note":
        # Convert footnote content to LaTeX
        def to_latex(elem):
            if isinstance(elem, dict):
                t = elem.get("t")
                c = elem.get("c", [])
                if t == "Math":
                    math_type, math_content = c[0], c[1]
                    return f"${math_content}$" if math_type == "InlineMath" else f"$${math_content}$$"
                elif t == "Str":
                    s = c
                    # Escape LaTeX special chars
                    s = s.replace("\\", "\\textbackslash{}")
                    s = s.replace("&", "\\&")
                    s = s.replace("%", "\\%")
                    s = s.replace("$", "\\$")
                    s = s.replace("#", "\\#")
                    s = s.replace("^", "\\textasciicircum{}")
                    s = s.replace("_", "\\_")
                    s = s.replace("{", "\\{")
                    s = s.replace("}", "\\}")
                    s = s.replace("~", "\\textasciitilde{}")
                    return s
                elif t == "Space":
                    return " "
                elif t == "SoftBreak":
                    return " "
                elif t == "Emph":
                    return f"\\emph{{{''.join(to_latex(x) for x in c)}}}"
                elif t == "Strong":
                    return f"\\textbf{{{''.join(to_latex(x) for x in c)}}}"
                elif t == "Para":
                    return ''.join(to_latex(x) for x in c)
                else:
                    return ''.join(to_latex(x) for x in c) if isinstance(c, list) else str(c)
            elif isinstance(elem, list):
                return ''.join(to_latex(x) for x in elem)
            return str(elem)
        
        # Get footnote content - Note structure is [note_id, content]
        if isinstance(value, list) and len(value) >= 2:
            footnote_content = value[1]  # Content is second element
        else:
            footnote_content = value
        
        text = ''.join(to_latex(x) for x in footnote_content) if isinstance(footnote_content, list) else to_latex(footnote_content)
        return pf.RawInline("latex", f"\\marginnote{{{text}}}")
    
    # Handle margin notes (explicit divs)
    if key == "Div" and isinstance(value, list) and len(value) >= 2:
        attrs, content = value[0], value[1]
        classes = attrs[1] if len(attrs) >= 2 else []
        
        if "marginnote" in classes:
            # Convert content blocks to LaTeX inline
            # Build LaTeX string preserving structure
            def to_latex(elem):
                if isinstance(elem, dict):
                    t = elem.get("t")
                    c = elem.get("c", [])
                    if t == "Math":
                        math_type, math_content = c[0], c[1]
                        return f"${math_content}$" if math_type == "InlineMath" else f"$${math_content}$$"
                    elif t == "Str":
                        s = c
                        # Escape LaTeX special chars (but not in math mode)
                        s = s.replace("\\", "\\textbackslash{}")
                        s = s.replace("&", "\\&")
                        s = s.replace("%", "\\%")
                        s = s.replace("$", "\\$")
                        s = s.replace("#", "\\#")
                        s = s.replace("^", "\\textasciicircum{}")
                        s = s.replace("_", "\\_")
                        s = s.replace("{", "\\{")
                        s = s.replace("}", "\\}")
                        s = s.replace("~", "\\textasciitilde{}")
                        return s
                    elif t == "Space":
                        return " "
                    elif t == "SoftBreak":
                        return " "
                    elif t == "Emph":
                        return f"\\emph{{{''.join(to_latex(x) for x in c)}}}"
                    elif t == "Strong":
                        return f"\\textbf{{{''.join(to_latex(x) for x in c)}}}"
                    elif t == "Para":
                        return ''.join(to_latex(x) for x in c)
                    else:
                        return ''.join(to_latex(x) for x in c) if isinstance(c, list) else str(c)
                elif isinstance(elem, list):
                    return ''.join(to_latex(x) for x in elem)
                return str(elem)
            
            text = ''.join(to_latex(x) for x in content)
            return pf.RawBlock("latex", f"\\marginnote{{{text}}}")
    
    # Handle tables - Pandoc will handle them, but we can customize if needed
    # For now, let Pandoc handle tables normally
    
    # Handle raw LaTeX blocks (for TikZ figures)
    if key == "CodeBlock":
        attrs, code = value[0], value[1]
        if attrs and len(attrs) >= 2:
            classes = attrs[1]
            if "latex" in classes:
                # Handle \input commands
                if code.strip().startswith("\\input"):
                    return pf.RawBlock("latex", code)
                else:
                    return pf.RawBlock("latex", code)
    
    # Handle figure references
    if key == "Image":
        attrs, caption, target = value[0], value[1], value[2]
        url, title = target[0], target[1] if len(target) > 1 else ""
        
        # Check if it's a figure that should use TikZ
        if url.endswith(".svg") and "#fig:" in title:
            fig_id = title.split("#fig:")[-1] if "#fig:" in title else ""
            # Return placeholder - will be replaced by actual TikZ
            return pf.RawBlock("latex", f"% Figure {fig_id} - TikZ code should be inserted here")
        
        # Regular image handling
        caption_text = pf.stringify(caption) if caption else ""
        return pf.RawBlock("latex", 
            f"\\begin{{figure}}[h]\n"
            f"\\centering\n"
            f"\\includegraphics{{{url}}}\n"
            f"\\caption{{{caption_text}}}\n"
            f"\\label{{{fig_id}}}\n"
            f"\\end{{figure}}")

if __name__ == "__main__":
    pf.toJSONFilter(latex_filter)
