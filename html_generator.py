import os
import re
import html
from config import REPORTS_DIR


def generate_html_report(markdown_text: str = None, file_id: str = "report", transcript_text: str = None, **kwargs) -> str:
    """
    Generate an interactive, Read.ai-grade HTML dashboard with 4 core tabs:
    [📋 Recap] [📝 Transcript] [🔬 Deep Dive] [🎯 Coaching]
    """
    md_content = markdown_text or ""
    md_content = md_content.replace('\r\n', '\n').replace('\r', '\n')

    # Parse sections from Markdown report
    sections = _parse_markdown_sections(md_content)

    # Convert sections to HTML
    recap_html = _render_recap_section(sections)
    deep_dive_html = _render_deep_dive_section(sections)
    coaching_html = _render_coaching_section(sections)
    transcript_html = _render_transcript_section(transcript_text, file_id)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meeting Intelligence & Assessment — {file_id}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0a0e17;
            --bg-secondary: #111827;
            --bg-card: rgba(17, 24, 39, 0.85);
            --bg-card-hover: rgba(31, 41, 55, 0.95);
            --border: rgba(255, 255, 255, 0.08);
            --border-highlight: rgba(99, 102, 241, 0.35);
            --accent: #6366f1;
            --accent-light: #818cf8;
            --accent-glow: rgba(99, 102, 241, 0.25);
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            --success: #10b981;
            --success-bg: rgba(16, 185, 129, 0.15);
            --warning: #f59e0b;
            --warning-bg: rgba(245, 158, 11, 0.15);
            --danger: #ef4444;
            --danger-bg: rgba(239, 68, 68, 0.15);
            --radius: 14px;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.65;
            padding: 24px 16px;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(16, 185, 129, 0.08) 0%, transparent 40%);
            background-attachment: fixed;
        }}

        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}

        /* Header */
        .top-nav {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 24px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            margin-bottom: 24px;
            backdrop-filter: blur(16px);
        }}

        .nav-brand {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .brand-icon {{
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--accent), #8b5cf6);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }}

        .brand-text h1 {{
            font-size: 1.2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}

        .brand-text p {{
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}

        .btn-export {{
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .btn-export:hover {{
            background: var(--accent);
            border-color: var(--accent);
        }}

        /* Tab Navigation */
        .tabs-header {{
            display: flex;
            gap: 8px;
            background: rgba(17, 24, 39, 0.6);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid var(--border);
            margin-bottom: 28px;
        }}

        .tab-btn {{
            flex: 1;
            padding: 12px 18px;
            border-radius: 8px;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            font-size: 0.92rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }}

        .tab-btn:hover {{
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.04);
        }}

        .tab-btn.active {{
            background: var(--accent);
            color: #ffffff;
            box-shadow: 0 4px 14px var(--accent-glow);
        }}

        /* Tab Content Panes */
        .tab-pane {{
            display: none;
            animation: fadeIn 0.3s ease;
        }}

        .tab-pane.active {{
            display: block;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Card Styles */
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 24px;
            margin-bottom: 24px;
            backdrop-filter: blur(12px);
        }}

        .card-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 18px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }}

        .card-title {{
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-primary);
        }}

        /* Markdown Styling inside Cards */
        p {{
            margin-bottom: 14px;
            color: #d1d5db;
        }}

        ul, ol {{
            margin-bottom: 16px;
            padding-left: 24px;
        }}

        li {{
            margin-bottom: 8px;
            color: #d1d5db;
        }}

        strong {{
            color: #ffffff;
            font-weight: 600;
        }}

        /* Chapters Timeline */
        .chapter-card {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 16px;
            border-left: 4px solid var(--accent);
            transition: transform 0.15s, border-color 0.15s;
        }}

        .chapter-card:hover {{
            transform: translateX(4px);
            border-left-color: var(--accent-light);
            background: rgba(255, 255, 255, 0.04);
        }}

        .chapter-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }}

        .time-badge {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            font-weight: 700;
            background: rgba(99, 102, 241, 0.2);
            color: var(--accent-light);
            padding: 3px 8px;
            border-radius: 6px;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }}

        .chapter-title {{
            font-size: 1.02rem;
            font-weight: 700;
            color: var(--text-primary);
        }}

        .chapter-desc {{
            font-size: 0.9rem;
            color: #9ca3af;
            line-height: 1.6;
        }}

        /* Tables */
        .table-wrapper {{
            overflow-x: auto;
            margin: 16px 0;
            border-radius: 10px;
            border: 1px solid var(--border);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            text-align: left;
        }}

        th {{
            background: rgba(255, 255, 255, 0.04);
            color: var(--text-secondary);
            font-weight: 600;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}

        td {{
            padding: 12px 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            color: #d1d5db;
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}

        /* Transcript Viewer */
        .transcript-controls {{
            display: flex;
            gap: 12px;
            margin-bottom: 20px;
        }}

        .search-input {{
            flex: 1;
            padding: 12px 18px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border);
            border-radius: 10px;
            color: var(--text-primary);
            font-size: 0.9rem;
            font-family: inherit;
            outline: none;
        }}

        .search-input:focus {{
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }}

        .dialogue-stream {{
            display: flex;
            flex-direction: column;
            gap: 12px;
            max-height: 700px;
            overflow-y: auto;
            padding-right: 8px;
        }}

        .dialogue-stream::-webkit-scrollbar {{
            width: 6px;
        }}
        .dialogue-stream::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.15);
            border-radius: 4px;
        }}

        .dialogue-row {{
            display: flex;
            gap: 14px;
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 10px;
            transition: background 0.15s;
        }}

        .dialogue-row:hover {{
            background: rgba(255, 255, 255, 0.04);
        }}

        .dialogue-time {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.76rem;
            color: var(--accent-light);
            white-space: nowrap;
            padding-top: 2px;
        }}

        .dialogue-content {{
            font-size: 0.88rem;
            color: #d1d5db;
            line-height: 1.6;
        }}

        .dialogue-speaker {{
            font-weight: 700;
            color: #ffffff;
            margin-right: 6px;
        }}

        /* Action Items Pills */
        .action-item {{
            display: flex;
            align-items: flex-start;
            gap: 12px;
            padding: 12px 16px;
            background: rgba(99, 102, 241, 0.06);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 10px;
            margin-bottom: 10px;
        }}

        .action-icon {{
            color: var(--accent-light);
            font-size: 1.1rem;
            margin-top: 1px;
        }}

        .action-text {{
            font-size: 0.9rem;
            color: #e5e7eb;
        }}

        @media print {{
            .tabs-header, .btn-export, .transcript-controls {{ display: none !important; }}
            .tab-pane {{ display: block !important; margin-bottom: 40px; page-break-after: always; }}
            body {{ background: #fff !important; color: #000 !important; }}
            .card {{ border: 1px solid #ddd !important; background: #fff !important; color: #000 !important; box-shadow: none !important; }}
            table, th, td {{ color: #000 !important; border-color: #ddd !important; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Top Nav -->
        <div class="top-nav">
            <div class="nav-brand">
                <div class="brand-icon">🎓</div>
                <div class="brand-text">
                    <h1>Meeting & Viva Assessment Intelligence</h1>
                    <p>File ID: {file_id} • AI-Powered Comprehensive Evaluation</p>
                </div>
            </div>
            <button class="btn-export" onclick="window.print()">🖨️ Export to PDF</button>
        </div>

        <!-- 4-Tab Navigation Header -->
        <div class="tabs-header">
            <button class="tab-btn active" data-tab="recap" onclick="switchTab('recap')">📋 Recap</button>
            <button class="tab-btn" data-tab="transcript" onclick="switchTab('transcript')">📝 Transcript</button>
            <button class="tab-btn" data-tab="deepdive" onclick="switchTab('deepdive')">🔬 Deep Dive</button>
            <button class="tab-btn" data-tab="coaching" onclick="switchTab('coaching')">🎯 Coaching</button>
        </div>

        <!-- Tab 1: Recap -->
        <div id="tab-recap" class="tab-pane active">
            {recap_html}
        </div>

        <!-- Tab 2: Transcript -->
        <div id="tab-transcript" class="tab-pane">
            {transcript_html}
        </div>

        <!-- Tab 3: Deep Dive -->
        <div id="tab-deepdive" class="tab-pane">
            {deep_dive_html}
        </div>

        <!-- Tab 4: Coaching -->
        <div id="tab-coaching" class="tab-pane">
            {coaching_html}
        </div>
    </div>

    <script>
        function switchTab(tabName) {{
            document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            
            const targetPane = document.getElementById('tab-' + tabName);
            if (targetPane) targetPane.classList.add('active');
            
            const btn = document.querySelector('.tab-btn[data-tab="' + tabName + '"]');
            if (btn) btn.classList.add('active');
        }}

        function filterTranscript() {{
            const input = document.getElementById('transcript-search');
            if (!input) return;
            const query = input.value.toLowerCase();
            const rows = document.querySelectorAll('.dialogue-row');
            rows.forEach(row => {{
                const text = row.innerText.toLowerCase();
                row.style.display = text.includes(query) ? 'flex' : 'none';
            }});
        }}
    </script>
</body>
</html>
"""

    output_path = os.path.join(REPORTS_DIR, f"{file_id}_report.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"[HTML Report] Generated: {output_path}")
    return output_path


def _parse_markdown_sections(md_text: str) -> dict:
    """Split markdown into named sections by top-level headers."""
    sections = {}
    current_sec = "intro"
    current_lines = []

    for line in md_text.split('\n'):
        if line.startswith('## '):
            if current_lines:
                sections[current_sec] = '\n'.join(current_lines).strip()
            current_sec = line.replace('## ', '').strip().lower()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_sec] = '\n'.join(current_lines).strip()

    return sections


def _markdown_to_html_simple(text: str) -> str:
    """Convert markdown to safe, self-contained HTML.

    All raw text is HTML-escaped first so transcript/LLM content can never
    inject or break the surrounding document structure (e.g. an unclosed
    <title> or <script> swallowing the interactive controls). Converts
    headings, bold/italic/code, tables, bullet and numbered lists, and
    preserves line breaks so dense content (e.g. Q&A lists) stays readable.
    """
    if not text:
        return ""

    # Escape everything first; markdown markers (##, **, `, |) survive escaping.
    text = html.escape(text, quote=False)

    # ---- tables (line-based, emitted as blocks) ----
    lines = text.split('\n')
    blocks = []
    table_rows = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|') and stripped.endswith('|'):
            if not in_table:
                in_table = True
                table_rows = []
            table_rows.append(stripped)
        else:
            if in_table:
                blocks.append(_format_html_table(table_rows))
                in_table = False
                table_rows = []
            blocks.append(line)

    if in_table:
        blocks.append(_format_html_table(table_rows))

    # ---- line-based structural conversion ----
    out = []
    in_list = None  # 'ol' or 'ul'
    for b in blocks:
        stripped = b.strip()
        if not stripped:
            if not in_list:
                out.append('<br>')
            continue
        if stripped in ('---', '***', '___'):
            if in_list:
                out.append('</%s>' % in_list)
                in_list = None
            out.append('<hr>')
            continue
        if stripped.startswith('<div class="table-wrapper">'):
            out.append(b)
            continue
        mh = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if mh:
            if in_list:
                out.append('</%s>' % in_list)
                in_list = None
            lvl = len(mh.group(1))
            out.append(f'<h{lvl}>{mh.group(2)}</h{lvl}>')
            continue
        mnum = re.match(r'^(\d{1,2})\.\s+(.+)$', stripped)
        mbul = re.match(r'^[-*]\s+(.+)$', stripped)
        if mnum:
            if in_list != 'ol':
                if in_list:
                    out.append('</%s>' % in_list)
                out.append('<ol style="margin:0 0 16px;padding-left:1.4rem;">')
                in_list = 'ol'
            out.append(f'<li style="margin:0 0 4px;">{mnum.group(2)}</li>')
        elif mbul:
            if in_list != 'ul':
                if in_list:
                    out.append('</%s>' % in_list)
                out.append('<ul style="margin:0 0 16px;padding-left:1.4rem;">')
                in_list = 'ul'
            out.append(f'<li style="margin:0 0 4px;">{mbul.group(1)}</li>')
        else:
            if in_list:
                out.append(f'<br>{stripped}')
            else:
                out.append(stripped + '<br>')

    if in_list:
        out.append('</%s>' % in_list)

    content = '\n'.join(out)

    # ---- inline formatting across the assembled content ----
    content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
    content = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'<em>\1</em>', content)
    content = re.sub(r'`([^`]+)`', r'<code style="background:rgba(255,255,255,0.08);padding:2px 6px;border-radius:4px;font-family:\'JetBrains Mono\',monospace;font-size:0.85em;">\1</code>', content)

    return content


def _format_html_table(table_rows: list) -> str:
    """Format markdown table rows into styled HTML table."""
    if not table_rows:
        return ""
    html = ['<div class="table-wrapper"><table>']
    is_head = True
    for row in table_rows:
        cells = [c.strip() for c in row.strip('|').split('|')]
        if any('---' in c for c in cells):
            continue
        if is_head:
            html.append('<thead><tr>' + ''.join(f'<th>{c}</th>' for c in cells) + '</tr></thead><tbody>')
            is_head = False
        else:
            html.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
    html.append('</tbody></table></div>')
    return '\n'.join(html)


def _strip_section_header(text: str) -> str:
    """Remove the leading markdown heading from a section body.

    Each card already renders its own title, so a body that starts with its
    own '## Section Name' would otherwise render a duplicate heading.
    """
    return re.sub(r'^\s*#{1,6}\s+.*?(?:\n|$)', '', text, count=1).strip()


def _render_recap_section(sections: dict) -> str:
    """Render the Recap tab content."""
    html = []
    
    # Executive Summary
    for k, v in sections.items():
        if "summary" in k or "executive" in k or "overview" in k:
            html.append(f'<div class="card"><div class="card-header"><span style="font-size:1.4rem;">⚡</span><h2 class="card-title">Executive Summary & Participant Overview</h2></div>{_markdown_to_html_simple(_strip_section_header(v))}</div>')

    # Action Items
    for k, v in sections.items():
        if "action" in k or "task" in k or "step" in k:
            html.append(f'<div class="card"><div class="card-header"><span style="font-size:1.4rem;">⏱️</span><h2 class="card-title">Assigned Action Items & Next Steps</h2></div>{_markdown_to_html_simple(_strip_section_header(v))}</div>')

    # Chapters / Key Discussion Points
    for k, v in sections.items():
        if "chapter" in k or "discussion" in k or "timeline" in k or "proctor" in k or "log" in k:
            html.append(f'<div class="card"><div class="card-header"><span style="font-size:1.4rem;">📑</span><h2 class="card-title">Chronological Key Discussion Points & Chapters</h2></div>{_render_chapters_ui(v)}</div>')

    return '\n'.join(html) if html else '<div class="card"><p>No recap available.</p></div>'


def _render_chapters_ui(chapter_text: str) -> str:
    """Render chapters as timeline cards cleanly."""
    cards = []
    
    # Clean out top-level header if present
    text = re.sub(r'^##\s+.*?\n', '', chapter_text, flags=re.MULTILINE).strip()
    
    # Match chapters pattern: ### `[MM:SS]` — Title \n Description
    pattern = r'###?\s+[`\[]?([\d:–\s]+)[`\]]?\s*[-—–]?\s*([^\n]+)\n([\s\S]*?)(?=(?:###?\s+[`\[]?[\d:–\s]+[`\]]?)|$)'
    matches = re.findall(pattern, text)
    
    if matches:
        for timestamp, title, desc in matches:
            timestamp = timestamp.strip('`[] ')
            title = title.replace('`', '').replace('**', '').replace('🔹', '').strip(' -—–')
            desc_html = _markdown_to_html_simple(desc.strip())
            
            cards.append(f"""
            <div class="chapter-card">
                <div class="chapter-header">
                    <span class="time-badge">{html.escape(timestamp)}</span>
                    <span class="chapter-title">{html.escape(title)}</span>
                </div>
                <div class="chapter-desc">{desc_html}</div>
            </div>
            """)
        return '\n'.join(cards)
    
    return _markdown_to_html_simple(chapter_text)


def _render_deep_dive_section(sections: dict) -> str:
    """Render Deep Dive scorecard, topic matrix, and questions explored."""
    html = []
    for k, v in sections.items():
        if "scorecard" in k or "performance" in k or "rating" in k or "rubric" in k:
            html.append(f'<div class="card"><div class="card-header"><span style="font-size:1.4rem;">📊</span><h2 class="card-title">Performance Scorecard & Benchmarks</h2></div>{_markdown_to_html_simple(_strip_section_header(v))}</div>')
        elif "matrix" in k or "topic" in k or "mastery" in k or "competency" in k:
            html.append(f'<div class="card"><div class="card-header"><span style="font-size:1.4rem;">🎯</span><h2 class="card-title">Topic Mastery Matrix</h2></div>{_markdown_to_html_simple(_strip_section_header(v))}</div>')
        elif "question" in k or "explored" in k or "technical" in k or "verdict" in k:
            html.append(f'<div class="card"><div class="card-header"><span style="font-size:1.4rem;">❓</span><h2 class="card-title">Key Technical Questions Explored</h2></div>{_markdown_to_html_simple(_strip_section_header(v))}</div>')

    return '\n'.join(html) if html else '<div class="card"><p>Deep dive information available in full report.</p></div>'


def _render_coaching_section(sections: dict) -> str:
    """Render Coaching and Feedback section."""
    html = []
    for k, v in sections.items():
        if "coach" in k or "feedback" in k or "recommendation" in k or "audit" in k or "guidance" in k or "note" in k or "observation" in k or "flag" in k:
            html.append(f'<div class="card"><div class="card-header"><span style="font-size:1.4rem;">🎯</span><h2 class="card-title">Coaching & Professional Feedback</h2></div>{_markdown_to_html_simple(_strip_section_header(v))}</div>')

    return '\n'.join(html) if html else '<div class="card"><p>Coaching guidance summarized in main assessment.</p></div>'


def _render_transcript_section(transcript_text: str, file_id: str) -> str:
    """Render the full interactive, searchable transcript tab."""
    if not transcript_text:
        return '<div class="card"><p>No transcript text available.</p></div>'

    rows = []
    lines = [l.strip() for l in transcript_text.split('\n') if l.strip() and not l.startswith('#') and not l.startswith('===')]

    for line in lines:
        t_match = re.match(r'\[?(\d{1,2}:\d{2})\]?\s*(.*)', line)
        if t_match:
            timestamp = t_match.group(1)
            content = t_match.group(2)
        else:
            timestamp = "00:00"
            content = line

        rows.append(f"""
        <div class="dialogue-row">
            <span class="dialogue-time">{html.escape(timestamp)}</span>
            <div class="dialogue-content">{html.escape(content)}</div>
        </div>
        """)

    return f"""
    <div class="card">
        <div class="transcript-controls">
            <input type="text" id="transcript-search" class="search-input" placeholder="🔍 Search transcript keywords (e.g., GIL, Deadlock, Docker, Samar, Nithin)..." onkeyup="filterTranscript()">
        </div>
        <div class="dialogue-stream">
            {''.join(rows)}
        </div>
    </div>
    """
