"""Theme injection for Multi-Agent Orchestrator demo."""

import streamlit as st

_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Sora:wght@400;600;700&"
    "family=Nunito+Sans:opsz,wght@6..12,400;6..12,500;6..12,600&"
    "family=Fira+Code:wght@400;500&display=swap"
)

_CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{_FONTS_URL}" rel="stylesheet">
<style>
    /* Fonts */
    html, body, [class*="css"] {{
        font-family: 'Nunito Sans', sans-serif;
    }}
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Sora', sans-serif !important;
    }}
    code, pre, .stCode {{
        font-family: 'Fira Code', monospace !important;
    }}

    /* Hide Streamlit chrome */
    #MainMenu, footer, header {{ visibility: hidden; }}
    [data-testid="stToolbar"] {{ display: none; }}

    /* Circuit-board SVG background */
    .stApp {{
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Cpath d='M10 10h10v10H10zM40 10h10v10H40zM70 10h10v10H70zM10 40h10v10H10zM40 40h10v10H40zM70 40h10v10H70zM10 70h10v10H10zM40 70h10v10H40zM70 70h10v10H70z' fill='none' stroke='rgba(245,158,11,0.07)' stroke-width='1'/%3E%3Cpath d='M15 10v-5h25v5M45 10v-5h25v5M15 20v5h10v5h20v-5h10v-5M15 40v-5h25v5M45 40v-5h25v5' fill='none' stroke='rgba(245,158,11,0.05)' stroke-width='0.5'/%3E%3C/svg%3E");
        background-size: 100px 100px;
    }}

    /* Agent pipeline cards */
    .agent-card {{
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 8px;
        border-style: dashed;
        padding: 1rem;
        text-align: center;
        transition: border-color 0.2s ease;
    }}
    .agent-card.running {{
        border-color: rgba(245, 158, 11, 0.7);
        border-style: solid;
        box-shadow: 0 0 12px rgba(245, 158, 11, 0.25);
    }}
    .agent-card.done {{
        border-color: rgba(16, 185, 129, 0.5);
        border-style: solid;
        background: rgba(16, 185, 129, 0.06);
    }}
    .agent-card.planner {{
        border-color: rgba(245, 158, 11, 0.5);
        background: rgba(245, 158, 11, 0.06);
    }}
    .agent-card.parallel {{
        border-color: rgba(167, 139, 250, 0.5);
        background: rgba(167, 139, 250, 0.06);
    }}
    .agent-card.pending {{
        border-color: rgba(100, 116, 139, 0.4);
        background: rgba(100, 116, 139, 0.05);
    }}

    /* Output box */
    .output-box {{
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        font-family: 'Fira Code', monospace;
        font-size: 0.85rem;
        line-height: 1.6;
    }}

    /* Skeleton shimmer */
    @keyframes shimmer {{
        0% {{ background-position: -468px 0; }}
        100% {{ background-position: 468px 0; }}
    }}
    .skeleton {{
        background: linear-gradient(
            to right,
            rgba(255,255,255,0.04) 8%,
            rgba(255,255,255,0.10) 18%,
            rgba(255,255,255,0.04) 33%
        );
        background-size: 800px 104px;
        animation: shimmer 1.4s ease-in-out infinite;
        border-radius: 6px;
        height: 2.5rem;
        margin: 0.25rem 0;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{ background-color: #120E00; }}
</style>
"""


def apply_theme() -> None:
    """Inject fonts, chrome hiding, and agent card CSS."""
    st.markdown(_CSS, unsafe_allow_html=True)
