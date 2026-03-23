"""Theme injection for Multi-Agent Orchestrator demo."""

import streamlit as st

_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Plus+Jakarta+Sans:wght@400;500;600;700&"
    "family=JetBrains+Mono:wght@400;500&display=swap"
)

_CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{_FONTS_URL}" rel="stylesheet">
<style>
    /* Fonts */
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}
    code, pre, .stCode {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* Hide Streamlit chrome */
    #MainMenu, footer, header {{ visibility: hidden; }}
    [data-testid="stToolbar"] {{ display: none; }}

    /* Agent pipeline cards */
    .agent-card {{
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        transition: border-color 0.2s ease;
    }}
    .agent-card.running {{
        border-color: rgba(99, 102, 241, 0.7);
        box-shadow: 0 0 12px rgba(99, 102, 241, 0.25);
    }}
    .agent-card.done {{
        border-color: rgba(16, 185, 129, 0.5);
        background: rgba(16, 185, 129, 0.06);
    }}

    /* Output box */
    .output-box {{
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        font-family: 'JetBrains Mono', monospace;
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
    [data-testid="stSidebar"] {{ background-color: #13132a; }}
</style>
"""


def apply_theme() -> None:
    """Inject fonts, chrome hiding, and agent card CSS."""
    st.markdown(_CSS, unsafe_allow_html=True)
