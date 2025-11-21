# src/ui.py

def base_css() -> str:
    return (
        """
        <style>
        .app-header {font-size: 1.8rem; font-weight: 700; margin: 0 0 0.25rem 0}
        .app-sub    {color: #6b7280; margin-bottom: 1rem}
        .card {background: white; padding: 1rem 1.25rem; border-radius: 12px; border: 1px solid #e5e7eb; box-shadow: 0 2px 10px rgba(0,0,0,0.04); margin-bottom: 0.75rem}
        .pill {display:inline-block; padding: 0.25rem 0.6rem; border-radius: 999px; font-weight:600; font-size:0.9rem}
        .pill-green {background:#ecfdf5; color:#065f46; border:1px solid #a7f3d0}
        .pill-red   {background:#fef2f2; color:#991b1b; border:1px solid #fecaca}
        .bar {height: 10px; border-radius: 999px; background: #e5e7eb; overflow: hidden}
        .bar > span {display:block; height: 100%; background: linear-gradient(90deg,#22c55e,#ef4444);}
        </style>
        """
    )


def header(title: str, subtitle: str) -> str:
    return (
        f"<div class='app-header'>{title}</div>"
        f"<div class='app-sub'>{subtitle}</div>"
    )


def prediction_card_html(label: str, pred: int, prob: float, score: float) -> str:
    pill_class = "pill-red" if pred == 1 else "pill-green"
    return (
        f"<div class='card'><div class='pill {pill_class}'>{label}</div>"
        f"<div style='margin-top:0.75rem'>Confidence</div>"
        f"<div class='bar'><span style='width:{prob*100:.0f}%'></span></div>"
        f"<div style='margin-top:0.5rem; color:#6b7280'>{prob*100:.1f}%</div>"
        f"<div style='margin-top:0.5rem; font-size:0.9rem; color:#6b7280'>Decision score: {score:.3f}</div>"
        f"</div>"
    )
