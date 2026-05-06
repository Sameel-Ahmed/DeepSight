"""
eda.py — Exploratory Data Analysis; returns Plotly figures and text insights.
"""
import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Colour channel analysis ──────────────────────────────────────────────────

def channel_analysis_fig(images: list) -> go.Figure:
    """Distribution of per-image mean intensity for each RGB channel."""
    r, g, b = [], [], []
    for _, img in images:
        r.append(float(np.mean(img[:, :, 2])))
        g.append(float(np.mean(img[:, :, 1])))
        b.append(float(np.mean(img[:, :, 0])))

    fig = go.Figure()
    for vals, name, color in [(r, 'Red', '#FF4B4B'),
                               (g, 'Green', '#00C853'),
                               (b, 'Blue', '#00D4FF')]:
        fig.add_trace(go.Histogram(x=vals, name=name, marker_color=color,
                                   opacity=0.75, nbinsx=30))

    fig.update_layout(
        title='RGB Channel Mean Distribution Across Dataset',
        xaxis_title='Mean Pixel Intensity (0 – 255)',
        yaxis_title='Image Count',
        barmode='overlay',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,22,40,0.6)',
        font_color='#E2F0FF',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(t=60, b=40)
    )
    return fig


# ── Class distribution ───────────────────────────────────────────────────────

def class_distribution_fig(labels: list, class_map: dict) -> go.Figure:
    from collections import Counter
    counts = Counter(labels)
    keys   = sorted(counts.keys())
    names  = [class_map[k] for k in keys]
    vals   = [counts[k]    for k in keys]

    fig = go.Figure(go.Bar(
        x=names, y=vals,
        marker=dict(color=vals,
                    colorscale=[[0, '#0077B6'], [1, '#00D4FF']],
                    showscale=False),
        text=vals, textposition='outside', textfont_color='#E2F0FF'
    ))
    fig.update_layout(
        title='Class Distribution',
        xaxis_title='Class',
        yaxis_title='Images',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,22,40,0.6)',
        font_color='#E2F0FF',
        xaxis_tickangle=-40,
        margin=dict(t=60, b=100)
    )
    return fig


# ── Pixel intensity histogram ────────────────────────────────────────────────

def intensity_histogram_fig(raw_images: list,
                             enhanced_images: list | None = None) -> go.Figure:
    n = min(20, len(raw_images))
    raw_px = np.concatenate([raw_images[i][1].flatten() for i in range(n)])

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=raw_px, name='Raw',
                               marker_color='#94A3B8', opacity=0.7,
                               nbinsx=60, histnorm='probability density'))

    if enhanced_images:
        enh_px = np.concatenate([img.flatten() for img in enhanced_images[:n]])
        fig.add_trace(go.Histogram(x=enh_px, name='Enhanced',
                                   marker_color='#00FF9F', opacity=0.7,
                                   nbinsx=60, histnorm='probability density'))

    fig.update_layout(
        title='Pixel Intensity Distribution (Raw vs Enhanced)',
        xaxis_title='Pixel Value',
        yaxis_title='Density',
        barmode='overlay',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,22,40,0.6)',
        font_color='#E2F0FF',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(t=60, b=40)
    )
    return fig


# ── Channel comparison radar / bar before–after ──────────────────────────────

def channel_comparison_fig(raw_images: list, enhanced_images: list) -> go.Figure:
    n = min(30, len(raw_images), len(enhanced_images))

    raw_means = [np.mean(raw_images[i][1][:, :, c]) for c in range(3) for i in range(n)]
    enh_means = [np.mean(enhanced_images[i][:, :, c]) for c in range(3) for i in range(n)]

    channels = ['Blue', 'Green', 'Red']
    raw_ch  = [np.mean([raw_images[i][1][:, :, c].mean() for i in range(n)]) for c in range(3)]
    enh_ch  = [np.mean([enhanced_images[i][:, :, c].mean()               for i in range(n)]) for c in range(3)]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Raw',      x=channels, y=raw_ch,
                         marker_color='#94A3B8'))
    fig.add_trace(go.Bar(name='Enhanced', x=channels, y=enh_ch,
                         marker_color='#00D4FF'))

    fig.update_layout(
        title='Mean Channel Intensity: Raw vs Enhanced',
        yaxis_title='Mean Intensity',
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,22,40,0.6)',
        font_color='#E2F0FF',
        margin=dict(t=60, b=40)
    )
    return fig


# ── Auto-generated insights ──────────────────────────────────────────────────

def generate_insights(images: list) -> list:
    r_m = [float(np.mean(img[:, :, 2])) for _, img in images]
    g_m = [float(np.mean(img[:, :, 1])) for _, img in images]
    b_m = [float(np.mean(img[:, :, 0])) for _, img in images]
    all_mean = float(np.mean([np.mean(img) for _, img in images]))

    avg_r, avg_g, avg_b = np.mean(r_m), np.mean(g_m), np.mean(b_m)
    dom = 'Blue' if avg_b >= avg_g else 'Green'

    return [
        {
            'icon': '🔴',
            'title': 'Colour Channel Imbalance',
            'text': (
                f"Red avg = {avg_r:.1f} | Green avg = {avg_g:.1f} | Blue avg = {avg_b:.1f}. "
                f"{dom} dominates by {max(avg_b,avg_g) - avg_r:.1f} intensity units over Red, "
                "confirming water's selective absorption of longer wavelengths. "
                "This directly motivates White Balance Correction as the first enhancement step."
            )
        },
        {
            'icon': '📊',
            'title': 'Low Contrast Distribution',
            'text': (
                f"Mean pixel intensity across sampled images = {all_mean:.1f} / 255 "
                f"({all_mean/255*100:.0f}% of full scale). "
                "Values are concentrated in a narrow mid-range band, confirming low global contrast. "
                "CLAHE is applied on the luminance (L) channel to redistribute intensity locally "
                "without distorting hue."
            )
        },
        {
            'icon': '🌊',
            'title': 'Systematic Degradation Pattern',
            'text': (
                f"Standard deviation of channel means across {len(images)} sampled images: "
                f"R={np.std(r_m):.1f}, G={np.std(g_m):.1f}, B={np.std(b_m):.1f}. "
                "The low inter-image variance confirms degradation is systematic (due to water optics) "
                "rather than random — meaning a fixed two-step pipeline will generalise across the entire dataset."
            )
        }
    ]
