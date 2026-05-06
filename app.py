"""
app.py — DeepSight: Underwater Image Enhancement & Classification Pipeline
"""
import streamlit as st
import cv2
import numpy as np
import os, sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.ingestion     import load_dataset
from pipeline.preprocessing import load_sample, get_dataset_stats
from pipeline.eda           import (channel_analysis_fig, class_distribution_fig,
                                    intensity_histogram_fig, channel_comparison_fig,
                                    generate_insights)
from pipeline.enhancement   import enhance_image, compute_psnr, enhance_batch
from pipeline.features      import build_feature_matrix, feature_names, FEATURE_DIM
from pipeline.detection     import detect_salient_object, draw_bounding_box
from pipeline.model         import (train_model, save_model, load_model, predict_image,
                                    confusion_matrix_fig, feature_importance_fig,
                                    per_class_metrics_fig)
from ultralytics            import YOLO

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepSight Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #042F2E; }
::-webkit-scrollbar-thumb { background: #115E59; border-radius: 3px; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #022C22 0%, #042F2E 100%);
    border-right: 1px solid rgba(20,184,166,0.2);
}
[data-testid="stSidebar"] * { color: #F0FDFA !important; }
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 15% 10%, #064E3B 0%, #042F2E 55%, #022C22 100%);
}
[data-testid="stHeader"] { background: transparent; }

.hero {
    background: linear-gradient(135deg, #0F766E 0%, #115E59 50%, #134E4A 100%);
    border: 1px solid rgba(20,184,166,0.25);
    border-radius: 16px; padding: 2.4rem 2rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -60%; right: -20%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(250,204,21,0.07) 0%, transparent 65%);
    pointer-events: none;
}
.hero h1 {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(135deg, #2DD4BF, #FACC15);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}
.hero p { color: #99F6E4; font-size: 0.97rem; margin: 0; }

.step-header {
    display: flex; align-items: center; gap: 12px;
    background: linear-gradient(90deg, rgba(20,184,166,0.15) 0%, transparent 100%);
    border-left: 3px solid #0D9488;
    padding: 0.75rem 1.2rem; border-radius: 0 8px 8px 0; margin-bottom: 1.2rem;
}
.step-num {
    background: #0F766E; color: white; font-weight: 700; font-size: 0.72rem;
    padding: 0.22rem 0.6rem; border-radius: 20px; white-space: nowrap;
}
.step-title { color: #F0FDFA; font-size: 1.3rem; font-weight: 600; margin: 0; }

.card {
    background: rgba(6,78,59,0.85); border: 1px solid rgba(20,184,166,0.15);
    border-radius: 12px; padding: 1.4rem; margin-bottom: 1rem;
}
.card-title {
    color: #2DD4BF; font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.5rem;
}

.metric-row { display: flex; gap: 0.8rem; margin-bottom: 1rem; flex-wrap: wrap; }
.metric-box {
    flex: 1; min-width: 100px;
    background: linear-gradient(135deg, rgba(20,184,166,0.12), rgba(250,204,21,0.05));
    border: 1px solid rgba(20,184,166,0.22); border-radius: 12px;
    padding: 1.1rem; text-align: center;
}
.metric-box .val {
    font-size: 1.9rem; font-weight: 700; color: #2DD4BF;
    font-family: 'JetBrains Mono', monospace; line-height: 1;
}
.metric-box .lbl {
    font-size: 0.72rem; color: #5EEAD4;
    text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.3rem;
}

.insight {
    background: linear-gradient(135deg, rgba(20,184,166,0.07), rgba(250,204,21,0.04));
    border: 1px solid rgba(20,184,166,0.2); border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 0.7rem;
}
.insight-title { color: #2DD4BF; font-weight: 600; font-size: 0.93rem; }
.insight-text  { color: #99F6E4; font-size: 0.86rem; margin-top: 0.3rem; line-height: 1.65; }

.badge {
    display: inline-block; padding: 0.2rem 0.7rem; border-radius: 20px;
    font-size: 0.73rem; font-weight: 600; letter-spacing: 0.04em;
}
.badge-uieb  { background: rgba(20,184,166,0.15); color: #2DD4BF; border: 1px solid #0F766E; }
.badge-class { background: rgba(250,204,21,0.12); color: #FACC15; border: 1px solid #CA8A04; }
.badge-flat  { background: rgba(16,185,129,0.12); color: #10B981; border: 1px solid #059669; }

.pred-bar-wrap { margin-bottom: 0.55rem; }
.pred-label { color: #F0FDFA; font-size: 0.87rem; margin-bottom: 0.15rem; }
.pred-bar { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #0F766E, #FACC15); }
.pred-pct { color: #2DD4BF; font-size: 0.78rem; text-align: right; margin-top: 0.1rem; }

.pipe-box {
    text-align: center;
    background: rgba(20,184,166,0.12);
    border: 1px solid rgba(20,184,166,0.25);
    border-radius: 10px; padding: 0.7rem 0.4rem;
}
.pipe-icon { font-size: 1.4rem; line-height: 1.3; }
.pipe-lbl  { font-size: 0.72rem; color: #2DD4BF; font-weight: 600; }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0F766E, #0D9488);
    color: white; border: none; border-radius: 8px;
    font-family: 'Inter', sans-serif; font-weight: 600;
    padding: 0.5rem 1.5rem; transition: opacity 0.2s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.82; }
div[data-testid="stTextInput"] input {
    background: rgba(6,78,59,0.95); border: 1px solid rgba(20,184,166,0.3);
    color: #F0FDFA; border-radius: 8px;
}
[data-testid="stMetricValue"] { color: #2DD4BF !important; }
[data-testid="stMetricLabel"] { color: #5EEAD4 !important; }
div[data-testid="stTabs"] [data-baseweb="tab"] { color: #5EEAD4 !important; }
div[data-testid="stTabs"] [aria-selected="true"] { color: #2DD4BF !important; }
hr { border-color: rgba(20,184,166,0.15) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

PLOTLY_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(19,11,36,0.7)',
    font_color='#F3F0FF'
)

def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def clean_path(raw: str) -> str:
    return str(Path(raw.strip().strip('"').strip("'")))

def step_header(num, icon, title):
    st.markdown(f"""
    <div class="step-header">
        <span class="step-num">STEP {num}</span>
        <span style="font-size:1.35rem">{icon}</span>
        <p class="step-title">{title}</p>
    </div>""", unsafe_allow_html=True)

def metric_boxes(metrics):
    html = '<div class="metric-row">'
    for val, lbl in metrics:
        html += f'<div class="metric-box"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def insight_card(icon, title, text):
    st.markdown(f"""
    <div class="insight">
        <div class="insight-title">{icon}&nbsp;{title}</div>
        <div class="insight-text">{text}</div>
    </div>""", unsafe_allow_html=True)

def show_image_grid(images, captions=None, n_cols=4, max_n=12):
    items = images[:max_n]
    cols  = st.columns(n_cols)
    for i, item in enumerate(items):
        img = item[1] if isinstance(item, tuple) else item
        cap = (captions[i] if captions else
               (Path(item[0]).name if isinstance(item, tuple) else f"img_{i}"))
        with cols[i % n_cols]:
            st.image(bgr_to_rgb(img), caption=cap, use_container_width=True)

def mark_done(key):
    if 'done' not in st.session_state:
        st.session_state['done'] = set()
    st.session_state['done'].add(key)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

PAGES = [
    ("🔬", "Home"),
    ("📁", "1 · Data Ingestion"),
    ("🔧", "2 · Preprocessing"),
    ("📊", "3 · EDA"),
    ("✨", "4 · Enhancement"),
    ("🧬", "5 · Feature Extraction"),
    ("🤖", "6 · Model Training"),
    ("🎯", "7 · Live Demo"),
]

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.8rem 0 1.5rem 0;">
        <div style="font-size:2.2rem;">🔬</div>
        <div style="font-size:1.15rem;font-weight:700;
                    background:linear-gradient(135deg,#2DD4BF,#F59E0B);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            DeepSight
        </div>
        <div style="font-size:0.72rem;color:#5EEAD4;letter-spacing:0.1em;">
            UNDERWATER PIPELINE
        </div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("nav", [p[1] for p in PAGES],
                    format_func=lambda x: f"{[p[0] for p in PAGES if p[1]==x][0]}  {x}",
                    label_visibility="collapsed")

    st.markdown('<hr/>', unsafe_allow_html=True)

    done      = st.session_state.get('done', set())
    s_keys    = ['ingestion','preprocessing','eda','enhancement','features','training']
    s_labels  = ['Ingestion','Preprocessing','EDA','Enhancement','Features','Training']
    for lbl, key in zip(s_labels, s_keys):
        tick = '✅' if key in done else '⬜'
        st.markdown(f'<div style="font-size:0.82rem;color:#5EEAD4;margin-bottom:0.2rem;">'
                    f'{tick} {lbl}</div>', unsafe_allow_html=True)

    completed = len([k for k in s_keys if k in done])
    st.progress(completed / len(s_keys))
    st.markdown(f'<div style="font-size:0.73rem;color:#134E4A;text-align:right;">'
                f'{completed}/{len(s_keys)} steps done</div>', unsafe_allow_html=True)

    st.markdown('<hr/>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;color:#3D2E60;text-align:center;">'
                'Intro to Data Science · Spring 2026<br>'
                'Sameel Ahmed &amp; Musa Salman</div>', unsafe_allow_html=True)


# ─── HOME ─────────────────────────────────────────────────────────────────────
if page == "Home":
    st.markdown("""
    <div class="hero">
        <h1>🔬 DeepSight Pipeline</h1>
        <p>A complete data science pipeline for underwater image enhancement
           and marine species classification.</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="card">
            <div class="card-title">📦 Datasets</div>
            <div style="color:#2DD4BF;font-size:0.88rem;font-weight:600;">UIEB Dataset</div>
            <div style="color:#5EEAD4;font-size:0.81rem;">950 underwater images with paired references for PSNR evaluation</div>
            <br>
            <div style="color:#F59E0B;font-size:0.88rem;font-weight:600;">Large Scale Fish</div>
            <div style="color:#5EEAD4;font-size:0.81rem;">9-class fish dataset for species classification</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card">
            <div class="card-title">🔬 Techniques</div>
            <div style="color:#5EEAD4;font-size:0.84rem;line-height:2.0;">
                • White Balance (Grey World)<br>
                • CLAHE (LAB colour space)<br>
                • RGB + Histogram features<br>
                • LBP texture descriptors<br>
                • Random Forest (100 trees)
            </div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card">
            <div class="card-title">📈 Evaluation</div>
            <div style="color:#5EEAD4;font-size:0.84rem;line-height:2.0;">
                • PSNR — image quality<br>
                • Accuracy (%)<br>
                • Weighted F1-Score<br>
                • Per-class Precision/Recall<br>
                • Confusion Matrix
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Pipeline diagram — using st.columns (no raw HTML loop)
    st.markdown('<div style="text-align:center;color:#5EEAD4;font-size:0.76rem;'
                'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem;">'
                'Pipeline Overview</div>', unsafe_allow_html=True)

    steps = [("📁","Ingest"),("🔧","Preprocess"),("📊","EDA"),
             ("✨","Enhance"),("🧬","Features"),("🤖","Train"),("🎯","Demo")]

    widths = []
    for i in range(len(steps)):
        widths.append(2)
        if i < len(steps) - 1:
            widths.append(0.35)

    cols = st.columns(widths)
    ci = 0
    for i, (ico, lbl) in enumerate(steps):
        with cols[ci]:
            st.markdown(f'<div class="pipe-box"><div class="pipe-icon">{ico}</div>'
                        f'<div class="pipe-lbl">{lbl}</div></div>',
                        unsafe_allow_html=True)
        ci += 1
        if i < len(steps) - 1:
            with cols[ci]:
                st.markdown('<div style="text-align:center;color:#115E59;'
                            'font-size:1.1rem;padding-top:0.85rem;">→</div>',
                            unsafe_allow_html=True)
            ci += 1

    st.markdown('<br><div style="text-align:center;color:#134E4A;font-size:0.85rem;">'
                '👈 Start with <b style="color:#2DD4BF;">1 · Data Ingestion</b> in the sidebar</div>',
                unsafe_allow_html=True)


# ─── STEP 1: INGESTION ────────────────────────────────────────────────────────
elif page == "1 · Data Ingestion":
    step_header("1", "📁", "Data Ingestion")

    st.markdown("""<div class="card">
        <div class="card-title">Supported Dataset Structures</div>
        <div style="display:flex;gap:1.2rem;flex-wrap:wrap;margin-top:0.5rem;">
            <div><span class="badge badge-uieb">UIEB</span>
                 <span style="color:#5EEAD4;font-size:0.81rem;margin-left:6px;">
                 Folder with raw/ + reference/ subfolders</span></div>
            <div><span class="badge badge-class">Classification</span>
                 <span style="color:#5EEAD4;font-size:0.81rem;margin-left:6px;">
                 One subfolder per class</span></div>
            <div><span class="badge badge-flat">Flat</span>
                 <span style="color:#5EEAD4;font-size:0.81rem;margin-left:6px;">
                 Images directly in one folder</span></div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="card">
        <div class="card-title">How to get your path on Windows</div>
        <div style="color:#5EEAD4;font-size:0.84rem;line-height:1.9;">
            Open the dataset folder in File Explorer &rarr; click the address bar at the top
            &rarr; Ctrl+C &rarr; paste below.<br>
            <b style="color:#2DD4BF;">Fish Dataset example:</b>&nbsp;
            <code style="color:#F59E0B;background:rgba(245,158,11,0.08);
            padding:0.1rem 0.4rem;border-radius:4px;">C:/Users/Sameel/Downloads/Fish_Dataset</code><br>
            <b style="color:#2DD4BF;">UIEB example:</b>&nbsp;
            <code style="color:#F59E0B;background:rgba(245,158,11,0.08);
            padding:0.1rem 0.4rem;border-radius:4px;">C:/Users/Sameel/Downloads/uieb</code>
        </div>
    </div>""", unsafe_allow_html=True)

    c_path, c_btn = st.columns([3, 1])
    with c_path:
        path_input = st.text_input(
            "📂 Paste dataset folder path here",
            value=st.session_state.get('dataset_path', ''),
            placeholder="C:/Users/YourName/Downloads/Fish_Dataset"
        )
    with c_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Auto-Detect ./data"):
            found = False
            for p in ['./data/Fish_Dataset', './data/uieb', './data']:
                abs_p = os.path.abspath(p)
                if os.path.exists(abs_p) and os.path.isdir(abs_p):
                    st.session_state['dataset_path'] = abs_p
                    found = True
                    st.rerun()
            if not found:
                st.error("Not found.")

    if st.button("🔍 Load Dataset"):
        if not path_input.strip():
            st.error("Please enter a path.")
        else:
            cleaned = clean_path(path_input)
            with st.spinner("Scanning dataset..."):
                try:
                    ds = load_dataset(cleaned)
                    st.session_state['dataset']      = ds
                    st.session_state['dataset_path'] = cleaned
                    mark_done('ingestion')
                    st.success(f"✅ Loaded {ds['total']} images successfully!")
                except Exception as e:
                    st.error(f"❌ {e}")
                    st.markdown("""<div class="card" style="border-color:rgba(239,68,68,0.3);">
                        <div style="color:#F87171;font-size:0.84rem;">
                        <b>Common fixes:</b><br>
                        • For Fish Dataset: point to the folder that <b>contains</b> the class
                          subfolders (e.g. Fish_Dataset/), not a class subfolder itself.<br>
                        • For UIEB: point to the folder that <b>contains</b> raw/ and reference/.<br>
                        • Remove any trailing backslash from the path.
                        </div>
                    </div>""", unsafe_allow_html=True)

    if 'dataset' in st.session_state:
        ds = st.session_state['dataset']
        bmap = {'uieb':('uieb','UIEB Mode'),'classification':('class','Classification Mode'),
                'flat':('flat','Flat Mode')}
        bk, bl = bmap.get(ds['mode'], ('flat', ds['mode']))
        st.markdown(f'<br><span class="badge badge-{bk}">{bl}</span><br><br>',
                    unsafe_allow_html=True)

        metric_boxes([
            (ds['total'], "Total Images"),
            (len(ds['references']) if ds['references'] else "—", "References"),
            (len(ds['class_names']) if ds['class_names'] else "—", "Classes"),
        ])

        if ds['class_names']:
            st.markdown("**Classes detected:**")
            gcols = st.columns(min(6, len(ds['class_names'])))
            for i, cls in enumerate(ds['class_names'][:30]):
                with gcols[i % len(gcols)]:
                    st.markdown(f'<div style="background:rgba(13,148,136,0.1);'
                                f'border-radius:6px;padding:0.25rem 0.4rem;'
                                f'font-size:0.77rem;color:#2DD4BF;margin-bottom:0.3rem;">'
                                f'{cls}</div>', unsafe_allow_html=True)


# ─── STEP 2: PREPROCESSING ────────────────────────────────────────────────────
elif page == "2 · Preprocessing":
    step_header("2", "🔧", "Preprocessing")

    if 'dataset' not in st.session_state:
        st.warning("⚠️ Complete Step 1 first.")
        st.stop()

    ds = st.session_state['dataset']

    st.markdown("""<div class="card">
        <div class="card-title">Operations Applied</div>
        <div style="color:#5EEAD4;font-size:0.86rem;line-height:2.1;">
            <b style="color:#2DD4BF;">①</b> Resize all images to 256×256 px (INTER_AREA)<br>
            <b style="color:#2DD4BF;">②</b> Validate each file — skip corrupted samples<br>
            <b style="color:#2DD4BF;">③</b> Probe resolution range and health stats<br>
            <b style="color:#2DD4BF;">④</b> BGR colour space preserved (OpenCV default)
        </div>
    </div>""", unsafe_allow_html=True)

    if st.button("⚙️ Run Preprocessing"):
        with st.spinner("Processing..."):
            stats  = get_dataset_stats(ds['images'])
            sample, failed = load_sample(ds['images'], max_n=48)
            st.session_state['pp_sample'] = sample
            st.session_state['pp_stats']  = stats
            st.session_state['pp_failed'] = failed
            mark_done('preprocessing')

    if 'pp_stats' in st.session_state:
        s = st.session_state['pp_stats']
        metric_boxes([
            (s.get('total',0),     "Total"),
            (s.get('valid',0),     "Valid"),
            (s.get('invalid',0),   "Corrupted"),
            (s.get('avg_res','—'), "Avg Resolution"),
        ])
        st.markdown("**Sample Images (resized to 256×256)**")
        show_image_grid(st.session_state['pp_sample'], n_cols=6, max_n=12)
        if st.session_state['pp_failed']:
            with st.expander(f"⚠️ {len(st.session_state['pp_failed'])} unreadable files"):
                for f in st.session_state['pp_failed'][:20]:
                    st.code(f, language=None)


# ─── STEP 3: EDA ──────────────────────────────────────────────────────────────
elif page == "3 · EDA":
    step_header("3", "📊", "Exploratory Data Analysis")

    if 'pp_sample' not in st.session_state:
        st.warning("⚠️ Complete Step 2 first.")
        st.stop()

    sample = st.session_state['pp_sample']
    ds     = st.session_state['dataset']

    if st.button("📊 Run EDA"):
        with st.spinner("Computing..."):
            st.session_state['insights'] = generate_insights(sample)
            mark_done('eda')

    t1, t2, t3, t4 = st.tabs(
        ["🔴 Channel Analysis", "📐 Class Distribution", "💡 Insights", "🖼️ Sample Grid"])

    with t1:
        st.plotly_chart(channel_analysis_fig(sample), use_container_width=True)
        enh_imgs = [r['enhanced'] for r in st.session_state.get('enh_results', [])[:20]]
        st.plotly_chart(intensity_histogram_fig(sample, enh_imgs or None),
                        use_container_width=True)

    with t2:
        if ds['labels'] and ds['class_map']:
            st.plotly_chart(class_distribution_fig(ds['labels'], ds['class_map']),
                            use_container_width=True)
        else:
            st.info("Class distribution requires a Classification-mode dataset.")

    with t3:
        if 'insights' in st.session_state:
            for ins in st.session_state['insights']:
                insight_card(ins['icon'], ins['title'], ins['text'])
        else:
            st.info("Click **Run EDA** to generate insights.")

    with t4:
        show_image_grid(sample, n_cols=5, max_n=20)


# ─── STEP 4: ENHANCEMENT ──────────────────────────────────────────────────────
elif page == "4 · Enhancement":
    step_header("4", "✨", "Image Enhancement")

    if 'dataset' not in st.session_state:
        st.warning("⚠️ Complete Step 1 first.")
        st.stop()

    ds = st.session_state['dataset']

    st.markdown("""<div class="card">
        <div class="card-title">Five-Stage High-Quality Enhancement Pipeline</div>
        <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:0.5rem;">
            <div style="flex:1;min-width:200px;">
                <div style="color:#2DD4BF;font-weight:600;font-size:0.88rem;">Stage 1-3 — Color & Light</div>
                <div style="color:#5EEAD4;font-size:0.82rem;margin-top:0.3rem;line-height:1.65;">
                    Red Channel Compensation, LAB White Balance, and Gamma Correction to completely fix underwater attenuation and lighting.
                </div>
            </div>
            <div style="flex:1;min-width:200px;">
                <div style="color:#F59E0B;font-weight:600;font-size:0.88rem;">Stage 4-5 — Contrast & Detail</div>
                <div style="color:#5EEAD4;font-size:0.82rem;margin-top:0.3rem;line-height:1.65;">
                    CLAHE (Contrast Limited Adaptive Histogram Equalization) and Unsharp Masking to restore micro-details and scales.
                </div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    max_n   = st.slider("Images to enhance", 1, ds['total'], min(100, ds['total']))
    out_dir = os.path.join(os.path.dirname(__file__), 'data', 'enhanced')

    if st.button("✨ Run Enhancement"):
        bar  = st.progress(0)
        info = st.empty()

        def cb(p):
            bar.progress(p)
            info.markdown(f'<span style="color:#2DD4BF;">Processing {int(p*100)}%…</span>',
                          unsafe_allow_html=True)

        refs = ds['references'][:max_n] if ds['references'] else None
        results = enhance_batch(ds['images'][:max_n], out_dir, refs, progress_cb=cb)
        st.session_state['enh_results'] = results
        mark_done('enhancement')
        info.markdown('<span style="color:#10B981;">✅ Done!</span>', unsafe_allow_html=True)

    if 'enh_results' in st.session_state:
        results   = st.session_state['enh_results']
        psnr_vals = [r['psnr'] for r in results if r['psnr'] is not None]

        if psnr_vals:
            metric_boxes([
                (len(results), "Enhanced"),
                (f"{np.mean(psnr_vals):.2f} dB", "Avg PSNR"),
                (f"{max(psnr_vals):.2f} dB", "Best PSNR"),
                (f"{min(psnr_vals):.2f} dB", "Min PSNR"),
            ])
            fig = go.Figure(go.Histogram(x=psnr_vals, nbinsx=30,
                                         marker_color='#0D9488', opacity=0.85))
            fig.add_vline(x=np.mean(psnr_vals), line_dash='dash', line_color='#F59E0B',
                          annotation_text=f"Mean {np.mean(psnr_vals):.2f} dB",
                          annotation_font_color='#F59E0B')
            fig.update_layout(title='PSNR Distribution', xaxis_title='PSNR (dB)',
                              yaxis_title='Count', **PLOTLY_BASE)
            st.plotly_chart(fig, use_container_width=True)
        else:
            metric_boxes([(len(results),"Enhanced"),("N/A","PSNR (no references)")])

        st.markdown("**Before / After Comparison**")
        for item in results[:6]:
            c1, c2 = st.columns(2)
            with c1:
                st.image(bgr_to_rgb(item['raw']),
                         caption=f"Raw — {Path(item['path']).name}",
                         use_container_width=True)
            with c2:
                cap = (f"Enhanced — PSNR: {item['psnr']:.2f} dB"
                       if item['psnr'] else "Enhanced")
                st.image(bgr_to_rgb(item['enhanced']), caption=cap,
                         use_container_width=True)

        if 'pp_sample' in st.session_state:
            fig2 = channel_comparison_fig(
                st.session_state['pp_sample'],
                [r['enhanced'] for r in results[:20]])
            st.plotly_chart(fig2, use_container_width=True)


# ─── STEP 5: FEATURE EXTRACTION ───────────────────────────────────────────────
elif page == "5 · Feature Extraction":
    step_header("5", "🧬", "Feature Extraction")

    if 'dataset' not in st.session_state:
        st.warning("⚠️ Complete Step 1 first.")
        st.stop()

    ds = st.session_state['dataset']
    if ds['mode'] != 'classification':
        st.info("Load the **Fish Dataset** (class subfolders) to use this step.")
        st.stop()

    st.markdown("""<div class="card">
        <div class="card-title">End-to-End Extraction on Salient Objects</div>
        <div style="color:#5EEAD4;font-size:0.85rem;line-height:2.0;">
            Features are extracted from the <b>enhanced and perfectly masked</b> object region.<br>
            <b style="color:#2DD4BF;">6 features</b> — Mean &amp; Std per RGB channel &nbsp;|&nbsp;
            <b style="color:#2DD4BF;">48 features</b> — Colour histograms (16 bins × 3 ch) &nbsp;|&nbsp;
            <b style="color:#F59E0B;">16 features</b> — LBP texture histogram &nbsp;|&nbsp;
            <b style="color:#10B981;">324 features</b> — HOG Structural Shape Features
        </div>
    </div>""", unsafe_allow_html=True)

    max_n = st.slider("Max images to process", 1, ds['total'], min(2000, ds['total']))

    if st.button("🧬 Extract Features"):
        bar  = st.progress(0)
        info = st.empty()

        def cb(p):
            bar.progress(p)
            info.markdown(f'<span style="color:#2DD4BF;">Extracting {int(p*100)}%…</span>',
                          unsafe_allow_html=True)

        X, y = build_feature_matrix(ds['images'][:max_n], ds['labels'][:max_n],
                                    progress_cb=cb)
        st.session_state['X'] = X
        st.session_state['y'] = y
        st.session_state['feat_class_names'] = ds['class_names']
        mark_done('features')
        info.markdown('<span style="color:#10B981;">✅ Done!</span>', unsafe_allow_html=True)

    if 'X' in st.session_state:
        X = st.session_state['X']
        y = st.session_state['y']
        metric_boxes([
            (X.shape[0], "Images"),
            (X.shape[1], "Features"),
            (len(np.unique(y)), "Classes"),
            (f"{X.nbytes/1e6:.1f} MB", "Matrix Size"),
        ])
        fn = feature_names()
        fig = go.Figure()
        for idx, (name, color) in enumerate([('B_mean','#00B4D8'),
                                              ('G_mean','#10B981'),
                                              ('R_mean','#EF4444')]):
            fidx = fn.index(name)
            fig.add_trace(go.Box(y=X[:, fidx], name=name, marker_color=color))
        fig.update_layout(title='RGB Channel Mean Distribution (Feature Matrix)', **PLOTLY_BASE)
        st.plotly_chart(fig, use_container_width=True)


# ─── STEP 6: MODEL TRAINING ───────────────────────────────────────────────────
elif page == "6 · Model Training":
    step_header("6", "🤖", "Model Training & Evaluation")

    if 'X' not in st.session_state:
        st.warning("⚠️ Complete Step 5 first.")
        st.stop()

    X           = st.session_state['X']
    y           = st.session_state['y']
    class_names = st.session_state['feat_class_names']
    model_path  = os.path.join(os.path.dirname(__file__), 'model.pkl')

    st.markdown("""<div class="card">
        <div class="card-title">Model Hyperparameters</div>
        <div style="color:#5EEAD4;font-size:0.86rem;line-height:1.9;">
            Configure the classifier before training.
        </div>
    </div>""", unsafe_allow_html=True)

    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        model_type = st.selectbox("Model Type", ["Random Forest", "SVM"])
    with c_m2:
        if model_type == "Random Forest":
            n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
        else:
            svm_c = st.number_input("C (Regularization)", 0.1, 100.0, 1.0)
    with c_m3:
        if model_type == "Random Forest":
            max_depth_sel = st.selectbox("Max Depth", ["None", "5", "10", "20", "50"])
            max_depth = None if max_depth_sel == "None" else int(max_depth_sel)
        else:
            svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
            
    with st.expander("🔍 See a Raw Feature Vector (Explainability)"):
        st.markdown("This is exactly what the model sees for **one** image after feature extraction:")
        sample_x = X[0]
        fig_feat = go.Figure(go.Bar(y=sample_x, x=feature_names(), marker_color='#2DD4BF'))
        fig_feat.update_layout(title="70-Dimensional Feature Vector (Image 0)", height=300, margin=dict(t=30, b=0), **PLOTLY_BASE)
        st.plotly_chart(fig_feat, use_container_width=True)

    if st.button("🚀 Train Model"):
        bar  = st.progress(0)
        info = st.empty()

        def cb(p):
            bar.progress(p)
            info.markdown(f'<span style="color:#2DD4BF;">Training {int(p*100)}%…</span>',
                          unsafe_allow_html=True)

        if model_type == "Random Forest":
            res = train_model(X, y, class_names, model_type=model_type, n_estimators=n_estimators, max_depth=max_depth, progress_cb=cb)
        else:
            res = train_model(X, y, class_names, model_type=model_type, svm_c=svm_c, svm_kernel=svm_kernel, progress_cb=cb)
            
        save_model(res, model_path)
        st.session_state['model_results'] = res
        mark_done('training')
        info.markdown('<span style="color:#10B981;">✅ Model saved to model.pkl</span>',
                      unsafe_allow_html=True)

    if 'model_results' in st.session_state:
        res = st.session_state['model_results']
        metric_boxes([
            (f"{res['accuracy']:.1f}%", "Accuracy"),
            (f"{res['f1']:.3f}", "Weighted F1"),
            (res['train_size'], "Train Samples"),
            (res['test_size'],  "Test Samples"),
        ])

        t1, t2, t3 = st.tabs(
            ["🔢 Confusion Matrix", "📊 Per-Class Metrics", "🌲 Feature Importance"])
        with t1:
            st.plotly_chart(confusion_matrix_fig(res['cm'], res['class_names']),
                            use_container_width=True)
        with t2:
            st.plotly_chart(per_class_metrics_fig(res['report'], res['class_names']),
                            use_container_width=True)
            df = pd.DataFrame({c: res['report'][c]
                               for c in res['class_names'] if c in res['report']}).T
            st.dataframe(df[['precision','recall','f1-score','support']].round(3),
                         use_container_width=True)
        with t3:
            st.plotly_chart(feature_importance_fig(res['model'], feature_names()),
                            use_container_width=True)


# ─── STEP 7: LIVE DEMO ────────────────────────────────────────────────────────
elif page == "7 · Live Demo":
    step_header("7", "🎯", "Live Demo")

    model_path  = os.path.join(os.path.dirname(__file__), 'model.pkl')
    model_ready = os.path.exists(model_path)

    if not model_ready:
        st.warning("No `model.pkl` found — complete Step 6 to enable classification. "
                   "Enhancement still works.")

    uploaded_files = st.file_uploader("Upload underwater images (Batch supported)",
                                type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if uploaded_files:
        if len(uploaded_files) > 1:
            st.info(f"Batch Processing {len(uploaded_files)} images...")
            results_data = []
            bar = st.progress(0)
            for i, uf in enumerate(uploaded_files):
                raw = cv2.imdecode(np.frombuffer(uf.read(), np.uint8), cv2.IMREAD_COLOR)
                if raw is None: continue
                raw_r    = cv2.resize(raw, (256, 256), interpolation=cv2.INTER_AREA)
                enhanced = enhance_image(raw_r)
                if model_ready:
                    bbox, cropped = detect_salient_object(enhanced)
                    md              = load_model(model_path)
                    lbl, conf, _ = predict_image(cropped, md)
                    results_data.append({"Filename": uf.name, "Prediction": lbl, "Confidence (%)": conf})
                bar.progress((i + 1) / len(uploaded_files))
            
            if results_data:
                df_res = pd.DataFrame(results_data)
                st.dataframe(df_res, use_container_width=True)
                csv = df_res.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results as CSV", csv, "predictions.csv", "text/csv")
        else:
            uploaded = uploaded_files[0]
            raw = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            st.error("Could not decode image.")
            st.stop()

        raw_r    = cv2.resize(raw, (256, 256), interpolation=cv2.INTER_AREA)
        enhanced = enhance_image(raw_r)
        psnr_val = compute_psnr(raw_r, enhanced)

        st.markdown("#### Enhancement Result")
        c1, c2 = st.columns(2)
        with c1:
            st.image(bgr_to_rgb(raw_r), caption="Original", use_container_width=True)
        with c2:
            st.image(bgr_to_rgb(enhanced), caption="Enhanced", use_container_width=True)
        st.metric("PSNR (raw vs enhanced)", f"{psnr_val:.2f} dB")

        st.markdown("#### RGB Channel Histograms")
        hcols = st.columns(3)
        for i, (name, color) in enumerate([('Blue','#00B4D8'),
                                            ('Green','#10B981'),
                                            ('Red','#EF4444')]):
            with hcols[i]:
                rh = cv2.calcHist([raw_r],    [i], None, [32], [0, 256]).flatten()
                eh = cv2.calcHist([enhanced], [i], None, [32], [0, 256]).flatten()
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=rh.tolist(), name='Raw',
                                         line=dict(color='#134E4A', width=1.5),
                                         fill='tozeroy', fillcolor='rgba(91,74,138,0.2)'))
                fig.add_trace(go.Scatter(y=eh.tolist(), name='Enhanced',
                                         line=dict(color=color, width=2),
                                         fill='tozeroy', fillcolor='rgba(20,184,166,0.15)'))
                fig.update_layout(title=name, height=200,
                                  margin=dict(t=30, b=15, l=10, r=10),
                                  **PLOTLY_BASE, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        if model_ready:
            st.markdown("#### Object Detection & Species Classification")
            
            det_mode = st.radio("Detection Mode", ["Classical Saliency (Fish Masking)", "Deep Learning (YOLOv8 Objects)"], horizontal=True)
            
            if det_mode == "Deep Learning (YOLOv8 Objects)":
                try:
                    with st.spinner("Loading YOLOv8 AI..."):
                        yolo_model = YOLO('yolov8n.pt')
                    res_yolo = yolo_model(enhanced)
                    res_plotted = res_yolo[0].plot()
                    st.image(bgr_to_rgb(res_plotted), caption="YOLOv8 General Object Detection", use_container_width=True)
                    st.info("💡 YOLOv8 is detecting general everyday objects (people, boats, birds). To detect specific fish species perfectly, train a custom YOLO model and replace the 'yolov8n.pt' file!")
                except Exception as e:
                    st.error(f"YOLO error: {e}")
            else:
                bbox, cropped = detect_salient_object(enhanced)
                bbox_img = draw_bounding_box(enhanced, bbox)
                
                c3, c4 = st.columns(2)
                with c3:
                    st.image(bgr_to_rgb(bbox_img), caption="Salient Object Detection", use_container_width=True)
                with c4:
                    st.image(bgr_to_rgb(cropped), caption="Cropped Region for Classification", use_container_width=True)
                
                md              = load_model(model_path)
                lbl, conf, top3 = predict_image(cropped, md)

                st.markdown(f"""
                <div class="card">
                    <div style="display:flex;gap:2rem;align-items:center;flex-wrap:wrap;">
                        <div>
                            <div style="color:#5EEAD4;font-size:0.73rem;
                                        text-transform:uppercase;letter-spacing:0.08em;">Predicted</div>
                            <div style="color:#10B981;font-size:1.6rem;font-weight:700;">{lbl}</div>
                        </div>
                        <div>
                            <div style="color:#5EEAD4;font-size:0.73rem;
                                        text-transform:uppercase;letter-spacing:0.08em;">Confidence</div>
                            <div style="color:#2DD4BF;font-size:1.6rem;font-weight:700;
                                        font-family:'JetBrains Mono',monospace;">{conf}%</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
    
                st.markdown("**Top 3 Predictions**")
                for pl, pp in top3:
                    st.markdown(f"""
                    <div class="pred-bar-wrap">
                        <div class="pred-label">{pl}</div>
                        <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:8px;">
                            <div class="pred-bar" style="width:{int(pp)}%;"></div>
                        </div>
                        <div class="pred-pct">{pp}%</div>
                    </div>""", unsafe_allow_html=True)
