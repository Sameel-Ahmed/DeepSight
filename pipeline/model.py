"""
model.py — Random Forest training, evaluation, serialisation, and prediction.
"""
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(X: np.ndarray, y: np.ndarray,
                class_names: list, model_type: str = 'Random Forest',
                n_estimators: int = 100, max_depth=None,
                svm_c: float = 1.0, svm_kernel: str = 'rbf',
                progress_cb=None) -> dict:
    """
    Train a classifier on feature matrix X with integer labels y.
    Returns a results dict with model, metrics, and plot data.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_type == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    elif model_type == 'SVM':
        clf = SVC(C=svm_c, kernel=svm_kernel, probability=True, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if progress_cb: progress_cb(0.2)
    clf.fit(X_tr, y_tr)

    if progress_cb: progress_cb(0.8)
    y_pred = clf.predict(X_te)

    acc  = accuracy_score(y_te, y_pred) * 100
    f1   = f1_score(y_te, y_pred, average='weighted')
    cm   = confusion_matrix(y_te, y_pred)
    rpt  = classification_report(y_te, y_pred,
                                  target_names=class_names,
                                  output_dict=True)

    if progress_cb: progress_cb(1.0)

    return {
        'model':       clf,
        'class_names': class_names,
        'accuracy':    acc,
        'f1':          f1,
        'cm':          cm,
        'report':      rpt,
        'y_test':      y_te,
        'y_pred':      y_pred,
        'train_size':  len(X_tr),
        'test_size':   len(X_te),
    }


# ── Serialisation ─────────────────────────────────────────────────────────────

def save_model(results: dict, path: str = 'model.pkl'):
    joblib.dump({'model': results['model'],
                 'class_names': results['class_names']}, path)


def load_model(path: str = 'model.pkl') -> dict:
    return joblib.load(path)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_image(img: np.ndarray, model_dict: dict) -> tuple:
    """
    Returns (top_label, top_confidence_pct, top3_list)
    top3_list : [(label, pct), ...]
    """
    from pipeline.features import extract_features
    clf    = model_dict['model']
    labels = model_dict['class_names']
    feats  = extract_features(img)

    pred  = clf.predict([feats])[0]
    proba = clf.predict_proba([feats])[0]

    top3_idx  = np.argsort(proba)[-3:][::-1]
    top3      = [(labels[i], round(float(proba[i]) * 100, 1)) for i in top3_idx]

    return labels[pred], round(float(proba[pred]) * 100, 1), top3


# ── Plotly figures ────────────────────────────────────────────────────────────

_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,22,40,0.6)',
    font_color='#E2F0FF',
    margin=dict(t=60, b=40)
)


def confusion_matrix_fig(cm: np.ndarray, class_names: list) -> go.Figure:
    # Clip class names for display
    disp = [n[:15] + '…' if len(n) > 15 else n for n in class_names]
    total = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), total,
                        out=np.zeros_like(cm, dtype=float),
                        where=total != 0)

    text = [[f"{cm[i][j]}<br>({cm_norm[i][j]*100:.0f}%)"
             for j in range(len(disp))]
            for i in range(len(disp))]

    fig = ff.create_annotated_heatmap(
        z=cm_norm, x=disp, y=disp,
        annotation_text=text,
        colorscale='Blues',
        showscale=True
    )
    fig.update_layout(
        title='Confusion Matrix (normalised)',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=max(400, 40 * len(class_names)),
        **_LAYOUT
    )
    return fig


def feature_importance_fig(model, feat_names: list, top_n: int = 20) -> go.Figure:
    if hasattr(model, 'feature_importances_'):
        imps = model.feature_importances_
    elif hasattr(model, 'coef_'):
        imps = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        fig = go.Figure()
        fig.add_annotation(text="Feature importances not available for this model (e.g., non-linear SVM)",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color="#A78BFA"))
        fig.update_layout(title=f'Feature Importances', height=500, **_LAYOUT)
        return fig

    indices = np.argsort(imps)[-top_n:]

    fig = go.Figure(go.Bar(
        x=imps[indices],
        y=[feat_names[i] for i in indices],
        orientation='h',
        marker=dict(color=imps[indices],
                    colorscale=[[0, '#0077B6'], [1, '#00D4FF']],
                    showscale=False)
    ))
    fig.update_layout(
        title=f'Top {top_n} Feature Importances',
        xaxis_title='Importance Score',
        height=500,
        **_LAYOUT
    )
    return fig


def per_class_metrics_fig(report: dict, class_names: list) -> go.Figure:
    f1s  = [report.get(c, {}).get('f1-score', 0) for c in class_names]
    prec = [report.get(c, {}).get('precision', 0) for c in class_names]
    rec  = [report.get(c, {}).get('recall', 0) for c in class_names]
    disp = [n[:15] + '…' if len(n) > 15 else n for n in class_names]

    fig = go.Figure()
    for vals, name, color in [(prec, 'Precision', '#00D4FF'),
                               (rec,  'Recall',    '#00FF9F'),
                               (f1s,  'F1-Score',  '#FFD60A')]:
        fig.add_trace(go.Bar(name=name, x=disp, y=vals,
                             marker_color=color, opacity=0.85))

    fig.update_layout(
        title='Per-Class Metrics',
        yaxis_title='Score',
        barmode='group',
        xaxis_tickangle=-40,
        **_LAYOUT
    )
    return fig
