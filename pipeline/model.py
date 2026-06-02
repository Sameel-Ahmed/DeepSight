"""
model.py — Multi-classifier training, evaluation, serialisation, and prediction.
Supports: Random Forest, SVM, KNN, Naive Bayes, XGBoost, and Ensemble variants.
"""
import os
import datetime
import numpy as np
import joblib

import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier,
                               HistGradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_auc_score)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_split(X, y):
    """80/20 stratified split with fallback if any class has < 2 samples."""
    try:
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        return train_test_split(X, y, test_size=0.2, random_state=42)


def _build_clf(model_type: str, n_estimators=100, max_depth=None,
               svm_c=1.0, svm_kernel='rbf', knn_k=5):
    """Build a sklearn Pipeline for the given model type."""
    if model_type == 'Random Forest':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                random_state=42, n_jobs=-1))
        ])
    elif model_type == 'SVM':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(C=svm_c, kernel=svm_kernel, probability=True, random_state=42))
        ])
    elif model_type == 'KNN':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=knn_k, n_jobs=-1))
        ])
    elif model_type == 'Naive Bayes':
        # GaussianNB works on any real-valued features (no non-negativity requirement)
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GaussianNB())
        ])
    elif model_type == 'XGBoost':
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', XGBClassifier(
                n_estimators=n_estimators, random_state=42,
                n_jobs=-1, eval_metric='mlogloss', verbosity=0))
        ])
    elif model_type == 'Ensemble (Voting)':
        rf  = Pipeline([('scaler', StandardScaler()),
                        ('clf', RandomForestClassifier(n_estimators=n_estimators,
                                                       max_depth=max_depth, random_state=42, n_jobs=-1))])
        svm = Pipeline([('scaler', StandardScaler()),
                        ('clf', SVC(C=svm_c, kernel=svm_kernel, probability=True, random_state=42))])
        return VotingClassifier(estimators=[('rf', rf), ('svm', svm)], voting='soft')
    elif model_type == 'Ensemble (RF+SVM+GBM)':
        rf  = Pipeline([('scaler', StandardScaler()),
                        ('clf', RandomForestClassifier(n_estimators=n_estimators,
                                                       max_depth=max_depth, random_state=42, n_jobs=-1))])
        svm = Pipeline([('scaler', StandardScaler()),
                        ('clf', SVC(C=svm_c, kernel=svm_kernel, probability=True, random_state=42))])
        gbm = Pipeline([('scaler', StandardScaler()),
                        ('clf', HistGradientBoostingClassifier(max_iter=150, random_state=42))])
        return VotingClassifier(estimators=[('rf', rf), ('svm', svm), ('gbm', gbm)], voting='soft')
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _compute_metrics(clf, X_te, y_te, class_names):
    """Return a dict of all evaluation metrics for a fitted classifier."""
    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)

    acc  = accuracy_score(y_te, y_pred) * 100
    prec = precision_score(y_te, y_pred, average='weighted', zero_division=0) * 100
    rec  = recall_score(y_te, y_pred,  average='weighted', zero_division=0) * 100
    f1   = f1_score(y_te, y_pred,      average='weighted', zero_division=0)

    try:
        # Step 1: clip near-zero probabilities and renormalise each row.
        # Fixes Naive Bayes numerical underflow where entire rows collapse to 0.
        y_proba_safe = np.clip(y_proba, 1e-10, None)
        row_sums = y_proba_safe.sum(axis=1, keepdims=True)
        y_proba_safe = y_proba_safe / row_sums

        # Step 2: align probability matrix to classes actually present in y_te.
        # Fixes Super Ensemble / any model whose classes_ spans more labels than y_te.
        all_classes = np.arange(y_proba_safe.shape[1])   # 0 … n_classes-1
        present_in_te = np.array(sorted(np.unique(y_te)), dtype=int)
        if len(present_in_te) < y_proba_safe.shape[1]:
            # Keep only the columns for classes that appear in y_te
            y_proba_safe = y_proba_safe[:, present_in_te]
            # Renormalise again after column drop
            row_sums2 = y_proba_safe.sum(axis=1, keepdims=True)
            y_proba_safe = y_proba_safe / np.where(row_sums2 == 0, 1, row_sums2)

        if len(present_in_te) == 1:
            # Only one class in test set — AUC undefined
            auc = float('nan')
        elif len(present_in_te) == 2:
            # Binary case — use the positive-class column
            auc = roc_auc_score(y_te, y_proba_safe[:, 1])
        else:
            auc = roc_auc_score(
                y_te, y_proba_safe,
                multi_class='ovr',
                average='weighted',
                labels=present_in_te,
            )
    except Exception:
        auc = float('nan')


    present_labels = sorted(np.unique(np.concatenate([y_te, y_pred])))
    present_names  = [class_names[i] for i in present_labels if i < len(class_names)]
    cm  = confusion_matrix(y_te, y_pred, labels=present_labels)
    rpt = classification_report(
        y_te, y_pred,
        labels=present_labels,
        target_names=present_names,
        output_dict=True,
        zero_division=0
    )

    return {
        'accuracy':    acc,
        'precision':   prec,
        'recall':      rec,
        'f1':          f1,
        'auc':         auc,
        'cm':          cm,
        'report':      rpt,
        'class_names': present_names,
        'y_pred':      y_pred,
    }


# ── Single-model Training ─────────────────────────────────────────────────────

def train_model(X: np.ndarray, y: np.ndarray,
                class_names: list, model_type: str = 'Random Forest',
                n_estimators: int = 100, max_depth=None,
                svm_c: float = 1.0, svm_kernel: str = 'rbf',
                knn_k: int = 5,
                progress_cb=None) -> dict:
    """
    Train a single classifier on feature matrix X with integer labels y.
    Returns a results dict compatible with the rest of the app.
    """
    X_tr, X_te, y_tr, y_te = _safe_split(X, y)

    clf = _build_clf(model_type, n_estimators, max_depth, svm_c, svm_kernel, knn_k)

    if progress_cb: progress_cb(0.2)
    clf.fit(X_tr, y_tr)
    if progress_cb: progress_cb(0.8)

    metrics = _compute_metrics(clf, X_te, y_te, class_names)

    if progress_cb: progress_cb(1.0)

    return {
        'model':       clf,
        'class_names': metrics['class_names'],
        'accuracy':    metrics['accuracy'],
        'precision':   metrics['precision'],
        'recall':      metrics['recall'],
        'f1':          metrics['f1'],
        'auc':         metrics['auc'],
        'cm':          metrics['cm'],
        'report':      metrics['report'],
        'y_test':      y_te,
        'y_pred':      metrics['y_pred'],
        'train_size':  len(X_tr),
        'test_size':   len(X_te),
        'type':        model_type,
    }


# ── Soft Ensemble (reuses already-fitted models) ──────────────────────────────

class SoftEnsemble:
    """
    Lightweight ensemble that averages predict_proba from already-trained
    classifiers — no re-training required, no extra time cost.
    """
    def __init__(self, estimators: list):
        # estimators: list of (name, fitted_clf)
        self.estimators   = estimators
        self.classes_     = None
        self.class_names_ = None

    def fit(self, X, y):
        """No-op: sub-classifiers are already fitted. Just record classes."""
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        probas = np.array([clf.predict_proba(X) for _, clf in self.estimators])
        return probas.mean(axis=0)          # soft vote = average probabilities

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def __repr__(self):
        names = [n for n, _ in self.estimators]
        return f"SoftEnsemble([{', '.join(names)}])"


# ── Compare All 5 Models ──────────────────────────────────────────────────────

ALL_MODEL_TYPES = ['Random Forest', 'SVM', 'KNN', 'Naive Bayes', 'XGBoost']
SUPER_ENSEMBLE_LABEL = '🏆 Super Ensemble (all 5)'

def train_all_models(X: np.ndarray, y: np.ndarray,
                     class_names: list,
                     n_estimators: int = 100,
                     progress_cb=None) -> tuple:
    """
    Train all 5 classifiers on the same train/test split, then build a
    SoftEnsemble from all of them and save THAT as the final model.

    Returns:
        comparison : list of dicts — 6 rows (5 individual + Super Ensemble)
        all_results: list of full result dicts
        best_result: the SoftEnsemble result dict (saved to model.pkl)
    """
    X_tr, X_te, y_tr, y_te = _safe_split(X, y)

    comparison   = []
    all_results  = []
    trained_clfs = []          # (name, fitted_clf) pairs for the ensemble
    n_steps      = len(ALL_MODEL_TYPES) + 1  # +1 for ensemble step

    for i, name in enumerate(ALL_MODEL_TYPES):
        try:
            clf = _build_clf(name, n_estimators=n_estimators)
            clf.fit(X_tr, y_tr)
            trained_clfs.append((name, clf))
            metrics = _compute_metrics(clf, X_te, y_te, class_names)

            comparison.append({
                'Model':         name,
                'Accuracy (%)':  round(metrics['accuracy'],  2),
                'Precision (%)': round(metrics['precision'], 2),
                'Recall (%)':    round(metrics['recall'],    2),
                'F1-Score':      round(metrics['f1'],        4),
                'AUC':           round(metrics['auc'],       4) if not (
                    isinstance(metrics['auc'], float) and metrics['auc'] != metrics['auc']
                ) else 'N/A',
            })
            all_results.append({
                'model':       clf,
                'class_names': metrics['class_names'],
                'accuracy':    metrics['accuracy'],
                'precision':   metrics['precision'],
                'recall':      metrics['recall'],
                'f1':          metrics['f1'],
                'auc':         metrics['auc'],
                'cm':          metrics['cm'],
                'report':      metrics['report'],
                'y_test':      y_te,
                'y_pred':      metrics['y_pred'],
                'train_size':  len(X_tr),
                'test_size':   len(X_te),
                'type':        name,
            })

        except Exception as e:
            comparison.append({
                'Model': name, 'Accuracy (%)': 'ERR', 'Precision (%)': 'ERR',
                'Recall (%)': 'ERR', 'F1-Score': 'ERR', 'AUC': str(e)[:60],
            })

        if progress_cb:
            progress_cb((i + 1) / n_steps)

    # ── Build Super Ensemble from all successfully trained models ────────────
    ensemble_result = None
    if len(trained_clfs) >= 2:
        try:
            se = SoftEnsemble(trained_clfs)
            se.fit(X_tr, y_tr)             # no-op internally, just sets classes_
            se.class_names_ = class_names
            metrics = _compute_metrics(se, X_te, y_te, class_names)

            comparison.append({
                'Model':         SUPER_ENSEMBLE_LABEL,
                'Accuracy (%)':  round(metrics['accuracy'],  2),
                'Precision (%)': round(metrics['precision'], 2),
                'Recall (%)':    round(metrics['recall'],    2),
                'F1-Score':      round(metrics['f1'],        4),
                'AUC':           round(metrics['auc'],       4) if not (
                    isinstance(metrics['auc'], float) and metrics['auc'] != metrics['auc']
                ) else 'N/A',
            })
            ensemble_result = {
                'model':       se,
                'class_names': metrics['class_names'],
                'accuracy':    metrics['accuracy'],
                'precision':   metrics['precision'],
                'recall':      metrics['recall'],
                'f1':          metrics['f1'],
                'auc':         metrics['auc'],
                'cm':          metrics['cm'],
                'report':      metrics['report'],
                'y_test':      y_te,
                'y_pred':      metrics['y_pred'],
                'train_size':  len(X_tr),
                'test_size':   len(X_te),
                'type':        SUPER_ENSEMBLE_LABEL,
            }
            all_results.append(ensemble_result)
        except Exception:
            pass

    if progress_cb:
        progress_cb(1.0)

    # Always return Super Ensemble as best_result if it was built
    if ensemble_result:
        best_result = ensemble_result
    else:
        valid = [r for r in all_results if isinstance(r['f1'], float)]
        best_result = max(valid, key=lambda r: r['f1']) if valid else (all_results[0] if all_results else None)

    return comparison, all_results, best_result


# ── Evaluate existing model ───────────────────────────────────────────────────

def evaluate_model(clf, X: np.ndarray, y: np.ndarray, class_names: list) -> dict:
    """Evaluate an already-trained model on new data X, y."""
    metrics = _compute_metrics(clf, X, y, class_names)

    return {
        'model':       clf,
        'class_names': metrics['class_names'],
        'accuracy':    metrics['accuracy'],
        'precision':   metrics['precision'],
        'recall':      metrics['recall'],
        'f1':          metrics['f1'],
        'auc':         metrics['auc'],
        'cm':          metrics['cm'],
        'report':      metrics['report'],
        'y_test':      y,
        'y_pred':      metrics['y_pred'],
        'train_size':  0,
        'test_size':   len(X),
        'type':        'Loaded Model',
    }


# ── Serialisation ─────────────────────────────────────────────────────────────

def save_model(results: dict, path: str = 'model.pkl'):
    joblib.dump({'model': results['model'],
                 'class_names': results['class_names']}, path)


def load_model(path: str = 'model.pkl') -> dict:
    return joblib.load(path)


def save_results(comparison: list, all_results: list,
                 cache_path: str = 'data/model_results.pkl'):
    """
    Persist the comparison table and per-model metrics to disk.
    Model objects are stripped to keep the file small (~1–5 MB).
    Confusion matrices, reports, and y_pred are all preserved so
    charts can be fully reconstructed on load.
    """
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)

    slim = [{k: v for k, v in r.items() if k != 'model'}
            for r in all_results]
    joblib.dump({
        'comparison':   comparison,
        'slim_results': slim,
        'saved_at':     datetime.datetime.now().isoformat(timespec='minutes'),
    }, cache_path)


def load_results(cache_path: str = 'data/model_results.pkl') -> dict | None:
    """
    Load previously saved training results.
    Returns dict with keys: comparison, slim_results, saved_at
    or None if the file does not exist or is corrupt.
    """
    if not os.path.exists(cache_path):
        return None
    try:
        return joblib.load(cache_path)
    except Exception:
        return None



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

    top3_idx = np.argsort(proba)[-3:][::-1]
    top3     = [(labels[i], round(float(proba[i]) * 100, 1)) for i in top3_idx]

    return labels[pred], round(float(proba[pred]) * 100, 1), top3


# ── Plotly figures ────────────────────────────────────────────────────────────

_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,22,40,0.6)',
    font_color='#E2F0FF',
    margin=dict(t=60, b=40)
)


def confusion_matrix_fig(cm: np.ndarray, class_names: list) -> go.Figure:
    disp  = [n[:15] + '…' if len(n) > 15 else n for n in class_names]
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
    inner = model
    if hasattr(inner, 'named_steps') and 'clf' in inner.named_steps:
        inner = inner.named_steps['clf']

    if hasattr(inner, 'feature_importances_'):
        imps = inner.feature_importances_
    elif hasattr(inner, 'coef_'):
        imps = np.abs(inner.coef_[0]) if inner.coef_.ndim > 1 else np.abs(inner.coef_)
    else:
        fig = go.Figure()
        fig.add_annotation(
            text="Feature importances not available for this model type.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#2DD4BF"))
        fig.update_layout(title='Feature Importances', height=500, **_LAYOUT)
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
    f1s  = [report.get(c, {}).get('f1-score',  0) for c in class_names]
    prec = [report.get(c, {}).get('precision',  0) for c in class_names]
    rec  = [report.get(c, {}).get('recall',     0) for c in class_names]
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


def model_comparison_fig(comparison: list) -> go.Figure:
    """Bar chart comparing all 5 models across all metrics."""
    models   = [r['Model'] for r in comparison]
    metrics  = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score', 'AUC']
    colors   = ['#2DD4BF', '#FACC15', '#F97316', '#A78BFA', '#34D399']

    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        vals = []
        for r in comparison:
            v = r.get(metric, 0)
            # Normalise F1/AUC to percentage scale for visual comparison
            if metric in ('F1-Score', 'AUC') and isinstance(v, float):
                v = round(v * 100, 2)
            vals.append(v if isinstance(v, (int, float)) else 0)

        fig.add_trace(go.Bar(
            name=metric,
            x=models,
            y=vals,
            marker_color=color,
            opacity=0.85,
            text=[f"{v:.1f}" for v in vals],
            textposition='outside',
        ))

    fig.update_layout(
        title='Model Comparison — All Metrics (F1 & AUC scaled ×100)',
        yaxis_title='Score',
        barmode='group',
        legend=dict(orientation='h', y=-0.2),
        height=480,
        **_LAYOUT
    )
    return fig
