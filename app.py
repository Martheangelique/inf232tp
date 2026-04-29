#!/usr/bin/env python3
"""
EpidemSys - Système Ultra-Performant de Collecte & Analyse Épidémiologique
Auteur: Claude / Anthropic
"""

import http.server
import socketserver
import json
import os
import sys
import threading
import webbrowser
import csv
import io
import base64
import traceback
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, silhouette_score
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  ÉTAT GLOBAL DE L'APPLICATION
# ─────────────────────────────────────────────
app_state = {
    "data": None,         # DataFrame courant
    "filename": None,
    "history": []         # historique des actions
}

PORT = 8765

# ─────────────────────────────────────────────
#  GÉNÉRATEUR DE DONNÉES SYNTHÉTIQUES
# ─────────────────────────────────────────────
def generate_synthetic_data(n=200, disease_type="infectious"):
    np.random.seed(42)
    ages = np.random.normal(45, 18, n).clip(1, 95).astype(int)
    sexes = np.random.choice(["Homme", "Femme"], n)
    regions = np.random.choice(["Centre", "Littoral", "Nord", "Sud", "Est", "Ouest"], n)

    if disease_type == "infectious":
        exposure = np.random.binomial(1, 0.4, n)
        incubation = np.random.exponential(5, n).round(1)
        severity = (0.3*exposure + 0.01*(ages/10) + np.random.normal(0, 0.1, n)).clip(0, 1)
        recovered = np.random.binomial(1, 0.85, n)
        hospitalized = np.random.binomial(1, severity * 0.5, n)
        deaths = np.random.binomial(1, severity * 0.05, n)
        duration = (incubation + np.random.exponential(7, n)).round(1)
        temperature = np.where(exposure, np.random.normal(38.5, 0.8, n), np.random.normal(36.8, 0.3, n)).round(1)
        viral_load = np.where(exposure, np.random.exponential(1000, n), np.random.exponential(10, n)).round(0)

        df = pd.DataFrame({
            "ID": range(1, n+1),
            "Age": ages,
            "Sexe": sexes,
            "Region": regions,
            "Exposition": exposure,
            "Incubation_jours": incubation,
            "Temperature": temperature,
            "Charge_virale": viral_load,
            "Severite": severity.round(3),
            "Hospitalise": hospitalized,
            "Gueri": recovered,
            "Deces": deaths,
            "Duree_maladie_jours": duration
        })

    elif disease_type == "chronic":
        bmi = np.random.normal(26, 4, n).clip(16, 45).round(1)
        smoking = np.random.binomial(1, 0.3, n)
        physical_act = np.random.normal(3, 1.5, n).clip(0, 7).round(1)
        systolic = (120 + 0.4*ages + 5*smoking - 2*physical_act + np.random.normal(0, 10, n)).round(0)
        diastolic = (80 + 0.2*ages + 3*smoking - physical_act + np.random.normal(0, 7, n)).round(0)
        glycemia = (1.0 + 0.01*ages + 0.05*bmi + 0.1*smoking + np.random.normal(0, 0.2, n)).round(2)
        cholesterol = (1.8 + 0.01*ages + 0.05*bmi + 0.08*smoking + np.random.normal(0, 0.3, n)).round(2)
        diabetes = np.where(glycemia > 1.26, 1, 0)
        hypertension = np.where(systolic > 140, 1, 0)

        df = pd.DataFrame({
            "ID": range(1, n+1),
            "Age": ages,
            "Sexe": sexes,
            "Region": regions,
            "IMC": bmi,
            "Fumeur": smoking,
            "Activite_physique_h": physical_act,
            "Tension_systolique": systolic,
            "Tension_diastolique": diastolic,
            "Glycemie": glycemia,
            "Cholesterol": cholesterol,
            "Diabete": diabetes,
            "Hypertension": hypertension
        })

    else:  # environmental
        temperature_env = np.random.normal(28, 6, n).round(1)
        pollution = np.random.exponential(40, n).clip(0, 200).round(1)
        humidity = np.random.normal(65, 15, n).clip(20, 100).round(1)
        cases = (2 + 0.05*pollution + 0.03*temperature_env + np.random.poisson(3, n)).round(0).astype(int)
        respiratory = np.random.binomial(1, (pollution/200)*0.7, n)
        skin = np.random.binomial(1, (humidity/100)*0.3, n)

        df = pd.DataFrame({
            "ID": range(1, n+1),
            "Age": ages,
            "Sexe": sexes,
            "Region": regions,
            "Temperature_env": temperature_env,
            "Pollution_PM25": pollution,
            "Humidite": humidity,
            "Cas_semaine": cases,
            "Maladie_respiratoire": respiratory,
            "Maladie_cutanee": skin
        })

    return df

# ─────────────────────────────────────────────
#  ANALYSE : RÉGRESSION SIMPLE
# ─────────────────────────────────────────────
def analyse_regression_simple(df, x_col, y_col):
    df_clean = df[[x_col, y_col]].dropna()
    X = df_clean[[x_col]].values
    y = df_clean[y_col].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    slope, intercept, r_val, p_val, std_err = stats.linregress(X.flatten(), y)

    # Graphique
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        ax.title.set_color('#e6edf3')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    axes[0].scatter(X, y, alpha=0.6, color='#58a6ff', s=30, label='Données')
    axes[0].plot(X, y_pred, color='#ff6e96', lw=2, label=f'Régression (R²={r2:.3f})')
    axes[0].set_xlabel(x_col)
    axes[0].set_ylabel(y_col)
    axes[0].set_title(f'Régression Simple: {x_col} → {y_col}')
    axes[0].legend(facecolor='#21262d', labelcolor='#e6edf3')

    residuals = y - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, color='#3fb950', s=30)
    axes[1].axhline(0, color='#ff6e96', lw=1.5, ls='--')
    axes[1].set_xlabel('Valeurs prédites')
    axes[1].set_ylabel('Résidus')
    axes[1].set_title('Analyse des Résidus')

    plt.tight_layout()
    img_b64 = fig_to_base64(fig)
    plt.close()

    return {
        "success": True,
        "type": "regression_simple",
        "params": {
            "coefficient": round(float(slope), 6),
            "intercept": round(float(intercept), 6),
            "r2": round(float(r2), 4),
            "r": round(float(r_val), 4),
            "p_value": round(float(p_val), 6),
            "mse": round(float(mse), 4),
            "rmse": round(float(np.sqrt(mse)), 4),
            "n": int(len(df_clean)),
            "equation": f"{y_col} = {slope:.4f} × {x_col} + {intercept:.4f}"
        },
        "image": img_b64
    }

# ─────────────────────────────────────────────
#  ANALYSE : RÉGRESSION MULTIPLE
# ─────────────────────────────────────────────
def analyse_regression_multiple(df, x_cols, y_col):
    df_clean = df[x_cols + [y_col]].dropna()
    X = df_clean[x_cols].values
    y = df_clean[y_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    # Coefficients standardisés
    coefs = dict(zip(x_cols, model.coef_.round(4)))

    # VIF approximatif
    vif_data = {}
    for i, col in enumerate(x_cols):
        X_others = np.delete(X_scaled, i, axis=1)
        r2_vif = r2_score(X_scaled[:, i], LinearRegression().fit(X_others, X_scaled[:, i]).predict(X_others))
        vif_data[col] = round(1 / (1 - r2_vif + 1e-10), 2)

    # Graphiques
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        ax.title.set_color('#e6edf3')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    axes[0].scatter(y, y_pred, alpha=0.6, color='#58a6ff', s=30)
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0].set_xlabel('Valeurs réelles')
    axes[0].set_ylabel('Valeurs prédites')
    axes[0].set_title(f'Prédictions vs Réel (R²={r2:.3f})')

    colors = ['#58a6ff' if v > 0 else '#ff6e96' for v in model.coef_]
    axes[1].barh(x_cols, model.coef_, color=colors)
    axes[1].axvline(0, color='#e6edf3', lw=0.5)
    axes[1].set_xlabel('Coefficient standardisé')
    axes[1].set_title('Importance des Variables')

    residuals = y - y_pred
    axes[2].scatter(y_pred, residuals, alpha=0.6, color='#3fb950', s=30)
    axes[2].axhline(0, color='#ff6e96', lw=1.5, ls='--')
    axes[2].set_xlabel('Valeurs prédites')
    axes[2].set_ylabel('Résidus')
    axes[2].set_title('Résidus')

    plt.tight_layout()
    img_b64 = fig_to_base64(fig)
    plt.close()

    return {
        "success": True,
        "type": "regression_multiple",
        "params": {
            "r2": round(float(r2), 4),
            "mse": round(float(mse), 4),
            "rmse": round(float(np.sqrt(mse)), 4),
            "intercept": round(float(model.intercept_), 4),
            "coefficients": {k: float(v) for k, v in coefs.items()},
            "vif": vif_data,
            "n": int(len(df_clean)),
            "n_predictors": len(x_cols)
        },
        "image": img_b64
    }

# ─────────────────────────────────────────────
#  ANALYSE : CLUSTERING K-MEANS
# ─────────────────────────────────────────────
def analyse_clustering(df, cols, n_clusters=3):
    df_clean = df[cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean.values)

    # Elbow method
    inertias = []
    sil_scores = []
    k_range = range(2, min(9, len(df_clean)//3))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        if k > 1:
            sil_scores.append(silhouette_score(X_scaled, labels))

    # K-Means final
    km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km_final.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)

    # PCA pour visualisation 2D
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_scaled)

    # Statistiques par cluster
    df_clean = df_clean.copy()
    df_clean['Cluster'] = labels
    cluster_stats = df_clean.groupby('Cluster').agg(['mean', 'std', 'count'])

    # Graphiques
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        ax.title.set_color('#e6edf3')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    palette = ['#58a6ff', '#3fb950', '#ff6e96', '#d2a8ff', '#ffa657', '#79c0ff']
    for i in range(n_clusters):
        mask = labels == i
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=palette[i % len(palette)], alpha=0.7, s=40, label=f'Cluster {i}')
    centers_2d = pca_2d.transform(km_final.cluster_centers_)
    axes[0].scatter(centers_2d[:, 0], centers_2d[:, 1], c='white', marker='*', s=200, zorder=5)
    axes[0].set_title(f'Clusters (Sil.={sil:.3f})')
    axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].legend(facecolor='#21262d', labelcolor='#e6edf3', fontsize=8)

    axes[1].plot(list(k_range), inertias, 'o-', color='#58a6ff', lw=2)
    axes[1].set_xlabel('Nombre de clusters (k)')
    axes[1].set_ylabel('Inertie')
    axes[1].set_title('Méthode du Coude')

    sizes = [int((labels == i).sum()) for i in range(n_clusters)]
    axes[2].pie(sizes, labels=[f'Cluster {i}\n(n={s})' for i, s in enumerate(sizes)],
               colors=palette[:n_clusters], autopct='%1.1f%%',
               textprops={'color': '#e6edf3'})
    axes[2].set_title('Distribution des Clusters')

    plt.tight_layout()
    img_b64 = fig_to_base64(fig)
    plt.close()

    cluster_summary = {}
    for i in range(n_clusters):
        mask = labels == i
        cluster_summary[f"Cluster_{i}"] = {
            "n": int(mask.sum()),
            "pct": round(float(mask.mean() * 100), 1),
            "means": {col: round(float(df_clean.loc[mask, col].mean()), 3) for col in cols}
        }

    return {
        "success": True,
        "type": "clustering",
        "params": {
            "n_clusters": n_clusters,
            "silhouette_score": round(float(sil), 4),
            "total_inertia": round(float(km_final.inertia_), 2),
            "cluster_summary": cluster_summary
        },
        "image": img_b64
    }

# ─────────────────────────────────────────────
#  ANALYSE : ACP
# ─────────────────────────────────────────────
def analyse_acp(df, cols, n_components=None):
    df_clean = df[cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean.values)

    max_comp = min(len(cols), len(df_clean))
    pca = PCA(n_components=max_comp)
    X_pca = pca.fit_transform(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_comp_90 = int(np.searchsorted(cumvar, 0.90) + 1)

    if n_components is None:
        n_components = n_comp_90

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        ax.title.set_color('#e6edf3')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    # Scree plot
    axes[0].bar(range(1, max_comp+1), pca.explained_variance_ratio_ * 100,
               color='#58a6ff', alpha=0.8)
    ax2 = axes[0].twinx()
    ax2.plot(range(1, max_comp+1), cumvar * 100, 'o-', color='#ff6e96', lw=2)
    ax2.axhline(90, color='#3fb950', ls='--', lw=1.5)
    ax2.set_ylabel('Variance cumulée (%)', color='#ff6e96')
    ax2.tick_params(colors='#ff6e96')
    axes[0].set_xlabel('Composante principale')
    axes[0].set_ylabel('Variance expliquée (%)')
    axes[0].set_title('Scree Plot')

    # Biplot (PC1 vs PC2)
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.4, color='#8b949e', s=20)
    loadings = pca.components_[:2].T
    scale = 3
    for i, col in enumerate(cols):
        axes[1].annotate('', xy=(loadings[i, 0]*scale, loadings[i, 1]*scale),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='#ff6e96', lw=1.5))
        axes[1].text(loadings[i, 0]*scale*1.15, loadings[i, 1]*scale*1.15,
                    col, color='#e6edf3', fontsize=8, ha='center')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].set_title('Biplot PC1 vs PC2')
    axes[1].axhline(0, color='#30363d'); axes[1].axvline(0, color='#30363d')

    # Heatmap des loadings
    n_show = min(n_components, 4)
    loadings_df = pd.DataFrame(
        pca.components_[:n_show].T,
        index=cols,
        columns=[f'PC{i+1}' for i in range(n_show)]
    )
    sns.heatmap(loadings_df, ax=axes[2], cmap='RdBu_r', center=0,
               annot=True, fmt='.2f', annot_kws={'size': 8},
               cbar_kws={'label': 'Loading'})
    axes[2].set_title('Matrice des Loadings')
    axes[2].tick_params(colors='#e6edf3')

    plt.tight_layout()
    img_b64 = fig_to_base64(fig)
    plt.close()

    loadings_dict = {}
    for i in range(min(n_components, max_comp)):
        loadings_dict[f"PC{i+1}"] = {
            "variance_pct": round(float(pca.explained_variance_ratio_[i] * 100), 2),
            "eigenvalue": round(float(pca.explained_variance_[i]), 4),
            "loadings": {col: round(float(v), 4) for col, v in zip(cols, pca.components_[i])}
        }

    return {
        "success": True,
        "type": "acp",
        "params": {
            "n_variables": len(cols),
            "n_obs": int(len(df_clean)),
            "n_components_90pct": n_comp_90,
            "total_variance_explained": round(float(cumvar[n_components-1] * 100), 2),
            "components": loadings_dict
        },
        "image": img_b64
    }

# ─────────────────────────────────────────────
#  ANALYSE : STATISTIQUES DESCRIPTIVES
# ─────────────────────────────────────────────
def analyse_descriptive(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    desc = {}
    for col in numeric_cols:
        s = df[col].dropna()
        skewness = float(s.skew())
        kurt = float(s.kurtosis())
        desc[col] = {
            "n": int(s.count()),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "q25": round(float(s.quantile(0.25)), 4),
            "median": round(float(s.median()), 4),
            "q75": round(float(s.quantile(0.75)), 4),
            "max": round(float(s.max()), 4),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurt, 4),
            "missing": int(df[col].isnull().sum())
        }

    # Graphique corrélation + distributions
    n_num = len(numeric_cols)
    if n_num >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('#0d1117')
        for ax in axes:
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='#8b949e')
            ax.title.set_color('#e6edf3')
            for spine in ax.spines.values():
                spine.set_edgecolor('#30363d')

        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, ax=axes[0], mask=mask, cmap='RdBu_r', center=0,
                   annot=True if n_num <= 8 else False, fmt='.2f',
                   annot_kws={'size': 8}, square=True,
                   cbar_kws={'shrink': 0.8})
        axes[0].set_title('Matrice de Corrélation')
        axes[0].tick_params(colors='#e6edf3')

        # Distribution des 4 premières numériques
        show_cols = numeric_cols[:4]
        colors_dist = ['#58a6ff', '#3fb950', '#ff6e96', '#d2a8ff']
        for i, col in enumerate(show_cols):
            df[col].dropna().hist(ax=axes[1], bins=30, alpha=0.5,
                                  color=colors_dist[i], label=col, density=True)
        axes[1].set_title('Distributions (normalisées)')
        axes[1].set_xlabel('Valeur')
        axes[1].set_ylabel('Densité')
        axes[1].legend(facecolor='#21262d', labelcolor='#e6edf3', fontsize=8)

        plt.tight_layout()
        img_b64 = fig_to_base64(fig)
        plt.close()
    else:
        img_b64 = None

    return {
        "success": True,
        "type": "descriptive",
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "statistics": desc,
        "image": img_b64
    }

# ─────────────────────────────────────────────
#  UTILITAIRE
# ─────────────────────────────────────────────
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def df_to_json_safe(df):
    return json.loads(df.to_json(orient='records', default_handler=str))

# ─────────────────────────────────────────────
#  SERVEUR HTTP
# ─────────────────────────────────────────────
class EpidemHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # Silencer les logs

    def send_json(self, data, code=200):
        payload = json.dumps(data, ensure_ascii=False, default=str).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', len(payload))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(payload)

    def send_html(self, content):
        payload = content.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(payload))
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/' or path == '/index.html':
            with open(os.path.join(os.path.dirname(__file__), 'index.html'), 'r', encoding='utf-8') as f:
                self.send_html(f.read())

        elif path == '/api/status':
            df = app_state["data"]
            if df is not None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                self.send_json({
                    "loaded": True,
                    "filename": app_state["filename"],
                    "rows": int(df.shape[0]),
                    "cols": int(df.shape[1]),
                    "columns": df.columns.tolist(),
                    "numeric_cols": numeric_cols,
                    "cat_cols": cat_cols,
                    "preview": df_to_json_safe(df.head(8))
                })
            else:
                self.send_json({"loaded": False})

        elif path == '/api/export':
            df = app_state["data"]
            if df is None:
                self.send_json({"error": "Aucune donnée"}, 400)
                return
            csv_str = df.to_csv(index=False)
            payload = csv_str.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/csv; charset=utf-8')
            self.send_header('Content-Disposition', 'attachment; filename="epidem_data.csv"')
            self.send_header('Content-Length', len(payload))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        content_len = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_len) if content_len else b''

        try:
            if path == '/api/generate':
                params = json.loads(body) if body else {}
                n = int(params.get('n', 200))
                disease_type = params.get('type', 'infectious')
                df = generate_synthetic_data(n, disease_type)
                app_state["data"] = df
                app_state["filename"] = f"synthétique_{disease_type}_{n}obs"
                app_state["history"].append(f"Données générées: {disease_type}, n={n}")
                self.send_json({
                    "success": True,
                    "rows": int(df.shape[0]),
                    "cols": int(df.shape[1]),
                    "columns": df.columns.tolist(),
                    "numeric_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
                    "cat_cols": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                    "preview": df_to_json_safe(df.head(8))
                })

            elif path == '/api/upload':
                # CSV upload (base64)
                params = json.loads(body)
                csv_b64 = params.get('data', '')
                filename = params.get('filename', 'upload.csv')
                csv_bytes = base64.b64decode(csv_b64)
                df = pd.read_csv(io.BytesIO(csv_bytes))
                app_state["data"] = df
                app_state["filename"] = filename
                app_state["history"].append(f"Fichier chargé: {filename}")
                self.send_json({
                    "success": True,
                    "rows": int(df.shape[0]),
                    "cols": int(df.shape[1]),
                    "columns": df.columns.tolist(),
                    "numeric_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
                    "cat_cols": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                    "preview": df_to_json_safe(df.head(8))
                })

            elif path == '/api/manual_entry':
                params = json.loads(body)
                records = params.get('records', [])
                if not records:
                    self.send_json({"error": "Aucun enregistrement"}, 400)
                    return
                df = pd.DataFrame(records)
                # Conversion numérique auto
                for col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except:
                        pass
                app_state["data"] = df
                app_state["filename"] = "saisie_manuelle"
                app_state["history"].append(f"Saisie manuelle: {len(records)} enregistrements")
                self.send_json({
                    "success": True,
                    "rows": int(df.shape[0]),
                    "cols": int(df.shape[1]),
                    "columns": df.columns.tolist(),
                    "numeric_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
                    "preview": df_to_json_safe(df.head(8))
                })

            elif path == '/api/analyse':
                params = json.loads(body)
                df = app_state["data"]
                if df is None:
                    self.send_json({"error": "Aucune donnée chargée"}, 400)
                    return

                analysis_type = params.get('type', 'descriptive')

                if analysis_type == 'descriptive':
                    result = analyse_descriptive(df)

                elif analysis_type == 'regression_simple':
                    x_col = params.get('x_col')
                    y_col = params.get('y_col')
                    if not x_col or not y_col:
                        self.send_json({"error": "Colonnes X et Y requises"}, 400)
                        return
                    result = analyse_regression_simple(df, x_col, y_col)

                elif analysis_type == 'regression_multiple':
                    x_cols = params.get('x_cols', [])
                    y_col = params.get('y_col')
                    if len(x_cols) < 2 or not y_col:
                        self.send_json({"error": "Au moins 2 variables X et 1 variable Y"}, 400)
                        return
                    result = analyse_regression_multiple(df, x_cols, y_col)

                elif analysis_type == 'clustering':
                    cols = params.get('cols', [])
                    n_clusters = int(params.get('n_clusters', 3))
                    if len(cols) < 2:
                        self.send_json({"error": "Au moins 2 variables requises"}, 400)
                        return
                    result = analyse_clustering(df, cols, n_clusters)

                elif analysis_type == 'acp':
                    cols = params.get('cols', [])
                    if len(cols) < 2:
                        self.send_json({"error": "Au moins 2 variables requises"}, 400)
                        return
                    result = analyse_acp(df, cols)

                else:
                    self.send_json({"error": f"Type d'analyse inconnu: {analysis_type}"}, 400)
                    return

                app_state["history"].append(f"Analyse: {analysis_type}")
                self.send_json(result)

            else:
                self.send_json({"error": "Route inconnue"}, 404)

        except Exception as e:
            tb = traceback.format_exc()
            self.send_json({"error": str(e), "traceback": tb}, 500)


def run_server():
    with socketserver.TCPServer(("", PORT), EpidemHandler) as httpd:
        httpd.allow_reuse_address = True
        httpd.serve_forever()


if __name__ == "__main__":
    print("=" * 60)
    print("  EpidemSys — Système d'Analyse Épidémiologique")
    print("=" * 60)
    print(f"  Démarrage sur http://localhost:{PORT}")
    print("  Appuyez sur Ctrl+C pour arrêter")
    print("=" * 60)

    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    webbrowser.open(f"http://localhost:{PORT}")

    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Serveur arrêté.")
