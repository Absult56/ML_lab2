import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Модули для масштабирования и понижения размерности
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from scipy.stats import kurtosis

# Модули для кластеризации и оценки
from sklearn.cluster import (KMeans, AffinityPropagation, MeanShift, 
                             SpectralClustering, AgglomerativeClustering, 
                             DBSCAN, HDBSCAN, OPTICS, Birch, BisectingKMeans)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# ШАГ 1: Загрузка и подготовка данных (Вариант 12)
# =====================================================================

# 1.1 Загрузка данных для PCA (столбцы 66-71 для 12-го варианта)
try:
    df_pca_full = pd.read_excel("D:\projects\python\ML_lab\lab2\data\Lab2_Dataset_PCA.xlsx")
    df_pca = df_pca_full.iloc[1:, 66:72].astype(float)
    df_pca.columns = ['a', 'b', 'c', 'd', 'e', 'y']
    df_pca.dropna(inplace=True)
    
    X_pca = df_pca[['a', 'b', 'c', 'd', 'e']].values
    y_pca = df_pca['y'].values
    features_pca = ['a', 'b', 'c', 'd', 'e']
except Exception as e:
    print(f"Ошибка загрузки Lab2_Dataset_PCA.xlsx: {e}")

# 1.2 Загрузка данных для кластеризации (столбцы 33-35 для 12-го варианта)
try:
    df_clust_full = pd.read_excel("D:\projects\python\ML_lab\lab2\data\Lab2_Dataset_Clustering.xlsx")
    df_clust = df_clust_full.iloc[1:, 33:36].astype(float)
    df_clust.columns = ['a', 'b', 'y']
    df_clust.dropna(inplace=True)
    
    X_clust = df_clust[['a', 'b']].values
    y_clust = df_clust['y'].values
except Exception as e:
    print(f"Ошибка загрузки Lab2_Dataset_Clustering.xlsx: {e}")

# =====================================================================
# ШАГ 2: Часть 1. Обучение без учителя. Понижение размерности
# =====================================================================

print("--- Анализ признаков (PCA, SVD, ICA) ---")

# 1.1 Матрица взаимных корреляций
plt.figure(figsize=(7, 6))
sns.heatmap(df_pca[features_pca].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Матрица взаимных корреляций (Вариант 12)")
plt.tight_layout()
plt.show()

# 1.2 Матрица разброса признаков
pd.plotting.scatter_matrix(df_pca[features_pca], figsize=(10, 10), diagonal='kde', alpha=0.7)
plt.suptitle("Матрица разброса признаков", y=1.02)
plt.show()

# 1.3 Оценка значимости признаков (PCA, SVD, ICA)
scaler_pca = StandardScaler()
X_scaled = scaler_pca.fit_transform(X_pca)

pca = PCA(n_components=5)
svd = TruncatedSVD(n_components=5)
ica = FastICA(n_components=5, random_state=42)

pca.fit(X_scaled)
svd.fit(X_scaled)
ica.fit(X_scaled)

# 1.4 Графики значимости компонент
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(range(1, 6), pca.explained_variance_ratio_)
axes[0].set_title('PCA: Объясненная дисперсия')
axes[0].set_xlabel('Компонента')

axes[1].bar(range(1, 6), svd.explained_variance_ratio_)
axes[1].set_title('SVD: Объясненная дисперсия')
axes[1].set_xlabel('Компонента')

# Для ICA используем нормализованный куртозис как меру значимости (негауссовости)
ica_sources = ica.transform(X_scaled)
kurt = np.abs(kurtosis(ica_sources, axis=0))
kurt_normalized = kurt / np.sum(kurt)
axes[2].bar(range(1, 6), kurt_normalized)
axes[2].set_title('ICA: Нормализованный куртозис')
axes[2].set_xlabel('Компонента')

plt.tight_layout()
plt.show()

# 1.5 Построение Scatter-графиков проекций (Biplot)
def biplot(scores, loadings, labels=None, targets=None, target_labels=None, title="Biplot"):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scalex = 1.0 / (scores[:, 0].max() - scores[:, 0].min())
    scaley = 1.0 / (scores[:, 1].max() - scores[:, 1].min())
    
    if targets is not None:
        scatter = ax.scatter(scores[:, 0] * scalex, scores[:, 1] * scaley, 
                             c=targets, cmap='viridis', alpha=0.7, edgecolors='k')
        if target_labels is not None:
            handles, _ = scatter.legend_elements()
            ax.legend(handles, target_labels, title="Classes")
    else:
        ax.scatter(scores[:, 0] * scalex, scores[:, 1] * scaley, alpha=0.7)
        
    n_features = loadings.shape[0]
    for i in range(n_features):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.7,
                 head_width=0.03, head_length=0.03, linewidth=1.5)
        if labels is not None:
            ax.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, labels[i],
                    color='darkred', ha='center', va='center', fontweight='bold')
            
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    plt.tight_layout()
    plt.show()

# Biplot для PCA
scores_pca = pca.transform(X_scaled)[:, :2]
loadings_pca = pca.components_[:2, :].T
biplot(scores_pca, loadings_pca, labels=features_pca, targets=y_pca, target_labels=np.unique(y_pca), title="PCA Biplot")

# Biplot для SVD
scores_svd = svd.transform(X_scaled)[:, :2]
loadings_svd = svd.components_[:2, :].T
biplot(scores_svd, loadings_svd, labels=features_pca, targets=y_pca, target_labels=np.unique(y_pca), title="SVD Biplot")

# Biplot для ICA (используем матрицу смешивания)
loadings_ica = ica.mixing_[:, :2]
loadings_ica = loadings_ica / np.max(np.abs(loadings_ica), axis=0) # Нормализация для визуала
biplot(ica_sources[:, :2], loadings_ica, labels=features_pca, targets=y_pca, target_labels=np.unique(y_pca), title="ICA Biplot")


# =====================================================================
# ШАГ 3: Часть 2. Кластеризация
# =====================================================================

print("\n--- Кластеризация и обнаружение аномалий ---")
X_c_scaled = StandardScaler().fit_transform(X_clust)

# Интеграция метода локтя для определения оптимального числа кластеров
distortions = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    km.fit(X_c_scaled)
    distortions.append(km.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(K_range, distortions, marker='o', color='b')
plt.xlabel('Количество кластеров')
plt.ylabel('Искажение (Inertia)')
plt.title('Метод локтя (Elbow Method)')
plt.grid(True)
plt.show()


OPTIMAL_CLUSTERS = 3 

cl_algs = {
    'K-Means': KMeans(n_clusters=OPTIMAL_CLUSTERS, random_state=42),
    'Affinity Prop': AffinityPropagation(random_state=42),
    'Mean-Shift': MeanShift(),
    'Spectral': SpectralClustering(n_clusters=OPTIMAL_CLUSTERS, random_state=42),
    'Ward': AgglomerativeClustering(n_clusters=OPTIMAL_CLUSTERS, linkage='ward'),
    'Agglomerative': AgglomerativeClustering(n_clusters=OPTIMAL_CLUSTERS, linkage='average'),
    'DBSCAN': DBSCAN(eps=0.3, min_samples=5), 
    'HDBSCAN': HDBSCAN(min_cluster_size=5),
    'OPTICS': OPTICS(min_samples=5),
    'GMM': GaussianMixture(n_components=OPTIMAL_CLUSTERS, random_state=42),
    'BIRCH': Birch(n_clusters=OPTIMAL_CLUSTERS),
    'Bisecting K-M': BisectingKMeans(n_clusters=OPTIMAL_CLUSTERS, random_state=42)
}

fig, axes = plt.subplots(3, 4, figsize=(22, 16))
axes = axes.flatten()

for i, (name, alg) in enumerate(cl_algs.items()):
    try:
        y_pred = alg.fit_predict(X_c_scaled)
    except AttributeError:
        # Обработка для алгоритмов без метода fit_predict (например, GMM)
        y_pred = alg.fit(X_c_scaled).predict(X_c_scaled)
        
    axes[i].scatter(X_c_scaled[:, 0], X_c_scaled[:, 1], c=y_pred, cmap='tab10', s=25, alpha=0.8)
    axes[i].set_title(name, fontsize=14)
    axes[i].set_xticks([])
    axes[i].set_yticks([])

plt.tight_layout()
plt.show()

# Расчет силуэтного коэффициента для оценки качества (на примере K-Means)
km_eval = KMeans(n_clusters=OPTIMAL_CLUSTERS, random_state=42)
y_km_eval = km_eval.fit_predict(X_c_scaled)
sil_score = silhouette_score(X_c_scaled, y_km_eval)
print(f"Силуэтный коэффициент для K-Means ({OPTIMAL_CLUSTERS} кластера): {sil_score:.3f}")

# =====================================================================
# ШАГ 4: Выделение аномалий
# =====================================================================

# Используем DBSCAN для выделения аномалий
db = DBSCAN(eps=0.35, min_samples=6).fit(X_c_scaled)
labels = db.labels_

plt.figure(figsize=(9, 7))
# Отрисовка нормальных кластеров
plt.scatter(X_c_scaled[labels != -1, 0], X_c_scaled[labels != -1, 1], 
            c=labels[labels != -1], cmap='tab10', label='Кластеры', s=40, alpha=0.8)
# Отрисовка аномалий
plt.scatter(X_c_scaled[labels == -1, 0], X_c_scaled[labels == -1, 1], 
            c='red', marker='x', label='Аномальные данные (Шум)', s=80, linewidths=2)

plt.title("Выделение аномальных данных с помощью DBSCAN")
plt.xlabel("Признак 1 (Scaled)")
plt.ylabel("Признак 2 (Scaled)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()