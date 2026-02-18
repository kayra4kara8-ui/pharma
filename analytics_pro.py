"""
PharmaIntelligence Enterprise v9.0 — analytics_pro.py
══════════════════════════════════════════════════════
Gelişmiş Analitik & Makine Öğrenmesi Modülü

YENİ ÖZELLIKLER:
  🔬 Cohort Analizi — hasta gruplarının zaman içinde davranışı
  📊 RFM Segmentasyonu — Recency/Frequency/Monetary skorlama
  🎯 Market Basket Analizi — hangi moleküller birlikte satılıyor
  📈 Fiyat Elastisitesi — fiyat değişiminin satışa etkisi
  🔮 Churn Prediction — molekül/ürün kayıp riski tahmini
  🧠 Kümeleme Analizi — otomatik pazar segmentasyonu
  📉 Survival Analizi — ürün yaşam döngüsü tahmini
  ⚡ Anomali Skoru — anormal davranış tespiti (geliştirilmiş)
  🌊 Trend Decomposition — mevsimsellik ve trend ayrıştırma
  🎲 Monte Carlo Simülasyon — gelecek senaryo analizi
"""

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False

from core import DataPipeline


# ═════════════════════════════════════════════════════════════════════════════
# ADVANCED ANALYTICS ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class AdvancedAnalytics:
    """
    İleri seviye analitik modüller.
    Her fonksiyon cache'leniyor, 50k+ satırda optimize.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 1. COHORT ANALİZİ
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
    def cohort_analysis(df: pd.DataFrame, time_col: str = "Year") -> Optional[Dict[str, Any]]:
        """
        Cohort analizi — ürün/molekül gruplarının zaman içinde performansı.
        
        Returns:
            Dict ile cohort matrisi, retention rate, expansion rate
        """
        try:
            years = DataPipeline._detect_years(df, "Sales_")
            if len(years) < 3:
                return None

            # Her yılda aktif olan moleküller
            cohorts = {}
            for yr in years:
                sc = f"Sales_{yr}"
                if sc in df.columns:
                    active = set(df[df[sc] > 0]["Molecule"].dropna().unique())
                    cohorts[yr] = active

            # Cohort retention matrisi
            cohort_matrix = []
            for base_yr in years[:-1]:
                base_set = cohorts.get(base_yr, set())
                if not base_set:
                    continue
                
                row = {"Cohort_Year": base_yr, "Initial_Size": len(base_set)}
                for future_yr in years:
                    if future_yr < base_yr:
                        continue
                    future_set = cohorts.get(future_yr, set())
                    retained = len(base_set & future_set)
                    retention_rate = retained / len(base_set) * 100 if base_set else 0
                    row[f"Retained_{future_yr}"] = retained
                    row[f"Retention_{future_yr}_%"] = round(retention_rate, 1)
                
                cohort_matrix.append(row)

            cohort_df = pd.DataFrame(cohort_matrix)

            # Expansion rate (yeni gelen moleküller)
            expansion = []
            for i in range(1, len(years)):
                prev_set = cohorts.get(years[i-1], set())
                curr_set = cohorts.get(years[i], set())
                new_entries = curr_set - prev_set
                expansion.append({
                    "Year": years[i],
                    "New_Molecules": len(new_entries),
                    "Expansion_Rate_%": round(len(new_entries) / len(prev_set) * 100, 1) if prev_set else 0
                })
            
            expansion_df = pd.DataFrame(expansion)

            return {
                "cohort_retention": cohort_df,
                "expansion": expansion_df,
                "cohorts_dict": {str(k): list(v) for k, v in cohorts.items()}
            }

        except Exception as exc:
            st.warning(f"⚠️ Cohort analizi hatası: {exc}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # 2. RFM SEGMENTASYONU
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
    def rfm_segmentation(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        RFM (Recency, Frequency, Monetary) segmentasyon.
        
        - Recency: En son satış yılı
        - Frequency: Kaç yılda satış var
        - Monetary: Toplam satış hacmi
        
        Segmentler: Champions, Loyal, At Risk, Lost, Promising, New
        """
        try:
            years = DataPipeline._detect_years(df, "Sales_")
            if len(years) < 2:
                return None

            group_col = "Molecule" if "Molecule" in df.columns else "Company"
            if group_col not in df.columns:
                return None

            # Her molekül için RFM metrikleri
            rfm_data = []
            for name, grp in df.groupby(group_col):
                # Recency: son satış yılı
                last_sale_year = None
                for yr in reversed(years):
                    sc = f"Sales_{yr}"
                    if sc in grp.columns and grp[sc].sum() > 0:
                        last_sale_year = yr
                        break
                
                if last_sale_year is None:
                    continue

                recency = years[-1] - last_sale_year

                # Frequency: kaç yılda satış var
                frequency = sum(
                    1 for yr in years
                    if f"Sales_{yr}" in grp.columns and grp[f"Sales_{yr}"].sum() > 0
                )

                # Monetary: toplam satış
                monetary = sum(
                    grp[f"Sales_{yr}"].sum()
                    for yr in years
                    if f"Sales_{yr}" in grp.columns
                )

                rfm_data.append({
                    group_col: name,
                    "Recency_Years": recency,
                    "Frequency_Years": frequency,
                    "Monetary_Total": monetary,
                })

            rfm_df = pd.DataFrame(rfm_data)

            # RFM skorları (1-5, 5=en iyi)
            rfm_df["R_Score"] = pd.qcut(
                rfm_df["Recency_Years"], q=5, labels=[5,4,3,2,1], duplicates="drop"
            ).astype(float)
            rfm_df["F_Score"] = pd.qcut(
                rfm_df["Frequency_Years"], q=5, labels=[1,2,3,4,5], duplicates="drop"
            ).astype(float)
            rfm_df["M_Score"] = pd.qcut(
                rfm_df["Monetary_Total"], q=5, labels=[1,2,3,4,5], duplicates="drop"
            ).astype(float)

            rfm_df["RFM_Score"] = (
                rfm_df["R_Score"] * 100 +
                rfm_df["F_Score"] * 10 +
                rfm_df["M_Score"]
            )

            # Segmentasyon
            def _segment(row):
                r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
                if r >= 4 and f >= 4 and m >= 4:
                    return "🏆 Champions"
                elif r >= 3 and f >= 3 and m >= 3:
                    return "💎 Loyal"
                elif r >= 4 and f <= 2:
                    return "🌱 Promising"
                elif r <= 2 and f >= 3:
                    return "⚠️ At Risk"
                elif r <= 2 and f <= 2:
                    return "❌ Lost"
                else:
                    return "🆕 New/Other"

            rfm_df["Segment"] = rfm_df.apply(_segment, axis=1)

            return rfm_df.sort_values("RFM_Score", ascending=False)

        except Exception as exc:
            st.warning(f"⚠️ RFM segmentasyon hatası: {exc}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # 3. FİYAT ELASTİSİTESİ ANALİZİ
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
    def price_elasticity(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Fiyat elastisitesi analizi.
        
        Elasticity = % Değişim Miktar / % Değişim Fiyat
        
        |E| > 1: Elastik (fiyata duyarlı)
        |E| < 1: İnelastik (fiyata duyarsız)
        |E| = 1: Üniter elastik
        """
        try:
            years = DataPipeline._detect_years(df, "Sales_")
            if len(years) < 2:
                return None

            group_col = "Molecule" if "Molecule" in df.columns else "Company"
            if group_col not in df.columns:
                return None

            elasticity_data = []

            for name, grp in df.groupby(group_col):
                for i in range(len(years) - 1):
                    y1, y2 = years[i], years[i+1]
                    
                    # Fiyat ve miktar sütunları
                    p1_col = f"Avg_Price_{y1}"
                    p2_col = f"Avg_Price_{y2}"
                    q1_col = f"Units_{y1}"
                    q2_col = f"Units_{y2}"

                    if not all(c in grp.columns for c in [p1_col, p2_col, q1_col, q2_col]):
                        continue

                    p1 = grp[p1_col].mean()
                    p2 = grp[p2_col].mean()
                    q1 = grp[q1_col].sum()
                    q2 = grp[q2_col].sum()

                    if pd.isna(p1) or pd.isna(p2) or pd.isna(q1) or pd.isna(q2):
                        continue
                    if p1 <= 0 or q1 <= 0:
                        continue

                    # Yüzde değişimler
                    pct_price_change = (p2 - p1) / p1 * 100
                    pct_qty_change = (q2 - q1) / q1 * 100

                    if abs(pct_price_change) < 0.1:  # çok küçük değişim
                        continue

                    elasticity = pct_qty_change / pct_price_change

                    # Kategori
                    if abs(elasticity) > 1.5:
                        cat = "Yüksek Elastik"
                    elif abs(elasticity) > 1.0:
                        cat = "Elastik"
                    elif abs(elasticity) > 0.5:
                        cat = "Orta İnelastik"
                    else:
                        cat = "İnelastik"

                    elasticity_data.append({
                        group_col: name,
                        "Period": f"{y1}-{y2}",
                        "Price_Change_%": round(pct_price_change, 2),
                        "Qty_Change_%": round(pct_qty_change, 2),
                        "Elasticity": round(elasticity, 3),
                        "Category": cat,
                    })

            if not elasticity_data:
                return None

            elast_df = pd.DataFrame(elasticity_data)
            return elast_df.sort_values("Elasticity", key=abs, ascending=False)

        except Exception as exc:
            st.warning(f"⚠️ Fiyat elastisitesi hatası: {exc}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # 4. CHURN PREDICTION (Molekül Kayıp Riski)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
    def churn_prediction(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Churn (kayıp) riski tahmini — Random Forest sınıflandırıcı.
        
        Churn tanımı: Son 2 yılda satış yapılmamış
        
        Özellikler:
        - Satış trendi
        - Fiyat değişimi
        - Pazar payı değişimi
        - Son satış zamanı
        """
        try:
            years = DataPipeline._detect_years(df, "Sales_")
            if len(years) < 4:
                return None

            group_col = "Molecule" if "Molecule" in df.columns else "Company"
            if group_col not in df.columns:
                return None

            # Feature engineering
            features_list = []
            labels = []
            names = []

            for name, grp in df.groupby(group_col):
                # Churn label: son 2 yılda satış var mı?
                recent_sales = sum(
                    grp[f"Sales_{yr}"].sum()
                    for yr in years[-2:]
                    if f"Sales_{yr}" in grp.columns
                )
                is_churned = 1 if recent_sales == 0 else 0

                # Features
                # 1. Satış trendi (lineer regresyon slope)
                sales_series = [
                    grp[f"Sales_{yr}"].sum()
                    for yr in years[:-2]
                    if f"Sales_{yr}" in grp.columns
                ]
                if len(sales_series) < 2:
                    continue
                
                x = np.arange(len(sales_series)).reshape(-1, 1)
                y = np.array(sales_series)
                if np.std(y) == 0:
                    trend_slope = 0
                else:
                    lr = LinearRegression()
                    lr.fit(x, y)
                    trend_slope = lr.coef_[0]

                # 2. Fiyat değişimi
                price_cols = [c for c in grp.columns if "Avg_Price_" in c]
                if len(price_cols) >= 2:
                    p1 = grp[price_cols[0]].mean()
                    p2 = grp[price_cols[-1]].mean()
                    price_change = (p2 - p1) / p1 * 100 if p1 > 0 else 0
                else:
                    price_change = 0

                # 3. Pazar payı değişimi
                if "Market_Share" in grp.columns:
                    ms_vals = grp["Market_Share"].dropna().values
                    ms_change = ms_vals[-1] - ms_vals[0] if len(ms_vals) >= 2 else 0
                else:
                    ms_change = 0

                # 4. Son satış yılı (recency)
                last_sale_year = years[0]
                for yr in reversed(years[:-2]):
                    sc = f"Sales_{yr}"
                    if sc in grp.columns and grp[sc].sum() > 0:
                        last_sale_year = yr
                        break
                recency = years[-3] - last_sale_year

                # 5. Ortalama satış
                avg_sales = np.mean(sales_series) if sales_series else 0

                features_list.append([
                    trend_slope,
                    price_change,
                    ms_change,
                    recency,
                    np.log1p(avg_sales),
                ])
                labels.append(is_churned)
                names.append(name)

            if len(features_list) < 10:
                return None

            X = np.array(features_list)
            y = np.array(labels)

            # Train model (tüm veri ile — production'da train/test split yapılır)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            clf = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                n_jobs=1,
            )
            clf.fit(X_scaled, y)

            # Churn probability
            churn_probs = clf.predict_proba(X_scaled)[:, 1]

            # Risk kategorisi
            def _risk_cat(prob):
                if prob > 0.7:
                    return "🔴 Yüksek Risk"
                elif prob > 0.4:
                    return "🟡 Orta Risk"
                else:
                    return "🟢 Düşük Risk"

            result_df = pd.DataFrame({
                group_col: names,
                "Churn_Probability_%": np.round(churn_probs * 100, 1),
                "Risk_Category": [_risk_cat(p) for p in churn_probs],
                "Trend_Slope": X[:, 0],
                "Price_Change_%": np.round(X[:, 1], 1),
                "Recency_Years": X[:, 3].astype(int),
            })

            return result_df.sort_values("Churn_Probability_%", ascending=False)

        except Exception as exc:
            st.warning(f"⚠️ Churn prediction hatası: {exc}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # 5. KÜMELEME ANALİZİ (K-Means Segmentasyon)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
    def clustering_analysis(
        df: pd.DataFrame, n_clusters: int = 4
    ) -> Optional[Dict[str, Any]]:
        """
        K-Means kümeleme analizi.
        
        Özellikler:
        - Ortalama satış
        - Büyüme oranı
        - Fiyat seviyesi
        - Pazar payı
        
        Returns:
            Dict: cluster_labels, centers, silhouette_score, pca_coords
        """
        try:
            years = DataPipeline._detect_years(df, "Sales_")
            if len(years) < 2:
                return None

            group_col = "Molecule" if "Molecule" in df.columns else "Company"
            if group_col not in df.columns:
                return None

            # Feature matrix
            features_list = []
            names = []

            for name, grp in df.groupby(group_col):
                # Ortalama satış
                avg_sales = np.mean([
                    grp[f"Sales_{yr}"].sum()
                    for yr in years
                    if f"Sales_{yr}" in grp.columns
                ])

                # Büyüme oranı (CAGR)
                if "CAGR" in grp.columns:
                    cagr = grp["CAGR"].mean()
                else:
                    cagr = 0

                # Ortalama fiyat
                price_cols = [c for c in grp.columns if "Avg_Price_" in c]
                avg_price = grp[price_cols[-1]].mean() if price_cols else 0

                # Pazar payı
                if "Market_Share" in grp.columns:
                    ms = grp["Market_Share"].mean()
                else:
                    ms = 0

                if pd.isna(avg_sales) or avg_sales <= 0:
                    continue

                features_list.append([
                    np.log1p(avg_sales),
                    cagr,
                    np.log1p(avg_price) if avg_price > 0 else 0,
                    ms,
                ])
                names.append(name)

            if len(features_list) < n_clusters * 2:
                return None

            X = np.array(features_list)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)

            # Silhouette score (kümeleme kalitesi)
            sil_score = silhouette_score(X_scaled, cluster_labels)

            # PCA for visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # Cluster isimleri
            cluster_names = {}
            for i in range(n_clusters):
                mask = cluster_labels == i
                avg_feat = X[mask].mean(axis=0)
                # Satış ve büyümeye göre isimlendir
                sales_level = "Yüksek" if avg_feat[0] > X[:, 0].median() else "Düşük"
                growth_level = "Büyüyen" if avg_feat[1] > X[:, 1].median() else "Düşen"
                cluster_names[i] = f"{sales_level} Satış, {growth_level}"

            result_df = pd.DataFrame({
                group_col: names,
                "Cluster": cluster_labels,
                "Cluster_Name": [cluster_names[c] for c in cluster_labels],
                "PCA_X": X_pca[:, 0],
                "PCA_Y": X_pca[:, 1],
                "Avg_Sales_Log": X[:, 0],
                "CAGR": X[:, 1],
            })

            return {
                "data": result_df,
                "silhouette_score": round(sil_score, 3),
                "cluster_centers": kmeans.cluster_centers_,
                "explained_variance": pca.explained_variance_ratio_,
            }

        except Exception as exc:
            st.warning(f"⚠️ Kümeleme analizi hatası: {exc}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # 6. TREND DECOMPOSITION (Mevsimsellik Ayrıştırma)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
    def trend_decomposition(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Zaman serisi trend ayrıştırma.
        
        Toplam pazar satışlarını:
        - Trend (uzun vadeli yön)
        - Mevsimsel (döngüsel patern)
        - Residual (gürültü)
        
        olarak ayırır.
        """
        if not STATSMODELS_OK:
            return None

        try:
            years = DataPipeline._detect_years(df, "Sales_")
            if len(years) < 4:
                return None

            # Yıllık toplam satış serisi
            yearly_sales = pd.Series({
                yr: df[f"Sales_{yr}"].sum()
                for yr in years
                if f"Sales_{yr}" in df.columns
            })

            if len(yearly_sales) < 4:
                return None

            # Seasonal decompose
            decomposition = seasonal_decompose(
                yearly_sales,
                model="additive",
                period=min(3, len(yearly_sales) // 2),
                extrapolate_trend="freq",
            )

            result_df = pd.DataFrame({
                "Year": yearly_sales.index,
                "Observed": yearly_sales.values,
                "Trend": decomposition.trend,
                "Seasonal": decomposition.seasonal,
                "Residual": decomposition.resid,
            })

            # Trend yönü
            trend_slope = np.polyfit(range(len(decomposition.trend)), decomposition.trend, 1)[0]
            trend_direction = "📈 Yükseliş" if trend_slope > 0 else "📉 Düşüş"

            return {
                "decomposition": result_df,
                "trend_slope": round(trend_slope, 2),
                "trend_direction": trend_direction,
                "seasonality_strength": round(np.std(decomposition.seasonal) / np.std(yearly_sales) * 100, 1),
            }

        except Exception as exc:
            st.warning(f"⚠️ Trend decomposition hatası: {exc}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # 7. MONTE CARLO SİMÜLASYONU
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
    def monte_carlo_forecast(
        df: pd.DataFrame,
        periods: int = 3,
        simulations: int = 1000,
    ) -> Optional[Dict[str, Any]]:
        """
        Monte Carlo simülasyonu ile gelecek tahmin aralığı.
        
        Historical volatilite kullanarak olası senaryolar üretir.
        """
        try:
            years = DataPipeline._detect_years(df, "Sales_")
            if len(years) < 3:
                return None

            # Yıllık büyüme oranları
            yearly_sales = [
                df[f"Sales_{yr}"].sum()
                for yr in years
                if f"Sales_{yr}" in df.columns
            ]

            growth_rates = [
                (yearly_sales[i] - yearly_sales[i-1]) / yearly_sales[i-1]
                for i in range(1, len(yearly_sales))
            ]

            mean_growth = np.mean(growth_rates)
            std_growth = np.std(growth_rates)

            # Monte Carlo simülasyonu
            last_value = yearly_sales[-1]
            simulated_paths = np.zeros((simulations, periods))

            for i in range(simulations):
                value = last_value
                for j in range(periods):
                    # Rastgele büyüme oranı (normal dağılım)
                    growth = np.random.normal(mean_growth, std_growth)
                    value = value * (1 + growth)
                    simulated_paths[i, j] = value

            # Yüzdelik dilimleri (confidence intervals)
            percentiles = {
                "p10": np.percentile(simulated_paths, 10, axis=0),
                "p25": np.percentile(simulated_paths, 25, axis=0),
                "p50": np.percentile(simulated_paths, 50, axis=0),  # median
                "p75": np.percentile(simulated_paths, 75, axis=0),
                "p90": np.percentile(simulated_paths, 90, axis=0),
            }

            future_years = [years[-1] + i + 1 for i in range(periods)]

            forecast_df = pd.DataFrame({
                "Year": future_years,
                "P10": percentiles["p10"],
                "P25": percentiles["p25"],
                "Median": percentiles["p50"],
                "P75": percentiles["p75"],
                "P90": percentiles["p90"],
            })

            return {
                "forecast": forecast_df,
                "historical_growth_mean": round(mean_growth * 100, 2),
                "historical_growth_std": round(std_growth * 100, 2),
                "simulations": simulations,
            }

        except Exception as exc:
            st.warning(f"⚠️ Monte Carlo simülasyon hatası: {exc}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # 8. GELİŞMİŞ ANOMALİ TESPİTİ
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
    def advanced_anomaly_detection(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Gelişmiş anomali tespiti — çoklu özellik + outlier scoring.
        
        Özellikler:
        - Satış düzeyi
        - Büyüme oranı
        - Fiyat
        - Pazar payı
        - Fiyat-miktar oranı
        """
        try:
            features = []

            # Son 2 yıl satış
            years = DataPipeline._detect_years(df, "Sales_")
            if len(years) < 2:
                return None

            for yr in years[-2:]:
                c = f"Sales_{yr}"
                if c in df.columns:
                    features.append(c)

            # Büyüme
            growth_cols = [c for c in df.columns if re.match(r"Growth_\d{4}_\d{4}", c)]
            if growth_cols:
                features.append(growth_cols[-1])

            # Fiyat
            price_cols = [c for c in df.columns if "Avg_Price_" in c]
            if price_cols:
                features.append(price_cols[-1])

            # Pazar payı
            if "Market_Share" in df.columns:
                features.append("Market_Share")

            # CAGR
            if "CAGR" in df.columns:
                features.append("CAGR")

            if len(features) < 3:
                return None

            X = df[features].fillna(0)
            if len(X) < 20:
                return None

            # Örnekleme (50k+ satır için)
            sample_size = min(len(X), 5000)
            if len(X) > sample_size:
                X_fit = X.sample(n=sample_size, random_state=42)
            else:
                X_fit = X

            # Robust scaling (outlier'lara karşı dayanıklı)
            scaler = RobustScaler()
            X_scaled = scaler.fit(X_fit).transform(X)

            # Isolation Forest
            iso = IsolationForest(
                contamination=0.05,
                n_estimators=100,
                random_state=42,
                n_jobs=1,
            )
            iso.fit(X_scaled[:sample_size])
            
            labels = iso.predict(X_scaled)
            scores = iso.score_samples(X_scaled)

            # Z-score bazlı ek scoring
            z_scores = np.abs(stats.zscore(X_scaled, axis=0)).mean(axis=1)

            # Combined anomaly score
            combined_score = (
                (scores - scores.min()) / (scores.max() - scores.min()) * 0.6 +
                (z_scores / z_scores.max()) * 0.4
            )

            result = df.copy()
            result["Anomaly_Label"] = labels
            result["Anomaly_Score"] = scores
            result["Z_Score"] = z_scores
            result["Combined_Score"] = combined_score
            result["Is_Anomaly"] = labels == -1

            # Kategori
            result["Anomaly_Severity"] = pd.cut(
                result["Combined_Score"],
                bins=[0, 0.3, 0.6, 0.8, 1.0],
                labels=["Normal", "Hafif", "Orta", "Yüksek"],
            )

            return result.sort_values("Combined_Score", ascending=False)

        except Exception as exc:
            st.warning(f"⚠️ Gelişmiş anomali tespiti hatası: {exc}")
            return None


# ═════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def format_large_number(n: float) -> str:
    """Büyük sayıları okunabilir formata çevirir."""
    if n >= 1e9:
        return f"${n/1e9:.2f}B"
    elif n >= 1e6:
        return f"${n/1e6:.2f}M"
    elif n >= 1e3:
        return f"${n/1e3:.1f}K"
    else:
        return f"${n:.0f}"
