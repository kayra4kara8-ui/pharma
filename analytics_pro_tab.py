"""
Gelişmiş Analitik sekmesi için render fonksiyonu.
app.py'ye eklenecek.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from analytics_pro import AdvancedAnalytics
from visualizer import EnterpriseVisualizer
from core import section_title, insight_card


def render_advanced_analytics_tab(df: pd.DataFrame) -> None:
    """🔬 Gelişmiş Analitik sekmesini render eder."""
    try:
        section_title("🔬 Gelişmiş Analitik & Makine Öğrenmesi")

        st.info(
            "ℹ️ **Enterprise Özellik:** Bu analizler ileri seviye istatistiksel modeller "
            "ve makine öğrenmesi algoritmalarıkullanır. İlk çalıştırmada 15-45 saniye sürebilir."
        )

        # Ana sekmeler
        analysis_tabs = st.tabs([
            "👥 Cohort Analizi",
            "💎 RFM Segmentasyon",
            "💰 Fiyat Elastisitesi",
            "⚠️ Churn Prediction",
            "🎯 Kümeleme",
            "📈 Trend Decomposition",
            "🎲 Monte Carlo",
            "🔍 Gelişmiş Anomali",
        ])

        # ══════════════════════════════════════════════════════════════════════
        # 1. COHORT ANALİZİ
        # ══════════════════════════════════════════════════════════════════════

        with analysis_tabs[0]:
            st.markdown("### 👥 Cohort (Kohort) Analizi")
            insight_card(
                "Ürün/molekül gruplarının zaman içinde nasıl performans gösterdiğini analiz eder. "
                "Retention rate ve expansion rate metrikleri ile portföy sağlığını ölçer.",
                "info", "Cohort Analizi Nedir?"
            )

            if st.button("▶️ Cohort Analizi Çalıştır", key="btn_cohort", type="primary"):
                with st.spinner("Cohort analizi hesaplanıyor..."):
                    result = AdvancedAnalytics.cohort_analysis(df)
                    st.session_state["cohort_result"] = result

            result = st.session_state.get("cohort_result")
            if result:
                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown("**📊 Retention Matrisi**")
                    if "cohort_retention" in result:
                        st.dataframe(
                            result["cohort_retention"],
                            use_container_width=True, hide_index=True
                        )
                
                with c2:
                    st.markdown("**🌱 Expansion (Yeni Girişler)**")
                    if "expansion" in result:
                        st.dataframe(
                            result["expansion"],
                            use_container_width=True, hide_index=True
                        )

        # ══════════════════════════════════════════════════════════════════════
        # 2. RFM SEGMENTASYONU
        # ══════════════════════════════════════════════════════════════════════

        with analysis_tabs[1]:
            st.markdown("### 💎 RFM Segmentasyonu")
            insight_card(
                "**R**ecency (ne kadar yeni), **F**requency (ne kadar sık), **M**onetary (ne kadar değerli) "
                "metriklerine göre ürün/molekülleri segmentlere ayırır: Champions, Loyal, At Risk, Lost, vb.",
                "info", "RFM Segmentasyon Nedir?"
            )

            if st.button("▶️ RFM Segmentasyon Çalıştır", key="btn_rfm", type="primary"):
                with st.spinner("RFM skorları hesaplanıyor..."):
                    rfm_df = AdvancedAnalytics.rfm_segmentation(df)
                    st.session_state["rfm_df"] = rfm_df

            rfm_df = st.session_state.get("rfm_df")
            if rfm_df is not None and not rfm_df.empty:
                # Segment dağılımı
                c1, c2 = st.columns([2, 1])
                
                with c1:
                    st.markdown("**🎯 Segment Dağılımı**")
                    seg_counts = rfm_df["Segment"].value_counts().reset_index()
                    seg_counts.columns = ["Segment", "Adet"]
                    st.dataframe(seg_counts, use_container_width=True, hide_index=True)
                
                with c2:
                    st.markdown("**📊 Top 10 RFM Skorları**")
                    top10 = rfm_df.head(10)[[
                        rfm_df.columns[0], "RFM_Score", "Segment"
                    ]]
                    st.dataframe(top10, use_container_width=True, hide_index=True)

                # Tam tablo
                st.markdown("---")
                st.markdown("**📋 Detaylı RFM Tablosu**")
                st.dataframe(rfm_df, use_container_width=True, hide_index=True)

        # ══════════════════════════════════════════════════════════════════════
        # 3. FİYAT ELASTİSİTESİ
        # ══════════════════════════════════════════════════════════════════════

        with analysis_tabs[2]:
            st.markdown("### 💰 Fiyat Elastisitesi Analizi")
            insight_card(
                "Fiyat değişimlerinin miktar/satış üzerindeki etkisini ölçer. "
                "|E| > 1 = Elastik (fiyata duyarlı), |E| < 1 = İnelastik (fiyata duyarsız).",
                "warning", "Elastisite Nedir?"
            )

            if st.button("▶️ Elastisite Analizi Çalıştır", key="btn_elasticity", type="primary"):
                with st.spinner("Fiyat elastisitesi hesaplanıyor..."):
                    elast_df = AdvancedAnalytics.price_elasticity(df)
                    st.session_state["elasticity_df"] = elast_df

            elast_df = st.session_state.get("elasticity_df")
            if elast_df is not None and not elast_df.empty:
                c1, c2 = st.columns([3, 1])
                
                with c1:
                    st.dataframe(elast_df, use_container_width=True, hide_index=True)
                
                with c2:
                    st.markdown("**📊 Kategori Dağılımı**")
                    cat_dist = elast_df["Category"].value_counts().reset_index()
                    cat_dist.columns = ["Kategori", "Adet"]
                    st.dataframe(cat_dist, use_container_width=True, hide_index=True)

        # ══════════════════════════════════════════════════════════════════════
        # 4. CHURN PREDICTION
        # ══════════════════════════════════════════════════════════════════════

        with analysis_tabs[3]:
            st.markdown("### ⚠️ Churn (Kayıp) Riski Tahmini")
            insight_card(
                "Random Forest sınıflandırıcı ile hangi ürün/moleküllerin kaybolma riski taşıdığını tahmin eder. "
                "Satış trendi, fiyat değişimi, recency gibi özellikleri kullanır.",
                "danger", "Churn Prediction Nedir?"
            )

            if st.button("▶️ Churn Prediction Çalıştır", key="btn_churn", type="primary"):
                with st.spinner("Churn riski hesaplanıyor... (ML modeli eğitiliyor)"):
                    churn_df = AdvancedAnalytics.churn_prediction(df)
                    st.session_state["churn_df"] = churn_df

            churn_df = st.session_state.get("churn_df")
            if churn_df is not None and not churn_df.empty:
                # Risk dağılımı
                c1, c2 = st.columns([2, 1])
                
                with c1:
                    st.markdown("**🔴 Yüksek Riskli Ürünler (Top 20)**")
                    high_risk = churn_df[churn_df["Risk_Category"] == "🔴 Yüksek Risk"].head(20)
                    st.dataframe(high_risk, use_container_width=True, hide_index=True)
                
                with c2:
                    st.markdown("**📊 Risk Kategorisi Dağılımı**")
                    risk_dist = churn_df["Risk_Category"].value_counts().reset_index()
                    risk_dist.columns = ["Risk", "Adet"]
                    st.dataframe(risk_dist, use_container_width=True, hide_index=True)

        # ══════════════════════════════════════════════════════════════════════
        # 5. KÜMELEME ANALİZİ
        # ══════════════════════════════════════════════════════════════════════

        with analysis_tabs[4]:
            st.markdown("### 🎯 K-Means Kümeleme Analizi")
            insight_card(
                "Ürün/molekülleri satış, büyüme, fiyat ve pazar payı özelliklerine göre "
                "otomatik olarak gruplara ayırır. Stratejik segmentasyon için kullanılır.",
                "info", "Kümeleme Nedir?"
            )

            n_clusters = st.slider("Küme Sayısı", 2, 8, 4, key="n_clusters")
            
            if st.button("▶️ Kümeleme Analizi Çalıştır", key="btn_clustering", type="primary"):
                with st.spinner(f"K-Means kümeleme ({n_clusters} cluster) hesaplanıyor..."):
                    cluster_result = AdvancedAnalytics.clustering_analysis(df, n_clusters)
                    st.session_state["cluster_result"] = cluster_result

            cluster_result = st.session_state.get("cluster_result")
            if cluster_result:
                st.markdown(f"**✅ Silhouette Score: {cluster_result['silhouette_score']}** "
                           "(0-1 arası, 1'e yakın = daha iyi kümeleme)")
                
                cluster_df = cluster_result["data"]
                
                # Cluster dağılımı
                c1, c2 = st.columns([1, 2])
                
                with c1:
                    st.markdown("**📊 Küme Dağılımı**")
                    dist = cluster_df["Cluster_Name"].value_counts().reset_index()
                    dist.columns = ["Küme", "Adet"]
                    st.dataframe(dist, use_container_width=True, hide_index=True)
                
                with c2:
                    st.markdown("**🗺️ PCA Görselleştirme**")
                    # Scatter plot — interaktif grafik için plotly kullanılabilir
                    st.dataframe(
                        cluster_df[["Cluster_Name", "PCA_X", "PCA_Y"]].head(100),
                        use_container_width=True, hide_index=True
                    )

        # ══════════════════════════════════════════════════════════════════════
        # 6. TREND DECOMPOSITION
        # ══════════════════════════════════════════════════════════════════════

        with analysis_tabs[5]:
            st.markdown("### 📈 Trend Decomposition (Ayrıştırma)")
            insight_card(
                "Zaman serisi verisini Trend + Mevsimsel + Residual bileşenlerine ayırır. "
                "Uzun vadeli eğilim ve döngüsel patternleri görselleştirir.",
                "info", "Trend Decomposition Nedir?"
            )

            if st.button("▶️ Trend Decomposition Çalıştır", key="btn_decomp", type="primary"):
                with st.spinner("Trend ayrıştırması hesaplanıyor..."):
                    decomp_result = AdvancedAnalytics.trend_decomposition(df)
                    st.session_state["decomp_result"] = decomp_result

            decomp_result = st.session_state.get("decomp_result")
            if decomp_result:
                st.markdown(f"**{decomp_result['trend_direction']}** "
                           f"(Slope: {decomp_result['trend_slope']})")
                st.markdown(f"**Mevsimsellik Gücü:** {decomp_result['seasonality_strength']}%")
                
                st.markdown("**📊 Decomposition Tablosu**")
                st.dataframe(
                    decomp_result["decomposition"],
                    use_container_width=True, hide_index=True
                )

        # ══════════════════════════════════════════════════════════════════════
        # 7. MONTE CARLO SİMÜLASYONU
        # ══════════════════════════════════════════════════════════════════════

        with analysis_tabs[6]:
            st.markdown("### 🎲 Monte Carlo Simülasyon Tahmini")
            insight_card(
                "Historical volatilite kullanarak 1000 olası gelecek senaryosu üretir. "
                "Confidence interval'ler (P10, P25, Median, P75, P90) ile risk aralığı gösterir.",
                "info", "Monte Carlo Nedir?"
            )

            periods = st.slider("Tahmin Dönemi (Yıl)", 1, 5, 3, key="mc_periods")
            
            if st.button("▶️ Monte Carlo Simülasyon Çalıştır", key="btn_mc", type="primary"):
                with st.spinner(f"1000 simülasyon çalıştırılıyor..."):
                    mc_result = AdvancedAnalytics.monte_carlo_forecast(df, periods)
                    st.session_state["mc_result"] = mc_result

            mc_result = st.session_state.get("mc_result")
            if mc_result:
                st.markdown(f"**Historical Ortalama Büyüme:** {mc_result['historical_growth_mean']}% "
                           f"± {mc_result['historical_growth_std']}%")
                
                st.markdown("**📊 Tahmin Aralıkları**")
                forecast_df = mc_result["forecast"]
                
                # Formatla
                for col in ["P10", "P25", "Median", "P75", "P90"]:
                    forecast_df[col] = forecast_df[col].apply(
                        lambda v: f"${v/1e6:.2f}M" if v >= 1e6 else f"${v:,.0f}"
                    )
                
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        # ══════════════════════════════════════════════════════════════════════
        # 8. GELİŞMİŞ ANOMALİ TESPİTİ
        # ══════════════════════════════════════════════════════════════════════

        with analysis_tabs[7]:
            st.markdown("### 🔍 Gelişmiş Anomali Tespiti")
            insight_card(
                "Çoklu özellik analizi + Z-score + Isolation Forest kombinasyonu ile "
                "anormal davranış gösteren ürünleri tespit eder. Combined Score ile sıralanır.",
                "warning", "Gelişmiş Anomali Tespiti"
            )

            if st.button("▶️ Gelişmiş Anomali Tespiti Çalıştır", key="btn_adv_anom", type="primary"):
                with st.spinner("Gelişmiş anomali analizi çalıştırılıyor..."):
                    anom_df = AdvancedAnalytics.advanced_anomaly_detection(df)
                    st.session_state["advanced_anom_df"] = anom_df

            anom_df = st.session_state.get("advanced_anom_df")
            if anom_df is not None and not anom_df.empty:
                # Özet
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    n_anom = int(anom_df["Is_Anomaly"].sum())
                    st.metric("Anomali Tespit Edilen", f"{n_anom:,}")
                
                with c2:
                    if "Anomaly_Severity" in anom_df.columns:
                        high_sev = len(anom_df[anom_df["Anomaly_Severity"] == "Yüksek"])
                        st.metric("Yüksek Şiddet", f"{high_sev:,}")
                
                with c3:
                    pct = n_anom / len(anom_df) * 100
                    st.metric("Anomali Oranı", f"{pct:.1f}%")

                # Top anomaliler
                st.markdown("**🚨 En Yüksek Anomali Skorları (Top 30)**")
                top_anom = anom_df[anom_df["Is_Anomaly"]].head(30)
                
                group_col = next(
                    (c for c in ["Molecule", "Company"] if c in top_anom.columns), None
                )
                display_cols = [group_col] if group_col else []
                display_cols += ["Combined_Score", "Anomaly_Severity"]
                
                if display_cols:
                    st.dataframe(
                        top_anom[display_cols],
                        use_container_width=True, hide_index=True
                    )

    except Exception as exc:
        st.error(f"❌ Gelişmiş Analitik sekmesi hatası: {exc}")
        import traceback
        st.code(traceback.format_exc())
