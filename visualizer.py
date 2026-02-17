"""
PharmaIntelligence Enterprise v8.0 — visualizer.py
────────────────────────────────────────────────────
Modüller:
  • EnterpriseVisualizer : Sankey, Waterfall, BCG, Fiyat Erozyonu,
                           EI, HHI, Tahmin, Anomali, Treemap, Kanibalizasyon
  • ReportGenerator      : Multi-sheet Excel, 10 sayfalık PDF, HTML raporu

Düzeltilen hata:
  ✅ _theme() içinde update_layout() çift 'title' keyword hatası giderildi.
     THEME dict'inden 'title' anahtarı kaldırıldı; başlık ayrı parametre olarak geçiliyor.
"""

import re
import traceback
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        HRFlowable, PageBreak, Paragraph, SimpleDocTemplate,
        Spacer, Table, TableStyle,
    )
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

from core import DataPipeline, fmt_currency


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 7 — ENTERPRISE VISUALIZER
# ─────────────────────────────────────────────────────────────────────────────

class EnterpriseVisualizer:
    """
    PharmaIntelligence v8.0 için profesyonel Plotly grafik fabrikası.

    Karanlık kurumsal tema tüm grafiklere otomatik uygulanır.

    DÜZELTME: THEME dict'inden 'title' anahtarı kaldırıldı.
    _theme() metodu title'ı ayrı olarak update_layout()'a geçiriyor,
    böylece 'got multiple values for keyword argument title' hatası oluşmuyor.
    """

    # ── 'title' buradan KALDIRILDI — ayrı parametre olarak geçiliyor ────────
    THEME = dict(
        paper_bgcolor="rgba(9,20,43,0)",
        plot_bgcolor="rgba(9,20,43,0)",
        font=dict(family="Sora, DM Sans, sans-serif", color="#e8f0fe", size=12),
        # title dict'i THEME'den çıkarıldı — _theme() içinde ayrıca set ediliyor
        legend=dict(
            bgcolor="rgba(17,37,72,0.6)",
            bordercolor="rgba(0,212,255,0.2)",
            borderwidth=1,
        ),
        colorway=[
            "#00d4ff", "#0070e0", "#7b2fff", "#00e5a0",
            "#ffb700", "#ff4757", "#a8e6cf", "#ff8b94",
        ],
    )

    GRID = dict(
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.15)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.15)"),
    )

    @classmethod
    def _theme(cls, fig: go.Figure, title: str = "") -> go.Figure:
        """
        Kurumsal karanlık temayı uygular.

        ÖNEMLİ: title ayrı keyword olarak set ediliyor,
        THEME dict'inde title anahtarı yok — çift keyword hatası önleniyor.
        """
        fig.update_layout(
            **cls.THEME,
            **cls.GRID,
            title=dict(text=title, font=dict(size=17, color="#e8f0fe"), x=0.02),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig

    # ── 7.1 Sankey Diyagramı ─────────────────────────────────────────────────

    @classmethod
    def sankey_chart(cls, df: pd.DataFrame) -> Optional[go.Figure]:
        """
        3 seviyeli Sankey: Şirket → Molekül → Sektör

        Returns: Plotly Figure veya None
        """
        try:
            years = DataPipeline._detect_years(df, "Sales_")
            if not years:
                return None
            lsc = f"Sales_{years[-1]}"
            if lsc not in df.columns:
                return None

            has_mol  = "Molecule" in df.columns
            has_comp = "Company" in df.columns
            has_sec  = "Sector" in df.columns

            if not has_comp or not (has_mol or has_sec):
                return None

            grp_cols = [c for c in ["Company", "Molecule", "Sector"] if c in df.columns]
            agg = df.groupby(grp_cols, observed=False)[lsc].sum().reset_index()
            agg = agg[agg[lsc] > 0].copy()

            top_comp = agg.groupby("Company")[lsc].sum().nlargest(10).index.tolist()
            agg = agg[agg["Company"].isin(top_comp)]

            labels: List[str] = []
            colors_list: List[str] = []
            source_ids: List[int] = []
            target_ids: List[int] = []
            values: List[float] = []

            comp_idx: Dict[str, int] = {}
            for c in agg["Company"].unique():
                comp_idx[str(c)] = len(labels)
                labels.append(str(c))
                colors_list.append("rgba(0,112,224,0.8)")

            mol_idx: Dict[str, int] = {}
            if has_mol:
                for m in agg["Molecule"].unique():
                    mol_idx[str(m)] = len(labels)
                    labels.append(str(m))
                    colors_list.append("rgba(0,212,255,0.7)")

            sec_idx: Dict[str, int] = {}
            if has_sec:
                for s in agg["Sector"].unique():
                    sec_idx[str(s)] = len(labels)
                    labels.append(str(s))
                    colors_list.append("rgba(123,47,255,0.7)")

            for _, row in agg.iterrows():
                sales = float(row[lsc])
                comp  = str(row["Company"])
                mol   = str(row["Molecule"]) if has_mol and "Molecule" in row else None
                sec   = str(row["Sector"])   if has_sec  and "Sector"   in row else None

                if has_mol and mol and comp in comp_idx and mol in mol_idx:
                    source_ids.append(comp_idx[comp])
                    target_ids.append(mol_idx[mol])
                    values.append(sales)

                if has_sec and has_mol and mol and sec and mol in mol_idx and sec in sec_idx:
                    source_ids.append(mol_idx[mol])
                    target_ids.append(sec_idx[sec])
                    values.append(sales)
                elif has_sec and not has_mol and sec and comp in comp_idx and sec in sec_idx:
                    source_ids.append(comp_idx[comp])
                    target_ids.append(sec_idx[sec])
                    values.append(sales)

            if not source_ids:
                return None

            fig = go.Figure(go.Sankey(
                node=dict(
                    pad=15, thickness=20,
                    label=labels, color=colors_list,
                    line=dict(color="rgba(255,255,255,0.1)", width=0.5),
                ),
                link=dict(
                    source=source_ids, target=target_ids,
                    value=values, color="rgba(0,212,255,0.2)",
                ),
            ))
            return cls._theme(fig, f"💰 Nakit Akışı: Şirket → Molekül → Sektör ({years[-1]})")

        except Exception as exc:
            st.warning(f"⚠️ Sankey hatası: {exc}")
            return None

    # ── 7.2 Waterfall / Satış Köprüsü ───────────────────────────────────────

    @classmethod
    def waterfall_chart(cls, bridge_df: pd.DataFrame) -> Optional[go.Figure]:
        """
        Fiyat ve Hacim etkisini gösteren Satış Köprüsü grafiği.

        Returns: Plotly Figure veya None
        """
        try:
            if bridge_df is None or bridge_df.empty:
                return None

            group_col = next(
                (c for c in ["Molecule", "Company"] if c in bridge_df.columns), None
            )
            if group_col is None:
                return None

            top = bridge_df.nlargest(12, "Satış_Değişimi")
            cats = list(top[group_col].astype(str))
            vol  = top["Hacim_Etkisi"].fillna(0).tolist()
            price = top["Fiyat_Etkisi"].fillna(0).tolist()
            totals = top["Satış_Değişimi"].fillna(0).tolist()

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="📦 Hacim Etkisi", x=cats, y=vol,
                marker_color="rgba(0,229,160,0.8)",
                text=[f"${v/1e6:.1f}M" for v in vol], textposition="outside",
            ))
            fig.add_trace(go.Bar(
                name="💲 Fiyat Etkisi", x=cats, y=price,
                marker_color="rgba(255,183,0,0.8)",
                text=[f"${v/1e6:.1f}M" for v in price], textposition="outside",
            ))
            fig.add_trace(go.Scatter(
                name="Δ Toplam", x=cats, y=totals,
                mode="lines+markers",
                line=dict(color="#00d4ff", width=2, dash="dot"),
                marker=dict(size=8, color="#00d4ff"),
            ))
            fig.update_layout(barmode="relative")
            return cls._theme(fig, "📊 Satış Köprüsü: Hacim & Fiyat Etkisi")

        except Exception as exc:
            st.warning(f"⚠️ Waterfall hatası: {exc}")
            return None

    # ── 7.3 BCG Kuadrant ─────────────────────────────────────────────────────

    @classmethod
    def bcg_chart(cls, bcg_df: pd.DataFrame) -> Optional[go.Figure]:
        """
        BCG baloncuk grafiği: Pazar Büyümesi (Y) vs. Pazar Payı % (X).

        Returns: Plotly Figure veya None
        """
        try:
            if bcg_df is None or bcg_df.empty:
                return None

            group_col = next(
                (c for c in ["Molecule", "Company"] if c in bcg_df.columns), None
            )
            if group_col is None:
                return None

            color_map = {
                "⭐ Yıldız": "#00d4ff",
                "💰 Nakit İneği": "#00e5a0",
                "❓ Soru İşareti": "#ffb700",
                "🐕 Köpek": "#ff4757",
            }

            fig = px.scatter(
                bcg_df,
                x="Pazar_Payı_Pct", y="Pazar_Büyümesi",
                size="Balon_Boyutu",
                color="BCG_Kuadrant",
                color_discrete_map=color_map,
                text=group_col,
                hover_data={
                    "Pazar_Payı_Pct": ":.2f",
                    "Pazar_Büyümesi": ":.1f",
                    "Toplam_Satış": ":,.0f",
                    "Balon_Boyutu": False,
                },
                size_max=60,
            )

            x_mid = bcg_df["Pazar_Payı_Pct"].median()
            y_mid = bcg_df["Pazar_Büyümesi"].median()
            fig.add_hline(y=y_mid, line_dash="dot", line_color="rgba(255,255,255,0.2)")
            fig.add_vline(x=x_mid, line_dash="dot", line_color="rgba(255,255,255,0.2)")

            x_max = float(bcg_df["Pazar_Payı_Pct"].max())
            x_min = float(bcg_df["Pazar_Payı_Pct"].min())
            y_max = float(bcg_df["Pazar_Büyümesi"].max())
            y_min = float(bcg_df["Pazar_Büyümesi"].min())

            for label, xy in [
                ("YILDIZLAR ⭐",    (x_max * 0.85, y_max * 0.9)),
                ("NAKİT İNEĞİ 💰",  (x_max * 0.85, y_min * 0.9)),
                ("SORU İŞARETİ ❓", (x_min * 1.1,  y_max * 0.9)),
                ("KÖPEK 🐕",        (x_min * 1.1,  y_min * 0.9)),
            ]:
                fig.add_annotation(
                    x=xy[0], y=xy[1], text=label,
                    showarrow=False,
                    font=dict(size=10, color="rgba(255,255,255,0.3)"),
                )

            fig.update_traces(textposition="top center", textfont=dict(size=9))
            return cls._theme(fig, "📊 BCG Kuadrant Analizi")

        except Exception as exc:
            st.warning(f"⚠️ BCG grafiği hatası: {exc}")
            return None

    # ── 7.4 Fiyat Erozyonu Grafiği ───────────────────────────────────────────

    @classmethod
    def price_erosion_chart(cls, erosion_df: pd.DataFrame) -> Optional[go.Figure]:
        """
        Birikimli SU fiyat erozyonu için çubuk grafik.

        Returns: Plotly Figure veya None
        """
        try:
            if erosion_df is None or erosion_df.empty:
                return None

            group_col = erosion_df.columns[0]
            top = (
                erosion_df.nlargest(15, "Total_Sales")
                if "Total_Sales" in erosion_df.columns
                else erosion_df.head(15)
            )

            if "Birikimli_Erozyon_Pct" not in top.columns:
                return None

            vals = top["Birikimli_Erozyon_Pct"].fillna(0)
            colors_list = ["#ff4757" if v < -5 else "#00e5a0" for v in vals]

            fig = go.Figure(go.Bar(
                x=top[group_col].astype(str),
                y=vals,
                marker_color=colors_list,
                text=[f"{v:.1f}%" for v in vals],
                textposition="outside",
            ))
            fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
            fig.add_hline(y=-5, line_color="#ffb700", line_dash="dot",
                          annotation_text="Orta Erozyon", annotation_position="right")
            fig.add_hline(y=-20, line_color="#ff4757", line_dash="dot",
                          annotation_text="Ağır Erozyon", annotation_position="right")

            return cls._theme(fig, "💲 Birikimli SU Fiyat Erozyonu (2022→2024)")

        except Exception as exc:
            st.warning(f"⚠️ Fiyat erozyonu grafiği hatası: {exc}")
            return None

    # ── 7.5 Evrim Endeksi Grafiği ────────────────────────────────────────────

    @classmethod
    def ei_chart(cls, ei_df: pd.DataFrame) -> Optional[go.Figure]:
        """
        Molekül/Ürün başına EI için yatay çubuk grafik.

        Returns: Plotly Figure veya None
        """
        try:
            if ei_df is None or ei_df.empty:
                return None

            ei_cols = [c for c in ei_df.columns if c.startswith("EI_Growth_")]
            if not ei_cols:
                return None

            last_ei = ei_cols[-1]
            group_col = next(
                (c for c in ["Molecule", "Company"] if c in ei_df.columns), None
            )
            if group_col is None:
                return None

            grp = ei_df.groupby(group_col, observed=False)[last_ei].mean().dropna()
            top = grp.nlargest(20)

            colors_list = ["#00d4ff" if v >= 100 else "#ff4757" for v in top.values]

            fig = go.Figure(go.Bar(
                x=top.values,
                y=top.index.astype(str),
                orientation="h",
                marker_color=colors_list,
                text=[f"{v:.0f}" for v in top.values],
                textposition="outside",
            ))
            fig.add_vline(x=100, line_color="#ffb700", line_dash="dash",
                          annotation_text="Pazar Kriteri (100)",
                          annotation_position="top right")
            return cls._theme(fig, "📈 Evrim Endeksi — Top 20 Ürün")

        except Exception as exc:
            st.warning(f"⚠️ EI grafiği hatası: {exc}")
            return None

    # ── 7.6 HHI Zaman Serisi ─────────────────────────────────────────────────

    @classmethod
    def hhi_chart(cls, hhi_df: pd.DataFrame) -> Optional[go.Figure]:
        """
        HHI ve Top-3 Payı için ikincil eksenli grafik.

        Returns: Plotly Figure veya None
        """
        try:
            if hhi_df is None or hhi_df.empty:
                return None

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Scatter(
                    x=hhi_df["Yıl"], y=hhi_df["HHI"],
                    name="HHI Endeksi",
                    mode="lines+markers+text",
                    text=hhi_df["HHI"].apply(lambda v: f"{v:,.0f}"),
                    textposition="top center",
                    line=dict(color="#00d4ff", width=3),
                    marker=dict(size=10),
                ),
                secondary_y=False,
            )

            if "Top3_Pay_Pct" in hhi_df.columns:
                fig.add_trace(
                    go.Bar(
                        x=hhi_df["Yıl"], y=hhi_df["Top3_Pay_Pct"],
                        name="Top-3 Payı %",
                        marker_color="rgba(123,47,255,0.5)",
                        text=hhi_df["Top3_Pay_Pct"].apply(lambda v: f"{v:.1f}%"),
                        textposition="outside",
                    ),
                    secondary_y=True,
                )

            for y_val, label, color in [
                (2500, "Yüksek Konsantre", "#ff4757"),
                (1500, "Orta Konsantre", "#ffb700"),
            ]:
                fig.add_hline(
                    y=y_val, line_dash="dot", line_color=color,
                    annotation_text=label, annotation_position="right",
                    secondary_y=False,
                )

            fig.update_yaxes(title_text="HHI Endeksi", secondary_y=False)
            fig.update_yaxes(title_text="Top-3 Payı (%)", secondary_y=True)
            cls._theme(fig, "🏭 Pazar Konsantrasyonu (HHI)")
            return fig

        except Exception as exc:
            st.warning(f"⚠️ HHI grafiği hatası: {exc}")
            return None

    # ── 7.7 Tahmin Grafiği ───────────────────────────────────────────────────

    @classmethod
    def forecast_chart(cls, fc_df: pd.DataFrame) -> Optional[go.Figure]:
        """
        Ensemble tahmini ve güven bantlarını gösterir.

        Returns: Plotly Figure veya None
        """
        try:
            if fc_df is None or fc_df.empty:
                return None

            hist = fc_df[fc_df["Tarihsel"] == True].copy()
            fwd  = fc_df[fc_df["Tarihsel"] == False].copy()

            fig = go.Figure()

            if "Satış" in hist.columns:
                fig.add_trace(go.Scatter(
                    x=hist["Yıl"], y=hist["Satış"],
                    mode="lines+markers",
                    name="Tarihsel Satış",
                    line=dict(color="#00e5a0", width=3),
                    marker=dict(size=10),
                ))

            fig.add_trace(go.Scatter(
                x=hist["Yıl"], y=hist["Tahmin"],
                mode="lines",
                name="Model Uyumu",
                line=dict(color="#00d4ff", width=2, dash="dot"),
            ))

            if not fwd.empty:
                if "Alt_CI_95" in fwd.columns and "Üst_CI_95" in fwd.columns:
                    fig.add_trace(go.Scatter(
                        x=pd.concat([fwd["Yıl"], fwd["Yıl"].iloc[::-1]]),
                        y=pd.concat([fwd["Üst_CI_95"], fwd["Alt_CI_95"].iloc[::-1]]),
                        fill="toself",
                        fillcolor="rgba(0,112,224,0.1)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="%95 GA",
                    ))

                if "Alt_CI_80" in fwd.columns and "Üst_CI_80" in fwd.columns:
                    fig.add_trace(go.Scatter(
                        x=pd.concat([fwd["Yıl"], fwd["Yıl"].iloc[::-1]]),
                        y=pd.concat([fwd["Üst_CI_80"], fwd["Alt_CI_80"].iloc[::-1]]),
                        fill="toself",
                        fillcolor="rgba(0,112,224,0.2)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="%80 GA",
                    ))

                fig.add_trace(go.Scatter(
                    x=fwd["Yıl"], y=fwd["Tahmin"],
                    mode="lines+markers",
                    name="Tahmin",
                    line=dict(color="#ffb700", width=3),
                    marker=dict(size=10, symbol="diamond", color="#ffb700"),
                ))

            if not hist.empty:
                fig.add_vline(
                    x=float(hist["Yıl"].max()),
                    line_dash="dash",
                    line_color="rgba(255,255,255,0.3)",
                    annotation_text="Tahmin →",
                )

            return cls._theme(fig, "🔮 Ensemble Pazar Tahmini (ES + Doğrusal Regresyon)")

        except Exception as exc:
            st.warning(f"⚠️ Tahmin grafiği hatası: {exc}")
            return None

    # ── 7.8 Anomali Scatter ──────────────────────────────────────────────────

    @classmethod
    def anomaly_chart(cls, anomaly_df: pd.DataFrame) -> Optional[go.Figure]:
        """
        Anormal ürünleri vurgulayan scatter grafiği.

        Returns: Plotly Figure veya None
        """
        try:
            if anomaly_df is None or anomaly_df.empty:
                return None

            years = DataPipeline._detect_years(anomaly_df, "Sales_")
            if len(years) < 2:
                return None

            x_col, y_col = f"Sales_{years[-2]}", f"Sales_{years[-1]}"
            if x_col not in anomaly_df.columns or y_col not in anomaly_df.columns:
                return None

            group_col = next(
                (c for c in ["Molecule", "Company"] if c in anomaly_df.columns), None
            )
            cat_col = "Anomali_Kategorisi" if "Anomali_Kategorisi" in anomaly_df.columns else None

            color_map = {
                "Kritik": "#ff4757",
                "Yüksek Risk": "#ffb700",
                "Orta": "#7b2fff",
                "Normal": "#00e5a0",
            }

            kwargs = dict(
                data_frame=anomaly_df,
                x=x_col, y=y_col,
                opacity=0.75,
            )
            if cat_col:
                kwargs["color"] = cat_col
                kwargs["color_discrete_map"] = color_map
            if group_col:
                kwargs["hover_name"] = group_col

            fig = px.scatter(**kwargs)
            return cls._theme(fig, "⚠️ Anomali Tespiti — Ürün Risk Haritası")

        except Exception as exc:
            st.warning(f"⚠️ Anomali grafiği hatası: {exc}")
            return None

    # ── 7.9 Pazar Payı Treemap ───────────────────────────────────────────────

    @classmethod
    def treemap_chart(cls, df: pd.DataFrame) -> Optional[go.Figure]:
        """
        Şirket > Molekül hiyerarşik treemap.

        Returns: Plotly Figure veya None
        """
        try:
            years = DataPipeline._detect_years(df, "Sales_")
            if not years:
                return None
            lsc = f"Sales_{years[-1]}"
            if lsc not in df.columns:
                return None

            path_cols = [c for c in ["Company", "Molecule"] if c in df.columns]
            if not path_cols:
                return None

            grp = df.groupby(path_cols, observed=False)[lsc].sum().reset_index()
            grp = grp[grp[lsc] > 0]

            if "Company" in grp.columns:
                top_comp = grp.groupby("Company")[lsc].sum().nlargest(15).index
                grp = grp[grp["Company"].isin(top_comp)]

            fig = px.treemap(
                grp,
                path=[px.Constant("Pazar")] + path_cols,
                values=lsc,
                color=lsc,
                color_continuous_scale=["#0d1f3c", "#0070e0", "#00d4ff"],
            )
            fig.update_traces(textinfo="label+percent parent", textfont_size=11)
            return cls._theme(fig, f"🗺️ Pazar Payı Treemap ({years[-1]})")

        except Exception as exc:
            st.warning(f"⚠️ Treemap hatası: {exc}")
            return None

    # ── 7.10 Kanibalizasyon Isı Haritası ─────────────────────────────────────

    @classmethod
    def cannibalization_heatmap(cls, corr_matrix: Optional[pd.DataFrame]) -> Optional[go.Figure]:
        """
        Şirket içi molekül büyüme korelasyon ısı haritası.

        Returns: Plotly Figure veya None
        """
        try:
            if corr_matrix is None or corr_matrix.empty:
                return None
            if corr_matrix.shape[0] > 25:
                corr_matrix = corr_matrix.iloc[:25, :25]

            fig = go.Figure(go.Heatmap(
                z=corr_matrix.values,
                x=list(corr_matrix.columns.astype(str)),
                y=list(corr_matrix.index.astype(str)),
                colorscale=[[0, "#ff4757"], [0.5, "#0d1f3c"], [1, "#00e5a0"]],
                zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                hovertemplate="Şirket A: %{y}<br>Şirket B: %{x}<br>Korelasyon: %{z:.2f}<extra></extra>",
            ))
            return cls._theme(fig, "🔗 Kanibalizasyon Korelasyon Matrisi")

        except Exception as exc:
            st.warning(f"⚠️ Isı haritası hatası: {exc}")
            return None

    # ── 7.11 Satış Trend ─────────────────────────────────────────────────────

    @classmethod
    def sales_trend_chart(cls, df: pd.DataFrame, top_n: int = 10) -> Optional[go.Figure]:
        """
        Top-N molekül/şirketin 2022–2024 satış trendi.

        Returns: Plotly Figure veya None
        """
        try:
            years = DataPipeline._detect_years(df, "Sales_")
            if len(years) < 2:
                return None

            group_col = "Molecule" if "Molecule" in df.columns else "Company"
            if group_col not in df.columns:
                return None

            lsc = f"Sales_{years[-1]}"
            if lsc not in df.columns:
                return None

            top_items = (
                df.groupby(group_col, observed=False)[lsc]
                .sum()
                .nlargest(top_n)
                .index.tolist()
            )
            grp = df[df[group_col].isin(top_items)]
            colors_cycle = cls.THEME["colorway"]

            fig = go.Figure()
            for i, item in enumerate(top_items):
                sub = grp[grp[group_col] == item]
                xs, ys = [], []
                for yr in years:
                    sc = f"Sales_{yr}"
                    if sc in sub.columns:
                        xs.append(yr)
                        ys.append(float(sub[sc].sum()))

                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="lines+markers",
                    name=str(item)[:30],
                    line=dict(color=colors_cycle[i % len(colors_cycle)], width=2),
                    marker=dict(size=8),
                ))

            return cls._theme(fig, f"📈 Satış Trendi — Top {top_n} {group_col}")

        except Exception as exc:
            st.warning(f"⚠️ Trend grafiği hatası: {exc}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 8 — REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class ReportGenerator:
    """
    Profesyonel çıktı üretimi:
      - Multi-sheet Excel (xlsxwriter)
      - 10 sayfalık executive PDF (ReportLab)
      - HTML interaktif raporu
    """

    # ── 8.1 Excel ────────────────────────────────────────────────────────────

    @staticmethod
    def generate_excel(
        df: pd.DataFrame,
        summary: Dict[str, Any],
        ei_df: Optional[pd.DataFrame],
        erosion_df: Optional[pd.DataFrame],
        hhi_df: Optional[pd.DataFrame],
        bcg_df: Optional[pd.DataFrame],
    ) -> Optional[bytes]:
        """
        Çok sayfalı Excel raporu üretir.

        Sayfalar:
          1. Yönetici Özeti
          2. Tam Veri
          3. Evrim Endeksi
          4. Fiyat Erozyonu
          5. Pazar Konsantrasyonu (HHI)
          6. BCG Sınıflandırması
          7. Şirket Bazlı Pivot

        Returns: Excel baytları veya None
        """
        try:
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                wb = writer.book

                hdr_fmt = wb.add_format({
                    "bold": True, "bg_color": "#0d1f3c", "font_color": "#00d4ff",
                    "border": 1, "align": "center", "font_name": "Calibri", "font_size": 11,
                })
                title_fmt = wb.add_format({
                    "bold": True, "font_size": 16, "font_color": "#00d4ff",
                    "font_name": "Calibri",
                })
                label_fmt = wb.add_format({
                    "bold": True, "font_color": "#8ba3c7",
                    "font_name": "Calibri", "font_size": 10,
                })

                # ── Sayfa 1: Yönetici Özeti ──────────────────────────────────
                ws1 = wb.add_worksheet("Yönetici Özeti")
                ws1.set_tab_color("#0070e0")
                ws1.set_column("A:A", 32)
                ws1.set_column("B:B", 22)
                ws1.write("A1", "PharmaIntelligence Enterprise v8.0 — Yönetici Özeti", title_fmt)
                ws1.write("A2", f"Oluşturulma: {datetime.now().strftime('%Y-%m-%d %H:%M')}", label_fmt)

                kpi_rows = [
                    ("Toplam Kayıt",  f"{summary.get('rows', 0):,}"),
                    ("Analiz Yılları", str(summary.get("years", []))),
                    ("Toplam Pazar (Son Yıl)", f"${summary.get('total_sales', 0)/1e6:.2f}M"),
                    ("Benzersiz Molekül", str(summary.get("molecules", 0))),
                    ("Benzersiz Şirket", str(summary.get("companies", 0))),
                    ("Kapsanan Ülke", str(summary.get("countries", 0))),
                    ("Eksik Veri %", f"{summary.get('missing_pct', 0):.2f}%"),
                    ("Bellek (MB)", f"{summary.get('memory_mb', 0):.2f}"),
                ]
                for r, (lbl, val) in enumerate(kpi_rows, start=4):
                    ws1.write(r, 0, lbl, label_fmt)
                    ws1.write(r, 1, val, hdr_fmt)

                # ── Sayfa 2: Tam Veri ─────────────────────────────────────────
                df.head(50000).to_excel(writer, sheet_name="Tam Veri", index=False)
                ws2 = writer.sheets["Tam Veri"]
                ws2.set_tab_color("#00d4ff")
                for ci, col_name in enumerate(df.columns):
                    ws2.set_column(ci, ci, min(max(len(str(col_name)) + 2, 10), 30))

                # ── Sayfa 3: Evrim Endeksi ────────────────────────────────────
                if ei_df is not None and not ei_df.empty:
                    show_cols = [c for c in ei_df.columns
                                 if c.startswith("EI_") or c in ["Molecule", "Company", "EI_Kategori"]]
                    ei_df[show_cols].head(500).to_excel(
                        writer, sheet_name="Evrim Endeksi", index=False
                    )
                    writer.sheets["Evrim Endeksi"].set_tab_color("#7b2fff")

                # ── Sayfa 4: Fiyat Erozyonu ───────────────────────────────────
                if erosion_df is not None and not erosion_df.empty:
                    erosion_df.head(500).to_excel(
                        writer, sheet_name="Fiyat Erozyonu", index=False
                    )
                    writer.sheets["Fiyat Erozyonu"].set_tab_color("#ffb700")

                # ── Sayfa 5: HHI ──────────────────────────────────────────────
                if hhi_df is not None and not hhi_df.empty:
                    hhi_df.to_excel(
                        writer, sheet_name="Pazar Konsantrasyonu (HHI)", index=False
                    )
                    writer.sheets["Pazar Konsantrasyonu (HHI)"].set_tab_color("#ff4757")

                # ── Sayfa 6: BCG ──────────────────────────────────────────────
                if bcg_df is not None and not bcg_df.empty:
                    bcg_df.to_excel(
                        writer, sheet_name="BCG Sınıflandırması", index=False
                    )
                    writer.sheets["BCG Sınıflandırması"].set_tab_color("#00e5a0")

                # ── Sayfa 7: Şirket Pivotu ────────────────────────────────────
                if "Company" in df.columns:
                    yr_list = DataPipeline._detect_years(df, "Sales_")
                    s_cols = [f"Sales_{yr}" for yr in yr_list if f"Sales_{yr}" in df.columns]
                    if s_cols:
                        pivot = df.groupby("Company", observed=False)[s_cols].sum()
                        pivot.to_excel(writer, sheet_name="Şirket Pivotu")
                        writer.sheets["Şirket Pivotu"].set_tab_color("#8ba3c7")

            buf.seek(0)
            return buf.read()

        except Exception as exc:
            st.error(f"❌ Excel üretim hatası: {exc}")
            st.code(traceback.format_exc())
            return None

    # ── 8.2 PDF ──────────────────────────────────────────────────────────────

    @staticmethod
    def generate_pdf(
        summary: Dict[str, Any],
        ei_df: Optional[pd.DataFrame],
        erosion_df: Optional[pd.DataFrame],
        hhi_df: Optional[pd.DataFrame],
        bcg_df: Optional[pd.DataFrame],
        fc_df: Optional[pd.DataFrame],
    ) -> Optional[bytes]:
        """
        10 sayfalık yönetici PDF raporu üretir.

        Sayfalar: Kapak · Özet · Pazar Trendi · EI · Fiyat Erozyonu ·
                  HHI · BCG · Tahmin · Öngörüler · Metodoloji

        Returns: PDF baytları veya None
        """
        if not REPORTLAB_OK:
            st.error("❌ ReportLab yüklü değil: pip install reportlab")
            return None

        try:
            buf = BytesIO()
            doc = SimpleDocTemplate(
                buf, pagesize=A4,
                leftMargin=2*cm, rightMargin=2*cm,
                topMargin=2*cm, bottomMargin=2*cm,
            )

            styles = getSampleStyleSheet()

            def sty(name, **kw):
                return ParagraphStyle(name, parent=styles["Normal"], **kw)

            cover_title = sty("CT", fontSize=26, textColor=rl_colors.HexColor("#0070e0"),
                              spaceAfter=12, fontName="Helvetica-Bold", alignment=TA_CENTER)
            cover_sub   = sty("CS", fontSize=12, textColor=rl_colors.HexColor("#8ba3c7"),
                              spaceAfter=6,  fontName="Helvetica", alignment=TA_CENTER)
            h1  = sty("H1", fontSize=16, textColor=rl_colors.HexColor("#0070e0"),
                      spaceAfter=10, fontName="Helvetica-Bold")
            h2  = sty("H2", fontSize=13, textColor=rl_colors.HexColor("#00d4ff"),
                      spaceAfter=8,  fontName="Helvetica-Bold")
            body = sty("BD", fontSize=10, leading=14, spaceAfter=6,
                       textColor=rl_colors.HexColor("#222222"), fontName="Helvetica")
            cap  = sty("CA", fontSize=8, textColor=rl_colors.grey,
                       fontName="Helvetica-Oblique", alignment=TA_CENTER)
            conf = sty("CN", fontSize=9, textColor=rl_colors.red,
                       fontName="Helvetica-Bold", alignment=TA_CENTER)

            def tbl_style():
                return TableStyle([
                    ("GRID",       (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#cccccc")),
                    ("FONTNAME",   (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE",   (0, 0), (-1, -1), 9),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [rl_colors.white, rl_colors.HexColor("#f5f9ff")]),
                    ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
                    ("TOPPADDING",    (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#0d1f3c")),
                    ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTCOLOR",  (0, 0), (-1, 0), rl_colors.HexColor("#00d4ff")),
                    ("FONTSIZE",   (0, 0), (-1, 0), 10),
                ])

            hr = lambda: HRFlowable(
                width="100%", thickness=1,
                color=rl_colors.HexColor("#0070e0"), spaceAfter=12,
            )

            story = []
            yrs_str = " → ".join(str(y) for y in summary.get("years", []))

            # ── Sayfa 1: Kapak ─────────────────────────────────────────────
            story += [
                Spacer(1, 4*cm),
                Paragraph("PharmaIntelligence", cover_title),
                Paragraph("Enterprise v8.0", cover_title),
                Spacer(1, 0.5*cm),
                hr(),
                Paragraph("Gelişmiş İlaç Pazar Analitiği Raporu", cover_sub),
                Paragraph(f"Oluşturulma: {datetime.now().strftime('%d %B %Y  |  %H:%M')}", cover_sub),
                Spacer(1, 2*cm),
            ]
            cover_data = [
                ["Metrik", "Değer"],
                ["Toplam Kayıt",   f"{summary.get('rows', 0):,}"],
                ["Analiz Dönemi",  yrs_str],
                ["Pazar Değeri",   f"${summary.get('total_sales', 0)/1e6:.2f}M USD"],
                ["Molekül Sayısı", f"{summary.get('molecules', 0):,}"],
                ["Şirket Sayısı",  f"{summary.get('companies', 0):,}"],
                ["Ülke Sayısı",    f"{summary.get('countries', 0):,}"],
            ]
            ct = Table(cover_data, colWidths=[8*cm, 7*cm])
            ct.setStyle(tbl_style())
            story += [ct, Spacer(1, 3*cm),
                      Paragraph("GİZLİ — Yalnızca Dahili Yönetici Kullanımı İçin", conf),
                      PageBreak()]

            # ── Sayfa 2: Yönetici Özeti ────────────────────────────────────
            story += [
                Paragraph("Yönetici Özeti", h1), hr(),
                Paragraph(
                    f"Bu rapor, {len(summary.get('years', []))} yıllık pazar verisi "
                    f"({yrs_str}) analizini sunmaktadır. Veri seti "
                    f"<b>{summary.get('rows', 0):,}</b> kayıt, "
                    f"<b>{summary.get('molecules', 0)}</b> benzersiz molekül ve "
                    f"<b>{summary.get('companies', 0)}</b> şirketi kapsamaktadır. "
                    f"Son dönem toplam pazar değeri "
                    f"<b>${summary.get('total_sales', 0)/1e6:.2f}M USD</b>'dır.",
                    body,
                ),
                Spacer(1, 0.5*cm),
                PageBreak(),
            ]

            # ── Sayfa 3: Evrim Endeksi ─────────────────────────────────────
            story += [Paragraph("Evrim Endeksi (EI) Analizi", h1), hr(),
                      Paragraph(
                          "EI = (Ürün Büyümesi / Pazar Medyanı) × 100. "
                          "EI > 100 pazar payı kazanan ürünleri gösterir.", body,
                      )]
            if ei_df is not None and not ei_df.empty:
                ei_c = [c for c in ei_df.columns if c.startswith("EI_Growth_")]
                gc = "Molecule" if "Molecule" in ei_df.columns else "Company"
                if ei_c and gc in ei_df.columns:
                    top_ei = ei_df[[gc, ei_c[-1]]].dropna().head(15)
                    ei_data = [[gc, "Evrim Endeksi"]]
                    for _, row in top_ei.iterrows():
                        ei_data.append([str(row[gc])[:30], f"{row[ei_c[-1]]:.1f}"])
                    et = Table(ei_data, colWidths=[10*cm, 7*cm])
                    et.setStyle(tbl_style())
                    story.append(et)
            story.append(PageBreak())

            # ── Sayfa 4: Fiyat Erozyonu ────────────────────────────────────
            story += [Paragraph("Fiyat Erozyonu Analizi", h1), hr(),
                      Paragraph(
                          "SU (Standart Birim) ortalama fiyatının yıldan yıla değişimi. "
                          "Negatif erozyon generik baskısı veya ihale etkisini gösterir.", body,
                      )]
            if erosion_df is not None and not erosion_df.empty and "Birikimli_Erozyon_Pct" in erosion_df.columns:
                gc_e = erosion_df.columns[0]
                er_top = erosion_df[[gc_e, "Birikimli_Erozyon_Pct"]].head(15)
                er_data = [[gc_e, "Birikimli Erozyon %"]]
                for _, row in er_top.iterrows():
                    er_data.append([str(row[gc_e])[:30], f"{row['Birikimli_Erozyon_Pct']:.1f}%"])
                ert = Table(er_data, colWidths=[10*cm, 7*cm])
                ert.setStyle(tbl_style())
                story.append(ert)
            story.append(PageBreak())

            # ── Sayfa 5: HHI ───────────────────────────────────────────────
            story += [Paragraph("Pazar Konsantrasyonu (HHI)", h1), hr(),
                      Paragraph(
                          "HHI = Σ(pazar_payı²). DOJ eşikleri: "
                          "<1.500 Rekabetçi | 1.500–2.500 Orta | >2.500 Yüksek.", body,
                      )]
            if hhi_df is not None and not hhi_df.empty:
                h_data = [list(hhi_df.columns)]
                for _, row in hhi_df.iterrows():
                    h_data.append([
                        f"{v:,.1f}" if isinstance(v, float) else str(v)
                        for v in row
                    ])
                cw = [17*cm / len(h_data[0])] * len(h_data[0])
                ht = Table(h_data, colWidths=cw)
                ht.setStyle(tbl_style())
                story.append(ht)
            story.append(PageBreak())

            # ── Sayfa 6: BCG ───────────────────────────────────────────────
            story += [Paragraph("BCG Portföy Sınıflandırması", h1), hr(),
                      Paragraph(
                          "Yıldız: yüksek büyüme + yüksek pay. "
                          "Nakit İneği: düşük büyüme + yüksek pay. "
                          "Soru İşareti: yatırım kararı gerektirir. "
                          "Köpek: çıkış adayı.", body,
                      )]
            if bcg_df is not None and not bcg_df.empty:
                gc_b = bcg_df.columns[0]
                b_data = [[gc_b, "Büyüme %", "Pay %", "Satış ($M)", "Kuadrant"]]
                for _, row in bcg_df.head(20).iterrows():
                    b_data.append([
                        str(row[gc_b])[:22],
                        f"{row.get('Pazar_Büyümesi', 0):.1f}%",
                        f"{row.get('Pazar_Payı_Pct', 0):.2f}%",
                        f"${row.get('Toplam_Satış', 0)/1e6:.2f}M",
                        str(row.get("BCG_Kuadrant", "—"))[:18],
                    ])
                bt = Table(b_data, colWidths=[5*cm, 3*cm, 3*cm, 3.5*cm, 3.5*cm])
                bt.setStyle(tbl_style())
                story.append(bt)
            story.append(PageBreak())

            # ── Sayfa 7: Tahmin ────────────────────────────────────────────
            story += [Paragraph("Pazar Tahmini (2025+)", h1), hr(),
                      Paragraph(
                          "Hibrit ensemble: ES (%60) + Doğrusal Regresyon (%40). "
                          "Bootstrap (500 iterasyon) ile %80 ve %95 güven aralığı.", body,
                      )]
            if fc_df is not None and not fc_df.empty:
                fwd = fc_df[fc_df["Tarihsel"] == False]
                if not fwd.empty:
                    fc_data = [["Yıl", "Tahmin ($M)", "Alt CI %80", "Üst CI %80", "YoY %"]]
                    for _, row in fwd.iterrows():
                        fc_data.append([
                            str(int(row["Yıl"])),
                            f"${row['Tahmin']/1e6:.2f}M" if pd.notna(row.get("Tahmin")) else "—",
                            f"${row.get('Alt_CI_80', 0)/1e6:.2f}M",
                            f"${row.get('Üst_CI_80', 0)/1e6:.2f}M",
                            f"{row.get('YoY_Büyüme_Pct', 0):.1f}%" if row.get("YoY_Büyüme_Pct") else "—",
                        ])
                    ft = Table(fc_data, colWidths=[2.5*cm, 3.5*cm, 3.5*cm, 3.5*cm, 4*cm])
                    ft.setStyle(tbl_style())
                    story.append(ft)
            else:
                story.append(Paragraph("Tahmin verisi yok. Önce AI Katmanı'nda tahmin çalıştırın.", body))
            story.append(PageBreak())

            # ── Sayfa 8: Öngörüler ─────────────────────────────────────────
            story += [Paragraph("Temel Öngörüler & Stratejik Öneriler", h1), hr()]
            insights = [
                ("Pazar Dinamiği", "EI > 150 olan Yıldız ürünler hızlandırılmış yatırım gerektirir. Büyüme ivmesini sürdürmek için kaynak önceliği belirleyin."),
                ("Fiyat Stratejisi", "CAGR > %5 erozyon generik baskısı sinyali verir. Değer bazlı fiyatlandırma veya hat uzatma stratejileri değerlendirin."),
                ("Portföy Optimizasyonu", "BCG Köpekleri çıkış veya yeniden konumlandırma için gözden geçirin. Soru İşaretleri için farklılaştırma planı yapın."),
                ("Rekabetçi Tepki", "HHI > 2.500 monopol riski işareti. Düzenleyici müdahale ve yeni oyuncu girişini izleyin."),
                ("Kanibalizasyon Riski", "r < -0.7 korelasyon gösteren molekül çiftleri aynı hasta popülasyonunu hedefliyor olabilir. Lansman sırasını gözden geçirin."),
                ("Tahmin Güvenilirliği", "Dar güven aralığı → kararlı talep. Geniş aralık → senaryo planlaması gerektirir."),
            ]
            for t, d in insights:
                story += [Paragraph(f"■ {t}", h2), Paragraph(d, body), Spacer(1, 0.2*cm)]
            story.append(PageBreak())

            # ── Sayfa 9: Metodoloji ────────────────────────────────────────
            story += [Paragraph("Metodoloji ve Ekler", h1), hr()]
            meth = [
                ["Modül", "Açıklama"],
                ["Sütun Standardizasyonu", "Regex ile MAT Q3 sütun adları otomatik eşlenir."],
                ["Dozaj Verimliliği", "SU / Birim oranı. >1 çok dozlu paket anlamına gelir."],
                ["Evrim Endeksi", "EI = (Ürün Büyümesi / Pazar Medyanı) × 100."],
                ["HHI", "HHI = Σ(pay²). Yıllık trend olarak hesaplanır."],
                ["Ensemble Tahmin", "ES %60 + LR %40 karışımı, 500 bootstrap CI."],
                ["Anomali Tespiti", "Isolation Forest, contamination=%10, 200 ağaç."],
                ["Kanibalizasyon", "Şirket içi molekül büyüme korelasyonu. r < -0.7 = yüksek risk."],
            ]
            mt = Table(meth, colWidths=[5*cm, 12*cm])
            mt.setStyle(tbl_style())
            story += [
                mt, Spacer(1, 1*cm),
                Paragraph(
                    f"© 2025 PharmaIntelligence Inc. — Enterprise v8.0  |  "
                    f"Tüm hakları saklıdır.  |  {datetime.now().strftime('%d.%m.%Y')}",
                    cap,
                ),
            ]

            doc.build(story)
            buf.seek(0)
            return buf.read()

        except Exception as exc:
            st.error(f"❌ PDF üretim hatası: {exc}")
            st.code(traceback.format_exc())
            return None

    # ── 8.3 HTML Raporu ──────────────────────────────────────────────────────

    @staticmethod
    def generate_html(df: pd.DataFrame, summary: Dict[str, Any]) -> str:
        """Kendi kendine yeten HTML raporu üretir."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        yrs = " → ".join(str(y) for y in summary.get("years", []))

        rows_html = ""
        for _, row in df.head(200).iterrows():
            rows_html += "<tr>" + "".join(f"<td>{str(v)[:40]}</td>" for v in row) + "</tr>"
        hdrs = "".join(f"<th>{c}</th>" for c in df.columns)

        return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>PharmaIntelligence v8.0 Raporu</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;background:#0a1628;color:#e8f0fe;margin:0;padding:20px}}
.header{{background:linear-gradient(135deg,#0d1f3c,#1a3560);border-radius:12px;padding:2rem;margin-bottom:2rem;border:1px solid rgba(0,212,255,0.2)}}
h1{{color:#00d4ff;font-size:2rem;margin:0}} .sub{{color:#8ba3c7;margin:.5rem 0 0 0}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin:1.5rem 0}}
.kpi{{background:#112548;border-radius:10px;padding:1.2rem;border:1px solid rgba(0,212,255,0.15)}}
.kpi-label{{font-size:.75rem;color:#8ba3c7;text-transform:uppercase;letter-spacing:1px}}
.kpi-val{{font-size:1.6rem;font-weight:800;margin:.3rem 0}}
table{{width:100%;border-collapse:collapse;background:#0d1f3c;border-radius:8px;overflow:hidden;margin-top:1rem}}
th{{background:#0d1f3c;color:#00d4ff;padding:10px 12px;font-size:.85rem;text-align:left;border-bottom:2px solid rgba(0,212,255,0.3)}}
td{{padding:8px 12px;font-size:.82rem;border-bottom:1px solid rgba(255,255,255,0.05)}}
tr:hover{{background:rgba(0,112,224,0.1)}}
.footer{{text-align:center;color:#4a6080;font-size:.8rem;margin-top:2rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.1)}}
</style>
</head>
<body>
<div class="header">
  <h1>⚕️ PharmaIntelligence Enterprise v8.0</h1>
  <p class="sub">Gelişmiş İlaç Pazar Analitiği — {ts}</p>
</div>
<div class="kpi-grid">
  <div class="kpi"><div class="kpi-label">Toplam Kayıt</div><div class="kpi-val">{summary.get('rows',0):,}</div></div>
  <div class="kpi"><div class="kpi-label">Dönem</div><div class="kpi-val">{yrs}</div></div>
  <div class="kpi"><div class="kpi-label">Pazar ($M)</div><div class="kpi-val">${summary.get('total_sales',0)/1e6:.1f}M</div></div>
  <div class="kpi"><div class="kpi-label">Molekül</div><div class="kpi-val">{summary.get('molecules',0):,}</div></div>
  <div class="kpi"><div class="kpi-label">Şirket</div><div class="kpi-val">{summary.get('companies',0):,}</div></div>
  <div class="kpi"><div class="kpi-label">Ülke</div><div class="kpi-val">{summary.get('countries',0):,}</div></div>
</div>
<h2 style="color:#00d4ff">📋 Veri Önizleme (ilk 200 satır)</h2>
<div style="overflow-x:auto">
<table><thead><tr>{hdrs}</tr></thead><tbody>{rows_html}</tbody></table>
</div>
<div class="footer">© 2025 PharmaIntelligence Inc. — Enterprise v8.0 | {ts}</div>
</body>
</html>"""
