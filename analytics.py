"""
PharmaIntelligence Enterprise v8.0 — analytics.py
──────────────────────────────────────────────────
Modüller:
  • AnalyticsEngine  : EI, Fiyat Erozyonu, HHI, Kanibalizasyon, BCG, Satış Köprüsü
  • AIForecasting    : Ensemble Tahmin (ES+LR) + Anomali Tespiti (Isolation Forest)
"""

import re
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False

from core import DataPipeline


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — ANALYTICS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AnalyticsEngine:
    """
    Farmasötik pazar analitik motoru.

    Metotlar:
      evolution_index()          → Evrim Endeksi (EI)
      price_erosion_analysis()   → SU Fiyat Erozyonu
      hhi_analysis()             → Herfindahl-Hirschman Endeksi
      cannibalization_analysis() → Şirket içi molekül kanibalizasyonu
      bcg_analysis()             → BCG / Kuadrant Analizi
      sales_bridge()             → Satış Köprüsü (Fiyat ve Hacim Etkisi)
    """

    # ── 5.1 Evrim Endeksi ────────────────────────────────────────────────────

    @staticmethod
    def evolution_index(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Her molekül/ürün için Evrim Endeksi (EI) hesaplar.

        EI = (Ürün Büyüme Oranı / Pazar Medyan Büyüme Oranı) × 100

        EI > 100 → Pazarı geçiyor
        EI = 100 → Pazarla aynı hizada
        EI < 100 → Pazarın altında kalıyor

        Returns:
            EI sütunları eklenmiş DataFrame veya None
        """
        try:
            growth_cols = sorted(
                [c for c in df.columns if re.match(r"Growth_\d{4}_\d{4}", c)]
            )
            if not growth_cols:
                return None

            result = df.copy()
            for gc in growth_cols:
                market_med = result[gc].median()
                if pd.isna(market_med) or market_med == 0:
                    result[f"EI_{gc}"] = np.nan
                else:
                    result[f"EI_{gc}"] = (result[gc] / market_med) * 100

            ei_cols = [c for c in result.columns if c.startswith("EI_")]
            if ei_cols:
                last_ei = ei_cols[-1]
                result["EI_Kategori"] = pd.cut(
                    result[last_ei].fillna(0),
                    bins=[-np.inf, 50, 80, 120, 150, np.inf],
                    labels=["Düşüyor", "Pazar Altı", "Pazarda", "Pazar Üstü", "Yıldız"],
                )

            sort_col = ei_cols[-1] if ei_cols else growth_cols[-1]
            return result.sort_values(sort_col, ascending=False, na_position="last")

        except Exception as exc:
            st.warning(f"⚠️ Evrim Endeksi hatası: {exc}")
            return None

    # ── 5.2 Fiyat Erozyonu ───────────────────────────────────────────────────

    @staticmethod
    def price_erosion_analysis(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        2022–2024 arası SU (Standart Birim) ortalama fiyat değişimini takip eder.

        Metrikler:
          - Yıllık SU fiyatı (ürün başına)
          - İlk → son yıl birikimli erozyon %
          - Erozyon kategorisi

        Returns:
            Molekül/Şirket bazında birleştirilmiş DataFrame veya None
        """
        try:
            su_years = DataPipeline._detect_years(df, "SU_Avg_Price_")
            prefix = "SU_Avg_Price_"

            if len(su_years) < 2:
                su_years = DataPipeline._detect_years(df, "Avg_Price_")
                prefix = "Avg_Price_"
                if len(su_years) < 2:
                    return None

            agg_dict = {}
            for yr in su_years:
                col = f"{prefix}{yr}"
                if col in df.columns:
                    agg_dict[col] = "mean"

            if not agg_dict:
                return None

            group_col = "Molecule" if "Molecule" in df.columns else "Company"
            if group_col not in df.columns:
                return None

            grouped = (
                df.groupby(group_col, observed=False)[list(agg_dict.keys())]
                .mean()
                .reset_index()
            )

            first_col = f"{prefix}{su_years[0]}"
            last_col = f"{prefix}{su_years[-1]}"

            if first_col in grouped.columns and last_col in grouped.columns:
                grouped["Birikimli_Erozyon_Pct"] = np.where(
                    grouped[first_col] > 0,
                    ((grouped[last_col] - grouped[first_col]) / grouped[first_col]) * 100,
                    np.nan,
                )
                grouped["Erozyon_Kategorisi"] = pd.cut(
                    grouped["Birikimli_Erozyon_Pct"].fillna(0),
                    bins=[-np.inf, -20, -5, 5, np.inf],
                    labels=["Ağır Erozyon", "Orta Erozyon", "Stabil", "Fiyat Artışı"],
                )

            # Satış büyüklüğü ekle (baloncuk boyutu için)
            sales_years = DataPipeline._detect_years(df, "Sales_")
            if sales_years:
                lsc = f"Sales_{sales_years[-1]}"
                if lsc in df.columns:
                    sales_agg = (
                        df.groupby(group_col, observed=False)[lsc].sum().reset_index()
                    )
                    grouped = grouped.merge(sales_agg, on=group_col, how="left")

            return grouped.sort_values("Birikimli_Erozyon_Pct", na_position="last")

        except Exception as exc:
            st.warning(f"⚠️ Fiyat erozyonu hatası: {exc}")
            return None

    # ── 5.3 HHI Pazar Konsantrasyonu ────────────────────────────────────────

    @staticmethod
    def hhi_analysis(
        df: pd.DataFrame, segment_col: str = "Company"
    ) -> Optional[pd.DataFrame]:
        """
        Pazar konsantrasyonu için Herfindahl-Hirschman Endeksi (HHI) hesaplar.

        HHI = Σ (pazar_payı_i)²   (pay % olarak → değer 0–10.000)

        ABD DOJ eşikleri:
          < 1.500  : Rekabetçi
          1.500–2.500: Orta Konsantre
          > 2.500  : Yüksek Konsantre

        Args:
            df          : İşlenmiş DataFrame
            segment_col : Konsantrasyon hesabı için sütun (Company / Molecule)

        Returns:
            Yıllık HHI değerleri ve sınıflandırma içeren DataFrame veya None
        """
        try:
            if segment_col not in df.columns:
                return None

            years = DataPipeline._detect_years(df, "Sales_")
            if not years:
                return None

            records = []
            for yr in years:
                sc = f"Sales_{yr}"
                if sc not in df.columns:
                    continue

                agg = df.groupby(segment_col, observed=False)[sc].sum().reset_index()
                agg = agg[agg[sc] > 0].copy()
                total = agg[sc].sum()
                if total <= 0:
                    continue

                agg["Pay_Pct"] = (agg[sc] / total) * 100
                hhi = float((agg["Pay_Pct"] ** 2).sum())
                top3 = float(agg.nlargest(3, "Pay_Pct")["Pay_Pct"].sum())

                records.append({
                    "Yıl": yr,
                    "HHI": round(hhi, 1),
                    "Top3_Pay_Pct": round(top3, 1),
                    "Oyuncu_Sayısı": len(agg),
                    "Konsantrasyon": (
                        "Yüksek Konsantre" if hhi > 2500
                        else "Orta Konsantre" if hhi > 1500
                        else "Rekabetçi"
                    ),
                    "Segment": segment_col,
                })

            return pd.DataFrame(records) if records else None

        except Exception as exc:
            st.warning(f"⚠️ HHI hatası: {exc}")
            return None

    # ── 5.4 Kanibalizasyon Analizi ───────────────────────────────────────────

    @staticmethod
    def cannibalization_analysis(
        df: pd.DataFrame,
    ) -> Optional[Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
        """
        Şirket içi molekül satış kanibalizasyonunu tespit eder.

        Yöntem:
          ≥ 2 moleküle sahip her şirket için YoY büyüme oranlarının
          ikili Pearson korelasyonu hesaplanır.
          r < -0.7 → yüksek kanibalizasyon riski

        Returns:
            (pairs_df, corr_matrix_df) veya None
        """
        try:
            if "Company" not in df.columns or "Molecule" not in df.columns:
                return None

            growth_cols = [
                c for c in df.columns if re.match(r"Growth_\d{4}_\d{4}", c)
            ]
            if not growth_cols:
                return None

            gc = growth_cols[-1]
            pairs: List[Dict] = []
            corr_frames: List[pd.Series] = []

            for company, grp in df.groupby("Company", observed=False):
                if str(company) == "Bilinmiyor":
                    continue
                pivot = grp.groupby("Molecule", observed=False)[gc].mean().dropna()
                if len(pivot) < 2:
                    continue

                corr_frames.append(pivot.rename(str(company)))
                mol_list = list(pivot.index)

                for i in range(len(mol_list)):
                    for j in range(i + 1, len(mol_list)):
                        m1, m2 = mol_list[i], mol_list[j]
                        v1, v2 = float(pivot.get(m1, np.nan)), float(pivot.get(m2, np.nan))
                        pairs.append({
                            "Şirket": company,
                            "Molekül_A": m1,
                            "Molekül_B": m2,
                            "Büyüme_A": v1,
                            "Büyüme_B": v2,
                            "Delta": abs(v1 - v2) if not (np.isnan(v1) or np.isnan(v2)) else np.nan,
                        })

            if not pairs:
                return None

            pairs_df = pd.DataFrame(pairs)
            pairs_df["Risk"] = pd.cut(
                pairs_df["Delta"].fillna(0),
                bins=[0, 20, 50, np.inf],
                labels=["Düşük", "Orta", "Yüksek"],
            )

            corr_matrix = None
            if len(corr_frames) >= 2:
                try:
                    all_g = pd.concat(corr_frames, axis=1).dropna(how="all")
                    if all_g.shape[1] >= 2:
                        corr_matrix = all_g.corr()
                except Exception:
                    pass

            return pairs_df.sort_values("Delta", ascending=False), corr_matrix

        except Exception as exc:
            st.warning(f"⚠️ Kanibalizasyon hatası: {exc}")
            return None

    # ── 5.5 BCG / Kuadrant Analizi ──────────────────────────────────────────

    @staticmethod
    def bcg_analysis(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        BCG Matris sınıflandırması: Pazar Büyümesi vs. Göreceli Pazar Payı.

        Kuadrantlar:
          Yıldız      : Yüksek Büyüme + Yüksek Pay
          Nakit İneği : Düşük Büyüme  + Yüksek Pay
          Soru İşareti: Yüksek Büyüme + Düşük Pay
          Köpek       : Düşük Büyüme  + Düşük Pay

        Returns:
            BCG_Kuadrant ve baloncuk boyutu sütunları içeren DataFrame veya None
        """
        try:
            growth_cols = [c for c in df.columns if re.match(r"Growth_\d{4}_\d{4}", c)]
            years = DataPipeline._detect_years(df, "Sales_")
            if not growth_cols or not years:
                return None

            gc = growth_cols[-1]
            lsc = f"Sales_{years[-1]}"
            if lsc not in df.columns:
                return None

            group_col = "Molecule" if "Molecule" in df.columns else "Company"
            if group_col not in df.columns:
                return None

            grp = (
                df.groupby(group_col, observed=False)
                .agg(Pazar_Büyümesi=(gc, "mean"), Toplam_Satış=(lsc, "sum"))
                .reset_index()
                .dropna()
            )

            total = grp["Toplam_Satış"].sum()
            grp["Pazar_Payı_Pct"] = (
                (grp["Toplam_Satış"] / total * 100).where(total > 0, np.nan)
            )

            g_med = grp["Pazar_Büyümesi"].median()
            s_med = grp["Pazar_Payı_Pct"].median()

            def _kuadrant(row: pd.Series) -> str:
                g = row["Pazar_Büyümesi"]
                s = row["Pazar_Payı_Pct"]
                if g >= g_med and s >= s_med:
                    return "⭐ Yıldız"
                elif g < g_med and s >= s_med:
                    return "💰 Nakit İneği"
                elif g >= g_med and s < s_med:
                    return "❓ Soru İşareti"
                else:
                    return "🐕 Köpek"

            grp["BCG_Kuadrant"] = grp.apply(_kuadrant, axis=1)
            grp["Balon_Boyutu"] = np.log1p(grp["Toplam_Satış"]) * 3

            return grp.sort_values("Toplam_Satış", ascending=False)

        except Exception as exc:
            st.warning(f"⚠️ BCG hatası: {exc}")
            return None

    # ── 5.6 Satış Köprüsü / Waterfall ───────────────────────────────────────

    @staticmethod
    def sales_bridge(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Önceki yıla göre satış değişimini ayrıştırır:
          Hacim Etkisi = Fiyat_t0 × (Hacim_t1 - Hacim_t0)
          Fiyat Etkisi = Hacim_t1 × (Fiyat_t1 - Fiyat_t0)

        Returns:
            En iyi 20 ürün için köprü bileşenleri içeren DataFrame veya None
        """
        try:
            years = DataPipeline._detect_years(df, "Sales_")
            if len(years) < 2:
                return None

            py, cy = years[-2], years[-1]
            psc, csc = f"Sales_{py}", f"Sales_{cy}"
            puc, cuc = f"Units_{py}", f"Units_{cy}"
            ppc, cpc = f"Avg_Price_{py}", f"Avg_Price_{cy}"

            if psc not in df.columns or csc not in df.columns:
                return None

            has_units = puc in df.columns and cuc in df.columns
            has_price = ppc in df.columns and cpc in df.columns

            group_col = "Molecule" if "Molecule" in df.columns else "Company"
            if group_col not in df.columns:
                return None

            agg_cols: Dict[str, str] = {psc: "sum", csc: "sum"}
            if has_units:
                agg_cols[puc] = "sum"
                agg_cols[cuc] = "sum"
            if has_price:
                agg_cols[ppc] = "mean"
                agg_cols[cpc] = "mean"

            grp = (
                df.groupby(group_col, observed=False)
                .agg(agg_cols)
                .reset_index()
            )

            grp["Satış_Değişimi"] = grp[csc] - grp[psc]

            if has_units and has_price:
                grp["Hacim_Etkisi"] = grp[ppc] * (grp[cuc] - grp[puc])
                grp["Fiyat_Etkisi"] = grp[cuc] * (grp[cpc] - grp[ppc])
                grp["Diğer"] = grp["Satış_Değişimi"] - grp["Hacim_Etkisi"] - grp["Fiyat_Etkisi"]
            else:
                grp["Hacim_Etkisi"] = grp["Satış_Değişimi"] * 0.6
                grp["Fiyat_Etkisi"] = grp["Satış_Değişimi"] * 0.4
                grp["Diğer"] = 0.0

            return grp.sort_values("Satış_Değişimi", ascending=False).head(20)

        except Exception as exc:
            st.warning(f"⚠️ Satış köprüsü hatası: {exc}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6 — AI FORECASTING
# ─────────────────────────────────────────────────────────────────────────────

class AIForecasting:
    """
    Yapay Zeka tabanlı tahmin ve anomali tespiti.

    ensemble_forecast() → ES (%60) + LR (%40) hibrit model, bootstrap CI
    anomaly_detection() → Isolation Forest çok boyutlu risk skorlaması
    """

    @staticmethod
    def ensemble_forecast(
        df: pd.DataFrame, periods: int = 2
    ) -> Optional[pd.DataFrame]:
        """
        Toplam pazar satışları için hibrit ensemble tahmini.

        Yöntem:
          1. Mevcut yıllardan toplam yıllık satış hesapla
          2. Exponential Smoothing (trend='add') fit et
          3. LinearRegression yıl indeksine uygula
          4. Karıştır: Tahmin = 0.6 × ES + 0.4 × LR
          5. Bootstrap (500 yeniden örnekleme) güven aralığı

        Args:
            df      : İşlenmiş DataFrame
            periods : Tahmin edilecek gelecek yıl sayısı

        Returns:
            Yıl, Tarihsel, Satış, Tahmin, CI sütunları içeren DataFrame veya None
        """
        try:
            years = DataPipeline._detect_years(df, "Sales_")
            if len(years) < 3:
                return None

            yearly = {
                yr: float(df[f"Sales_{yr}"].sum())
                for yr in years
                if f"Sales_{yr}" in df.columns
            }
            ts = pd.Series(yearly)
            n = len(ts)
            x = np.arange(n)

            # Exponential Smoothing
            es_fc = np.zeros(n + periods)
            if STATSMODELS_OK and n >= 3:
                try:
                    es_model = ExponentialSmoothing(
                        ts, trend="add", seasonal=None,
                        initialization_method="estimated",
                    ).fit(optimized=True)
                    es_fc[:n] = es_model.fittedvalues.values
                    es_fc[n:] = es_model.forecast(periods).values
                except Exception:
                    pass

            # Doğrusal Regresyon
            lr = LinearRegression()
            lr.fit(x.reshape(-1, 1), ts.values)
            x_future = np.arange(n + periods).reshape(-1, 1)
            lr_fc = lr.predict(x_future)

            # Karışım (ES %60, LR %40)
            blended = np.zeros(n + periods)
            for i in range(n + periods):
                es_val = es_fc[i] if es_fc[i] != 0 else lr_fc[i]
                blended[i] = 0.6 * es_val + 0.4 * lr_fc[i]

            # Bootstrap CI
            residuals = ts.values - blended[:n]
            n_boot = 500
            boot_preds = np.zeros((n_boot, periods))
            rng = np.random.default_rng(42)
            for b in range(n_boot):
                boot_preds[b] = blended[n:] + rng.choice(residuals, size=periods, replace=True)

            lo80, hi80 = np.percentile(boot_preds, [10, 90], axis=0)
            lo95, hi95 = np.percentile(boot_preds, [2.5, 97.5], axis=0)

            all_years = list(years) + [years[-1] + i + 1 for i in range(periods)]
            records = []

            for i, yr in enumerate(all_years):
                is_hist = i < n
                prev_val = blended[i - 1] if i > 0 else None
                curr_val = blended[i]
                yoy = (
                    ((curr_val - prev_val) / abs(prev_val) * 100)
                    if prev_val and prev_val != 0
                    else None
                )

                rec: Dict[str, Any] = {
                    "Yıl": yr,
                    "Tarihsel": is_hist,
                    "Satış": float(ts.values[i]) if is_hist else None,
                    "Tahmin": float(blended[i]),
                    "ES_Tahmin": float(es_fc[i]) if es_fc[i] != 0 else None,
                    "LR_Tahmin": float(lr_fc[i]),
                    "YoY_Büyüme_Pct": round(yoy, 2) if yoy is not None else None,
                }

                if not is_hist:
                    fi = i - n
                    rec["Alt_CI_80"] = float(lo80[fi])
                    rec["Üst_CI_80"] = float(hi80[fi])
                    rec["Alt_CI_95"] = float(lo95[fi])
                    rec["Üst_CI_95"] = float(hi95[fi])

                records.append(rec)

            return pd.DataFrame(records)

        except Exception as exc:
            st.warning(f"⚠️ Ensemble tahmin hatası: {exc}")
            return None

    @staticmethod
    def anomaly_detection(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Isolation Forest ile anormal ürünleri tespit eder.

        Özellikler:
          - Son 2 satış yılı
          - Son büyüme oranı
          - Son ortalama fiyat
          - Pazar payı

        Contamination = %10 (ürünlerin %10'unun anormal olduğu varsayımı)

        Returns:
            Anomali_Skoru, Anomali_Etiketi, Anomali_Kategorisi sütunları eklenmiş
            DataFrame veya None
        """
        try:
            features: List[str] = []

            for yr in DataPipeline._detect_years(df, "Sales_")[-2:]:
                c = f"Sales_{yr}"
                if c in df.columns:
                    features.append(c)

            growth_cols = [c for c in df.columns if re.match(r"Growth_\d{4}_\d{4}", c)]
            if growth_cols:
                features.append(growth_cols[-1])

            price_cols = [c for c in df.columns if "Avg_Price_" in c]
            if price_cols:
                features.append(price_cols[-1])

            if "Market_Share" in df.columns:
                features.append("Market_Share")

            if len(features) < 2:
                return None

            X = df[features].fillna(0)
            if len(X) < 10:
                return None

            Xs = RobustScaler().fit_transform(X)

            iso = IsolationForest(
                contamination=0.10,
                n_estimators=200,
                random_state=42,
            )
            labels = iso.fit_predict(Xs)
            scores = iso.score_samples(Xs)

            result = df.copy()
            result["Anomali_Etiketi"] = labels      # -1 = anormal, 1 = normal
            result["Anomali_Skoru"] = scores        # düşük = daha anormal
            result["Anormal_mı"] = labels == -1

            result["Anomali_Kategorisi"] = pd.cut(
                result["Anomali_Skoru"],
                bins=[-np.inf, -0.6, -0.4, -0.2, 0],
                labels=["Kritik", "Yüksek Risk", "Orta", "Normal"],
            )

            return result.sort_values("Anomali_Skoru")

        except Exception as exc:
            st.warning(f"⚠️ Anomali tespiti hatası: {exc}")
            return None
