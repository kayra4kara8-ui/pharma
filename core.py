"""
PharmaIntelligence Enterprise v8.0 — core.py
─────────────────────────────────────────────
Modüller:
  • ColumnStandardizer  : MAT sütun otomatik eşleme
  • DataPipeline        : ETL, temizleme, türetilmiş metrikler
  • AdvancedFilterSystem: Çok boyutlu filtreler (Sector/Region/Specialty/NFC123)
  • SessionManager      : Çökmeye dayanıklı session_state yönetimi
  • Yardımcı fonksiyonlar & CSS
"""

import re
import hashlib
import traceback
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

ENTERPRISE_CSS = """
<style>
:root {
    --bg-base:#060d1a; --bg-surface:#0d1f3c; --bg-card:#112548;
    --bg-hover:#1a3560; --accent-1:#00d4ff; --accent-2:#0070e0;
    --accent-3:#7b2fff; --success:#00e5a0; --warning:#ffb700;
    --danger:#ff4757; --text-primary:#e8f0fe; --text-secondary:#8ba3c7;
    --text-muted:#4a6080; --border:rgba(0,212,255,0.15);
    --glow-blue:0 0 20px rgba(0,212,255,0.3);
    --radius-sm:6px; --radius-md:12px; --radius-lg:18px;
    --transition:0.25s cubic-bezier(0.4,0,0.2,1);
    --font-mono:'JetBrains Mono','Fira Code','Courier New',monospace;
}
.stApp {
    background:radial-gradient(ellipse at 10% 0%,rgba(0,112,224,0.18) 0%,transparent 50%),
               radial-gradient(ellipse at 90% 100%,rgba(123,47,255,0.15) 0%,transparent 50%),
               var(--bg-base);
    font-family:'Sora','DM Sans','Segoe UI',sans-serif;
    color:var(--text-primary);
}
.pharma-hero {
    background:linear-gradient(135deg,rgba(0,112,224,0.25) 0%,rgba(123,47,255,0.20) 100%);
    border:1px solid var(--border); border-radius:var(--radius-lg);
    padding:2.5rem 3rem; margin-bottom:2rem; position:relative; overflow:hidden;
}
.pharma-hero::before {
    content:''; position:absolute; inset:0;
    background:repeating-linear-gradient(45deg,transparent,transparent 30px,
        rgba(0,212,255,0.02) 30px,rgba(0,212,255,0.02) 31px);
    pointer-events:none;
}
.pharma-title {
    font-size:2.8rem; font-weight:900; letter-spacing:-1.5px;
    background:linear-gradient(90deg,var(--accent-1),var(--accent-2),var(--accent-3));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; margin:0 0 0.4rem 0;
}
.pharma-subtitle { color:var(--text-secondary); font-size:1rem; font-weight:400; line-height:1.6; margin:0; }
.version-badge {
    display:inline-block;
    background:linear-gradient(135deg,var(--accent-1),var(--accent-2));
    color:var(--bg-base); font-size:0.72rem; font-weight:800;
    padding:0.2rem 0.7rem; border-radius:20px; letter-spacing:1px;
    text-transform:uppercase; margin-bottom:0.8rem;
}
.section-title {
    font-size:1.55rem; font-weight:800; color:var(--text-primary);
    margin:2rem 0 1.2rem 0; display:flex; align-items:center; gap:0.6rem;
}
.section-title::after {
    content:''; flex:1; height:1px;
    background:linear-gradient(90deg,var(--border),transparent); margin-left:0.5rem;
}
.subsection-title {
    font-size:1.15rem; font-weight:700; color:var(--text-secondary);
    margin:1.5rem 0 0.8rem 0; padding-left:0.8rem; border-left:3px solid var(--accent-2);
}
.kpi-card {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:var(--radius-md); padding:1.4rem 1.6rem;
    position:relative; overflow:hidden; transition:all var(--transition);
}
.kpi-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,var(--accent-1),var(--accent-3));
}
.kpi-card:hover { border-color:rgba(0,212,255,0.4); box-shadow:var(--glow-blue); transform:translateY(-3px); }
.kpi-label { font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:1.2px; color:var(--text-muted); margin-bottom:0.5rem; }
.kpi-value { font-size:2.1rem; font-weight:900; color:var(--text-primary); line-height:1; margin-bottom:0.3rem; }
.kpi-delta { font-size:0.82rem; font-weight:600; }
.kpi-delta.up { color:var(--success); } .kpi-delta.down { color:var(--danger); }
.kpi-icon { position:absolute; right:1.2rem; top:1.2rem; font-size:1.8rem; opacity:0.25; }
.insight-card {
    background:var(--bg-card); border-radius:var(--radius-md);
    padding:1.2rem 1.4rem; margin:0.6rem 0; border-left:4px solid;
    transition:all var(--transition);
}
.insight-card:hover { transform:translateX(4px); }
.insight-card.info { border-left-color:var(--accent-1); }
.insight-card.success { border-left-color:var(--success); }
.insight-card.warning { border-left-color:var(--warning); }
.insight-card.danger { border-left-color:var(--danger); }
.insight-title { font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:0.4rem; color:var(--text-secondary); }
.insight-content { font-size:0.95rem; color:var(--text-primary); line-height:1.5; }
.filter-header { font-size:0.78rem; font-weight:800; text-transform:uppercase; letter-spacing:1.5px; color:var(--accent-1); margin:1rem 0 0.5rem 0; }
.filter-status-box {
    background:linear-gradient(135deg,rgba(0,212,255,0.1),rgba(123,47,255,0.1));
    border:1px solid var(--border); border-radius:var(--radius-sm);
    padding:0.8rem; font-size:0.85rem; color:var(--text-secondary); margin-bottom:1rem;
}
.status-pill {
    display:inline-flex; align-items:center; gap:0.3rem;
    padding:0.25rem 0.8rem; border-radius:20px; font-size:0.78rem;
    font-weight:700; text-transform:uppercase; letter-spacing:0.5px;
}
.status-pill.live { background:rgba(0,229,160,0.15); color:var(--success); border:1px solid rgba(0,229,160,0.3); }
.status-pill.warn { background:rgba(255,183,0,0.15); color:var(--warning); border:1px solid rgba(255,183,0,0.3); }
.ai-badge {
    background:linear-gradient(135deg,var(--accent-3),var(--accent-2));
    color:#fff; font-size:0.68rem; font-weight:800;
    padding:0.2rem 0.6rem; border-radius:4px; text-transform:uppercase; letter-spacing:1px;
}
.upload-hero {
    border:2px dashed rgba(0,212,255,0.3); border-radius:var(--radius-lg);
    padding:3rem; text-align:center; background:rgba(0,112,224,0.05); transition:all var(--transition);
}
.upload-hero:hover { border-color:rgba(0,212,255,0.6); background:rgba(0,112,224,0.08); }
@keyframes pulse-dot { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:0.5; transform:scale(1.4); } }
.pulse-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--success); animation:pulse-dot 1.5s infinite; margin-right:4px; }
[data-testid="stMetricValue"] { font-size:1.9rem !important; font-weight:800 !important; }
.stDataFrame { border-radius:var(--radius-md) !important; }
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:var(--bg-base); }
::-webkit-scrollbar-thumb { background:var(--accent-2); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:var(--accent-1); }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────────────────────────

def fmt_currency(value: float, unit: str = "M") -> str:
    """Sayıyı para birimi formatına çevirir."""
    try:
        d = {"M": 1e6, "B": 1e9, "K": 1e3}.get(unit, 1)
        return f"${value / d:,.2f}{unit}"
    except Exception:
        return "—"


def fmt_pct(value: float, decimals: int = 1) -> str:
    """Sayıyı yüzde formatına çevirir."""
    try:
        return f"{value:.{decimals}f}%"
    except Exception:
        return "—"


def kpi_card(label: str, value: str, delta: str = "", delta_up: bool = True, icon: str = "") -> str:
    """KPI metrik kartı HTML döner."""
    delta_cls = "up" if delta_up else "down"
    arrow = "▲" if delta_up else "▼"
    delta_html = f'<div class="kpi-delta {delta_cls}">{arrow} {delta}</div>' if delta else ""
    icon_html = f'<div class="kpi-icon">{icon}</div>' if icon else ""
    return (
        f'<div class="kpi-card">{icon_html}'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{delta_html}</div>'
    )


def section_title(text: str) -> None:
    """Bölüm başlığı render eder."""
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def insight_card(text: str, kind: str = "info", title: str = "Bilgi") -> None:
    """İçgörü kartı render eder."""
    st.markdown(
        f'<div class="insight-card {kind}">'
        f'<div class="insight-title">{title}</div>'
        f'<div class="insight-content">{text}</div></div>',
        unsafe_allow_html=True,
    )


def safe_df(key: str) -> Optional[pd.DataFrame]:
    """
    session_state'ten güvenli DataFrame okur.
    None veya boş ise None döner — 'or' operatörü kullanmaz.
    """
    val = st.session_state.get(key)
    if val is None:
        return None
    if isinstance(val, pd.DataFrame) and val.empty:
        return None
    return val


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — COLUMN STANDARDIZER
# ─────────────────────────────────────────────────────────────────────────────

class ColumnStandardizer:
    """
    IMS/IQVIA tarzı ham sütun isimlerini standart formata otomatik eşler.

    Örnek:
        'MAT Q3 2024 USD MNF'           → 'Sales_2024'
        'MAT Q3 2022 SU Avg Price USD'  → 'SU_Avg_Price_2022'
        'Corporation'                   → 'Company'
    """

    PATTERN_MAP: List[Tuple[str, str]] = [
        (r"MAT\s*Q\d+\s*(20\d\d)\s*USD\s*MNF(?!\s*SU|\s*Unit\s*Avg)", "Sales_{1}"),
        (r"MAT\s*Q\d+\s*(20\d\d)\s*Unit\s*Avg\s*Price\s*USD\s*MNF", "Avg_Price_{1}"),
        (r"MAT\s*Q\d+\s*(20\d\d)\s*SU\s*Avg\s*Price\s*USD\s*MNF", "SU_Avg_Price_{1}"),
        (r"MAT\s*Q\d+\s*(20\d\d)\s*Standard\s*Units", "Standard_Units_{1}"),
        (r"MAT\s*Q\d+\s*(20\d\d)\s*Units", "Units_{1}"),
    ]

    STATIC_MAP: Dict[str, str] = {
        "Source.Name": "Source", "SourceName": "Source",
        "Country": "Country", "Sector": "Sector",
        "Corporation": "Company", "Manufacturer": "Manufacturer",
        "Molecule List": "Molecule_List", "MoleculeList": "Molecule_List",
        "Molecule": "Molecule", "Chemical Salt": "Chemical_Salt",
        "International Product": "International_Product",
        "Specialty Product": "Specialty_Product",
        "NFC123": "NFC123", "International Pack": "International_Pack",
        "International Strength": "International_Strength",
        "International Size": "International_Size",
        "International Volume": "International_Volume",
        "International Prescription": "International_Prescription",
        "Panel": "Panel", "Sub-Region": "Sub_Region",
        "SubRegion": "Sub_Region", "Region": "Region",
        "Specialty": "Specialty",
    }

    TR_MAP: Dict[str, str] = {
        "Ş": "S", "ş": "s", "İ": "I", "ı": "i",
        "Ğ": "G", "ğ": "g", "Ü": "U", "ü": "u",
        "Ö": "O", "ö": "o", "Ç": "C", "ç": "c",
    }

    @classmethod
    def standardize_columns(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Tüm sütunları standart isimlere çevirir. (renamed_df, mapping) döner."""
        mapping: Dict[str, str] = {}
        seen: Dict[str, int] = {}
        new_cols: List[str] = []

        for col in df.columns:
            cleaned = str(col)
            for tr, en in cls.TR_MAP.items():
                cleaned = cleaned.replace(tr, en)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

            result = cls._apply_patterns(cleaned) or cls._apply_static(cleaned) or cleaned

            if result in seen:
                seen[result] += 1
                result = f"{result}_{seen[result]}"
            else:
                seen[result] = 0

            mapping[col] = result
            new_cols.append(result)

        renamed = df.copy()
        renamed.columns = new_cols
        return renamed, mapping

    @classmethod
    def _apply_patterns(cls, col: str) -> Optional[str]:
        for pattern, template in cls.PATTERN_MAP:
            m = re.search(pattern, col, re.IGNORECASE)
            if m:
                return template.replace("{1}", m.group(1))
        return None

    @classmethod
    def _apply_static(cls, col: str) -> Optional[str]:
        if col in cls.STATIC_MAP:
            return cls.STATIC_MAP[col]
        for key, val in cls.STATIC_MAP.items():
            if key.lower() in col.lower():
                return val
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class DataPipeline:
    """
    İlaç pazar verisi için tam ETL hattı.

    Adımlar:
      1. Dosya yükleme (CSV / Excel)
      2. Sütun standardizasyonu
      3. Veri temizleme ve tip dönüşümü
      4. Türetilmiş metrikler:
           - Büyüme oranları (YoY, CAGR)
           - Pazar payı
           - Dozaj Verimliliği (SU/Birim)
           - SU Fiyat Değişimi
    """

    YEAR_RANGE = (2018, 2030)

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=5)
    def load(file_data: bytes, file_name: str) -> Optional[pd.DataFrame]:
        """
        Yüklenen dosyayı DataFrame'e çevirir.

        Args:
            file_data : Ham dosya baytları
            file_name : Uzantı tespiti için dosya adı

        Returns:
            Ham DataFrame veya None
        """
        try:
            buf = BytesIO(file_data)
            if file_name.lower().endswith(".csv"):
                df = pd.read_csv(buf, low_memory=False, encoding="utf-8", on_bad_lines="skip")
            elif file_name.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(buf, engine="openpyxl")
            else:
                st.error("❌ Desteklenmeyen dosya formatı. Lütfen CSV veya Excel yükleyin.")
                return None

            if df.empty:
                st.error("❌ Yüklenen dosya boş.")
                return None
            return df

        except Exception as exc:
            st.error(f"❌ Dosya yükleme hatası: {exc}")
            st.code(traceback.format_exc())
            return None

    @staticmethod
    def process(raw_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Tam ETL işlemi: standardize → temizle → zenginleştir.

        Args:
            raw_df : DataPipeline.load() çıktısı

        Returns:
            İşlenmiş DataFrame veya None
        """
        try:
            df, _ = ColumnStandardizer.standardize_columns(raw_df)
            df = DataPipeline._coerce_numerics(df)

            cat_cols = [
                "Company", "Molecule", "Country", "Sector",
                "Region", "Specialty", "NFC123", "Source",
                "Manufacturer", "Sub_Region",
            ]
            for col in cat_cols:
                if col in df.columns:
                    df[col] = df[col].fillna("Bilinmiyor").astype(str).str.strip()

            df = DataPipeline._compute_derived(df)
            df = DataPipeline._optimize_dtypes(df)
            return df

        except Exception as exc:
            st.error(f"❌ Pipeline işleme hatası: {exc}")
            st.code(traceback.format_exc())
            return None

    @staticmethod
    def _coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
        keywords = [
            "Sales_", "Units_", "Standard_Units_",
            "Avg_Price_", "SU_Avg_Price_", "Growth_", "CAGR",
            "Market_Share", "HHI", "EI_",
        ]
        for col in df.columns:
            if any(kw in col for kw in keywords):
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def _compute_derived(df: pd.DataFrame) -> pd.DataFrame:
        """
        Türetilmiş farmasötik metrikleri hesaplar:
          - Growth_{Y1}_{Y2} : Yıllık büyüme %
          - CAGR             : Bileşik yıllık büyüme
          - Market_Share     : En son yıl pazar payı %
          - Dosage_Efficiency: SU / Birim oranı
          - SU_Price_Change  : SU fiyat değişimi %
        """
        try:
            years = DataPipeline._detect_years(df, "Sales_")
            if not years:
                return df

            # YoY büyüme
            for i in range(1, len(years)):
                py, cy = years[i - 1], years[i]
                pcol, ccol = f"Sales_{py}", f"Sales_{cy}"
                if pcol in df.columns and ccol in df.columns:
                    df[f"Growth_{py}_{cy}"] = np.where(
                        df[pcol].abs() > 0,
                        ((df[ccol] - df[pcol]) / df[pcol].abs()) * 100,
                        np.nan,
                    )

            # CAGR
            if len(years) >= 2:
                fc, lc = f"Sales_{years[0]}", f"Sales_{years[-1]}"
                n = years[-1] - years[0]
                if fc in df.columns and lc in df.columns and n > 0:
                    df["CAGR"] = np.where(
                        df[fc] > 0,
                        (np.power(np.clip(df[lc] / df[fc], 0, None), 1 / n) - 1) * 100,
                        np.nan,
                    )

            # Pazar payı
            lsc = f"Sales_{years[-1]}"
            if lsc in df.columns:
                total = df[lsc].sum()
                if total > 0:
                    df["Market_Share"] = (df[lsc] / total) * 100
                else:
                    df["Market_Share"] = np.nan

            # Dozaj Verimliliği (SU / Birim)
            su_years = DataPipeline._detect_years(df, "Standard_Units_")
            u_years = DataPipeline._detect_years(df, "Units_")
            for yr in sorted(set(su_years) & set(u_years)):
                su_col, u_col = f"Standard_Units_{yr}", f"Units_{yr}"
                if su_col in df.columns and u_col in df.columns:
                    df[f"Dosage_Efficiency_{yr}"] = np.where(
                        df[u_col] > 0, df[su_col] / df[u_col], np.nan
                    )

            # SU Fiyat Değişimi
            su_price_years = DataPipeline._detect_years(df, "SU_Avg_Price_")
            for i in range(1, len(su_price_years)):
                py, cy = su_price_years[i - 1], su_price_years[i]
                pcol, ccol = f"SU_Avg_Price_{py}", f"SU_Avg_Price_{cy}"
                if pcol in df.columns and ccol in df.columns:
                    df[f"SU_Price_Change_{py}_{cy}"] = np.where(
                        df[pcol] > 0,
                        ((df[ccol] - df[pcol]) / df[pcol]) * 100,
                        np.nan,
                    )

        except Exception as exc:
            st.warning(f"⚠️ Türetilmiş metrik hatası: {exc}")

        return df

    @staticmethod
    def _detect_years(df: pd.DataFrame, prefix: str) -> List[int]:
        """Verilen öneke sahip sütunlardan yıl listesi çıkarır."""
        years = []
        for col in df.columns:
            if col.startswith(prefix):
                m = re.search(r"(20\d{2})", col)
                if m:
                    yr = int(m.group(1))
                    if DataPipeline.YEAR_RANGE[0] <= yr <= DataPipeline.YEAR_RANGE[1]:
                        years.append(yr)
        return sorted(set(years))

    @staticmethod
    def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Bellek kullanımını azaltmak için dtype optimizasyonu."""
        for col in df.select_dtypes(include="float64").columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        for col in df.select_dtypes(include="int64").columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        for col in ["Company", "Molecule", "Country", "Sector", "Region", "Specialty", "NFC123"]:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    @staticmethod
    def get_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Veri seti özet istatistiklerini döner."""
        years = DataPipeline._detect_years(df, "Sales_")
        last_yr = years[-1] if years else None
        lsc = f"Sales_{last_yr}" if last_yr else None

        return {
            "rows": len(df),
            "columns": len(df.columns),
            "years": years,
            "last_year": last_yr,
            "total_sales": float(df[lsc].sum()) if lsc and lsc in df.columns else 0.0,
            "molecules": int(df["Molecule"].nunique()) if "Molecule" in df.columns else 0,
            "companies": int(df["Company"].nunique()) if "Company" in df.columns else 0,
            "countries": int(df["Country"].nunique()) if "Country" in df.columns else 0,
            "missing_pct": round(float(df.isnull().values.mean()) * 100, 2),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — ADVANCED FILTER SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedFilterSystem:
    """
    Çok boyutlu filtre sistemi.

    Desteklenen sütunlar: Sector, Region, Specialty, NFC123,
    Country, Company, Molecule ve sayısal aralıklar.
    Streamlit session_state ile filtre durumu korunur.
    """

    CATEGORICAL_FILTERS: List[Tuple[str, str, str]] = [
        ("Country",    "🌍 Ülke",      "flt_country"),
        ("Company",    "🏢 Şirket",    "flt_company"),
        ("Molecule",   "🧪 Molekül",   "flt_molecule"),
        ("Sector",     "🏥 Sektör",    "flt_sector"),
        ("Region",     "🗺️ Bölge",   "flt_region"),
        ("Specialty",  "💊 Uzmanlık",  "flt_specialty"),
        ("NFC123",     "🔬 NFC123",    "flt_nfc123"),
        ("Sub_Region", "📍 Alt Bölge", "flt_subregion"),
    ]

    @classmethod
    def render_sidebar(cls, df: pd.DataFrame) -> Dict:
        """
        Sidebar filtre widgetlarını çizer ve aktif filtre konfigürasyonunu döner.

        Args:
            df : Tam işlenmiş DataFrame (filtrelenmeden önce)

        Returns:
            filter_config : cls.apply() tarafından kullanılan dict
        """
        st.sidebar.markdown(
            '<div class="filter-header">⚙️ FİLTRELER & SEGMENTASYON</div>',
            unsafe_allow_html=True,
        )

        filter_config: Dict[str, Any] = {}

        # Global arama
        search = st.sidebar.text_input(
            "🔎 Global Arama",
            value=st.session_state.get("flt_search", ""),
            placeholder="Molekül / Şirket / Ülke…",
            key="flt_search",
        )
        if search.strip():
            filter_config["search"] = search.strip()

        # Kategorik filtreler
        st.sidebar.markdown("---")
        for col, label, key in cls.CATEGORICAL_FILTERS:
            if col not in df.columns:
                continue
            options = sorted(df[col].dropna().astype(str).unique())
            if not options:
                continue
            options = options[:200]  # büyük listelerden korunma

            selected = st.sidebar.multiselect(
                label,
                options=["TÜMÜ"] + options,
                default=["TÜMÜ"],
                key=key,
            )
            if "TÜMÜ" not in selected and selected:
                filter_config[col] = selected

        # Sayısal filtreler
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            '<div class="filter-header">📊 SAYISAL ARALIKLAR</div>',
            unsafe_allow_html=True,
        )

        years = DataPipeline._detect_years(df, "Sales_")
        if years:
            lsc = f"Sales_{years[-1]}"
            if lsc in df.columns:
                col_vals = df[lsc].dropna()
                if not col_vals.empty:
                    lo, hi = float(col_vals.min()), float(col_vals.max())
                    if lo < hi:
                        sel = st.sidebar.slider(
                            f"Satış {years[-1]} (USD)",
                            lo, hi, (lo, hi),
                            format="%.0f",
                            key="flt_sales_range",
                        )
                        filter_config["sales_range"] = (sel, lsc)

        growth_cols = [c for c in df.columns if re.match(r"Growth_\d{4}_\d{4}", c)]
        if growth_cols:
            gc_col = growth_cols[-1]
            gv = df[gc_col].dropna()
            if not gv.empty:
                glo = float(max(gv.quantile(0.01), -500.0))
                ghi = float(min(gv.quantile(0.99), 500.0))
                glo, ghi = min(glo, -100.0), max(ghi, 100.0)
                gsel = st.sidebar.slider(
                    f"Büyüme % ({gc_col})",
                    glo, ghi, (glo, ghi),
                    format="%.1f%%",
                    key="flt_growth_range",
                )
                filter_config["growth_range"] = (gsel, gc_col)

        # Boolean filtreler
        st.sidebar.markdown("---")
        if "International_Product" in df.columns:
            intl = st.sidebar.selectbox(
                "🌐 Uluslararası Ürünler",
                ["Tümü", "Sadece Uluslararası", "Sadece Yerel"],
                key="flt_intl",
            )
            if intl != "Tümü":
                filter_config["international"] = intl

        pos_only = st.sidebar.checkbox("📈 Sadece Pozitif Büyüme", key="flt_pos_growth")
        if pos_only and growth_cols:
            filter_config["positive_growth"] = growth_cols[-1]

        # Aksiyon butonları
        st.sidebar.markdown("---")
        c1, c2 = st.sidebar.columns(2)
        with c1:
            if st.button("✅ Uygula", use_container_width=True, key="btn_apply_flt"):
                st.session_state["filters_applied"] = True
        with c2:
            if st.button("🗑️ Sıfırla", use_container_width=True, key="btn_reset_flt"):
                cls._reset_filters()
                st.rerun()

        return filter_config

    @classmethod
    def apply(cls, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Filtre konfigürasyonunu uygular ve filtrelenmiş kopyayı döner.

        Args:
            df     : Tam DataFrame
            config : render_sidebar() çıktısı

        Returns:
            Filtrelenmiş DataFrame (hiç satır kalmadıysa orijinali döner)
        """
        try:
            mask = pd.Series(True, index=df.index)

            if "search" in config:
                term = config["search"].lower()
                str_cols = (
                    df.select_dtypes(include="object").columns.tolist()
                    + df.select_dtypes(include="category").columns.tolist()
                )
                search_mask = pd.Series(False, index=df.index)
                for col in str_cols:
                    search_mask |= df[col].astype(str).str.lower().str.contains(term, na=False)
                mask &= search_mask

            for col, _, _ in cls.CATEGORICAL_FILTERS:
                if col in config and col in df.columns:
                    vals = [str(v) for v in config[col]]
                    mask &= df[col].astype(str).isin(vals)

            if "sales_range" in config:
                (lo, hi), col = config["sales_range"]
                if col in df.columns:
                    mask &= df[col].between(lo, hi)

            if "growth_range" in config:
                (lo, hi), col = config["growth_range"]
                if col in df.columns:
                    mask &= df[col].fillna(0).between(lo, hi)

            if "international" in config and "International_Product" in df.columns:
                intl_vals = df["International_Product"].astype(str)
                if config["international"] == "Sadece Uluslararası":
                    mask &= intl_vals.isin(["1", "1.0", "True"])
                else:
                    mask &= ~intl_vals.isin(["1", "1.0", "True"])

            if "positive_growth" in config:
                col = config["positive_growth"]
                if col in df.columns:
                    mask &= df[col].fillna(0) > 0

            result = df.loc[mask]
            return result if len(result) > 0 else df

        except Exception as exc:
            st.warning(f"⚠️ Filtre hatası: {exc}")
            return df

    @classmethod
    def _reset_filters(cls) -> None:
        """Tüm filtre session_state anahtarlarını temizler."""
        keys = ["flt_search", "flt_sales_range", "flt_growth_range",
                "flt_intl", "flt_pos_growth", "filters_applied"]
        keys += [k for _, _, k in cls.CATEGORICAL_FILTERS]
        for k in keys:
            st.session_state.pop(k, None)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — SESSION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class SessionManager:
    """
    Çökmeye dayanıklı Streamlit session_state sarmalayıcısı.

    NOT: DataFrame karşılaştırmasında Python 'or' operatörü kullanmaz;
    bunun yerine açık None kontrolleri yapar.
    """

    KEYS = [
        "raw_df", "processed_df", "filtered_df", "summary",
        "ei_df", "erosion_df", "hhi_df", "bcg_df",
        "bridge_df", "cannibal_result",
        "forecast_df", "anomaly_df",
        "file_name", "col_mapping", "file_hash",
        "filters_applied",
    ]

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """KeyError fırlatmadan güvenli okuma."""
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any) -> None:
        """Atomik yazma."""
        st.session_state[key] = value

    @staticmethod
    def clear(keys: Optional[List[str]] = None) -> None:
        """Belirtilen (veya tüm bilinen) anahtarları temizler."""
        targets = keys if keys is not None else SessionManager.KEYS
        for k in targets:
            st.session_state.pop(k, None)

    @staticmethod
    def is_loaded() -> bool:
        """İşlenmiş veri mevcut ve boş değilse True döner."""
        df = st.session_state.get("processed_df")
        if df is None:
            return False
        if not isinstance(df, pd.DataFrame):
            return False
        return not df.empty

    @staticmethod
    def get_df(key: str) -> Optional[pd.DataFrame]:
        """
        DataFrame için güvenli okuma.
        None veya boş DataFrame ise None döner.
        'or' operatörü KULLANMAZ — ValueError'u önler.
        """
        val = st.session_state.get(key)
        if val is None:
            return None
        if not isinstance(val, pd.DataFrame):
            return None
        if val.empty:
            return None
        return val

    @staticmethod
    def init_defaults() -> None:
        """Tüm beklenen anahtarları None olarak başlatır."""
        for k in SessionManager.KEYS:
            st.session_state.setdefault(k, None)
        st.session_state.setdefault("app_initialized", True)
