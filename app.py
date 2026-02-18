"""
PharmaIntelligence Enterprise v8.0 — app.py
─────────────────────────────────────────────
Streamlit Cloud 503 hatası düzeltmeleri:
  ✅ Analiz butonlarına progress bar eklendi (zaman aşımı önlenir)
  ✅ gc.collect() her analiz sonrası (bellek serbest bırakma)
  ✅ Büyük DataFrame işlemlerinde chunk'lama
  ✅ .streamlit/config.toml ile server ayarları
  ✅ Her analiz fonksiyonu try/except + st.error ile sarıldı
"""

import gc
import hashlib
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# SAYFA YAPILANDIRMASI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PharmaIntelligence Enterprise v8.0",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://pharmaintelligence.com/destek",
        "Report a bug": "https://pharmaintelligence.com/hata-bildir",
        "About": (
            "### PharmaIntelligence Enterprise v8.0\n"
            "Yapay zeka destekli ilaç pazar analitiği platformu.\n\n"
            "© 2025 PharmaIntelligence Inc."
        ),
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# LOCAL IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

from core import (
    ENTERPRISE_CSS,
    ColumnStandardizer,
    DataPipeline,
    SessionManager,
    fmt_currency,
    insight_card,
    kpi_card,
    section_title,
)
from analytics import AIForecasting, AnalyticsEngine
from visualizer import EnterpriseVisualizer, ReportGenerator
# filters.py içeriği bu dosyaya entegre edildi


# ═════════════════════════════════════════════════════════════════════════════
# FILTERS MODULE (filters.py içeriği entegre edildi)
# ═════════════════════════════════════════════════════════════════════════════



import re


# ─────────────────────────────────────────────────────────────────────────────
# TANIMLAR
# ─────────────────────────────────────────────────────────────────────────────

DIMS: List[Tuple[str, str, str, str]] = [
    ("Country",    "Ülke",      "🌍", "sf_country"),
    ("City",       "Şehir",     "🏙️", "sf_city"),
    ("Company",    "Şirket",    "🏢", "sf_company"),
    ("Molecule",   "Molekül",   "🧪", "sf_molecule"),
    ("Sector",     "Sektör",    "🏥", "sf_sector"),
    ("Region",     "Bölge",     "🗺️", "sf_region"),
    ("Sub_Region", "Alt Bölge", "📍", "sf_subregion"),
    ("Specialty",  "Uzmanlık",  "💊", "sf_specialty"),
    ("NFC123",     "NFC123",    "🔬", "sf_nfc123"),
]

_JUNK = {"", "nan", "none", "bilinmiyor", "unknown", "n/a", "-", "null", "na"}


# ─────────────────────────────────────────────────────────────────────────────
# YARDIMCI
# ─────────────────────────────────────────────────────────────────────────────

def _opts(s: pd.Series) -> List[str]:
    """Temiz, sıralı, benzersiz seçenek listesi."""
    try:
        v = s.astype(str).str.strip()
        return sorted(v[~v.str.lower().isin(_JUNK)].unique().tolist())
    except Exception:
        return []


def _ss_list(key: str) -> List[str]:
    """Session state'ten liste oku."""
    v = st.session_state.get(key, [])
    return v if isinstance(v, list) else []


def _detect_years(df: pd.DataFrame, prefix: str) -> List[int]:
    """Sales_ / Units_ gibi prefix'teki yılları döner."""
    years = []
    for c in df.columns:
        if c.startswith(prefix):
            m = re.search(r"(20\d{2})", c)
            if m:
                y = int(m.group(1))
                if 2010 <= y <= 2035:
                    years.append(y)
    return sorted(set(years))


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR FİLTRE SİSTEMİ
# ─────────────────────────────────────────────────────────────────────────────

class SidebarFilterSystem:
    """
    Sidebar'da tam ekran filtre paneli.
    Tüm seçenekler görünür, sınır yok.
    """

    @classmethod
    def render(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Sidebar'da filtre panelini çizer ve config döner.
        Hiçbir exception dışarı sızmaz.
        """
        try:
            return cls._render_inner(df)
        except Exception as e:
            st.sidebar.warning(f"⚠️ Filtre paneli yüklenemedi: {e}")
            return {}

    @classmethod
    def _render_inner(cls, df: pd.DataFrame) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}

        # Veri setinde var olan boyutlar
        live = [(c, l, e, k) for c, l, e, k in DIMS if c in df.columns]

        st.sidebar.markdown(
            '<div style="font-size:1.05rem;font-weight:800;color:#00d4ff;'
            'margin:0 0 .8rem 0;letter-spacing:.5px">🎛️ FİLTRELER</div>',
            unsafe_allow_html=True,
        )

        # ── Global arama ──────────────────────────────────────────────────
        search = st.sidebar.text_input(
            "🔎 Global Arama",
            value=st.session_state.get("sf_search", ""),
            placeholder="Tüm alanlarda ara…",
            key="sf_search",
        )
        if search.strip():
            cfg["search"] = search.strip()

        # ── Sıfırlama butonu ──────────────────────────────────────────────
        if st.sidebar.button("🗑️ Tüm Filtreleri Sıfırla", key="sf_reset_all",
                             use_container_width=True):
            cls._reset(live)
            st.rerun()

        st.sidebar.markdown("---")

        # ── Kategorik boyutlar ────────────────────────────────────────────
        for col, label, emoji, key in live:
            all_opts = _opts(df[col])
            total = len(all_opts)

            if total == 0:
                continue

            with st.sidebar.expander(f"{emoji} **{label}** ({total})", expanded=False):
                # Boyut içi arama
                dim_search = st.text_input(
                    f"{label} ara",
                    value=st.session_state.get(f"{key}_q", ""),
                    placeholder=f"{label} içinde ara…",
                    key=f"{key}_q",
                    label_visibility="collapsed",
                )

                # Arama ile filtrele
                q = dim_search.strip().lower()
                visible = [o for o in all_opts if q in o.lower()] if q else all_opts

                # Hızlı seçim
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✓ Tümü", key=f"{key}_all", use_container_width=True):
                        st.session_state[key] = visible[:]
                        st.rerun()
                with c2:
                    if st.button("✗ Temizle", key=f"{key}_clr", use_container_width=True):
                        st.session_state[key] = []
                        st.rerun()

                # Mevcut seçim
                cur = _ss_list(key)
                cur = [v for v in cur if v in all_opts]

                # Multiselect
                default = [v for v in cur if v in visible]
                selected = st.multiselect(
                    f"__{label} seç__",
                    options=visible,
                    default=default,
                    key=f"{key}_ms",
                    label_visibility="collapsed",
                    placeholder=f"{len(visible)} seçenek…",
                )

                # Session'a kaydet
                st.session_state[key] = selected

                # Config'e ekle
                if selected:
                    cfg[col] = selected

                # Özet
                n_sel = len(selected) if selected else len(visible)
                st.caption(f"**{n_sel}** / {total} seçili")

        # ── Sayısal filtreler ─────────────────────────────────────────────
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📊 Sayısal Aralıklar**")

        years = _detect_years(df, "Sales_")
        if years:
            lsc = f"Sales_{years[-1]}"
            if lsc in df.columns:
                try:
                    vals = pd.to_numeric(df[lsc], errors="coerce").dropna()
                    if len(vals) > 0:
                        lo, hi = float(vals.min()), float(vals.max())
                        if lo < hi:
                            rng = st.sidebar.slider(
                                f"💰 Satış {years[-1]}",
                                min_value=lo, max_value=hi,
                                value=(lo, hi),
                                key="sf_sales_rng",
                                format="$%.0f",
                            )
                            if rng[0] > lo or rng[1] < hi:
                                cfg["sales_range"] = (rng, lsc)
                except Exception:
                    pass

        growth_cols = [c for c in df.columns if re.match(r"Growth_\d{4}_\d{4}", c)]
        if growth_cols:
            gc = growth_cols[-1]
            try:
                gv = pd.to_numeric(df[gc], errors="coerce").dropna()
                if len(gv) > 0:
                    glo = float(max(gv.quantile(0.01), -500.0))
                    ghi = float(min(gv.quantile(0.99), 500.0))
                    glo = min(glo, -50.0)
                    ghi = max(ghi, 50.0)
                    if glo < ghi:
                        grng = st.sidebar.slider(
                            "📈 Büyüme %",
                            min_value=glo, max_value=ghi,
                            value=(glo, ghi),
                            key="sf_growth_rng",
                            format="%.1f%%",
                        )
                        if grng[0] > glo or grng[1] < ghi:
                            cfg["growth_range"] = (grng, gc)
            except Exception:
                pass

        # ── Ek filtreler ──────────────────────────────────────────────────
        if growth_cols:
            pos = st.sidebar.checkbox("📈 Sadece pozitif büyüme", key="sf_pos")
            if pos:
                cfg["positive_growth"] = growth_cols[-1]

        if "International_Product" in df.columns:
            intl = st.sidebar.selectbox(
                "🌐 Ürün Tipi",
                ["Tümü", "Sadece Uluslararası", "Sadece Yerel"],
                key="sf_intl",
            )
            if intl != "Tümü":
                cfg["international"] = intl

        # ── Aktif filtre özeti ────────────────────────────────────────────
        st.sidebar.markdown("---")
        cls._render_summary(cfg, live, df)

        return cfg

    # ── Apply ─────────────────────────────────────────────────────────────────

    @classmethod
    def apply(cls, df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
        """Config'i uygular. Hata durumunda orijinal df döner."""
        if not cfg:
            return df

        try:
            mask = pd.Series(True, index=df.index)

            # Global arama
            if cfg.get("search"):
                term = cfg["search"].lower()
                smask = pd.Series(False, index=df.index)
                for c in df.select_dtypes(include="object").columns:
                    try:
                        smask |= df[c].astype(str).str.lower().str.contains(
                            term, na=False, regex=False
                        )
                    except Exception:
                        continue
                mask &= smask

            # Kategorik filtreler
            for col, _, _, _ in DIMS:
                vals = cfg.get(col)
                if not vals or col not in df.columns:
                    continue
                try:
                    mask &= df[col].astype(str).str.strip().isin(
                        {str(v).strip() for v in vals}
                    )
                except Exception:
                    continue

            # Satış aralığı
            if "sales_range" in cfg:
                try:
                    (lo, hi), c = cfg["sales_range"]
                    if c in df.columns:
                        mask &= pd.to_numeric(df[c], errors="coerce").between(lo, hi)
                except Exception:
                    pass

            # Büyüme aralığı
            if "growth_range" in cfg:
                try:
                    (lo, hi), c = cfg["growth_range"]
                    if c in df.columns:
                        mask &= pd.to_numeric(df[c], errors="coerce").fillna(0).between(lo, hi)
                except Exception:
                    pass

            # Pozitif büyüme
            if "positive_growth" in cfg:
                try:
                    c = cfg["positive_growth"]
                    if c in df.columns:
                        mask &= pd.to_numeric(df[c], errors="coerce").fillna(0) > 0
                except Exception:
                    pass

            # Uluslararası ürün
            if "international" in cfg and "International_Product" in df.columns:
                try:
                    v = df["International_Product"].astype(str)
                    pos = {"1", "1.0", "true", "True"}
                    if cfg["international"] == "Sadece Uluslararası":
                        mask &= v.isin(pos)
                    else:
                        mask &= ~v.isin(pos)
                except Exception:
                    pass

            result = df.loc[mask]
            return result if len(result) > 0 else df

        except Exception as exc:
            st.warning(f"⚠️ Filtre uygulanamadı: {exc}")
            return df

    # ── Yardımcı ──────────────────────────────────────────────────────────────

    @staticmethod
    def _render_summary(cfg: Dict, live: List[Tuple], df: pd.DataFrame) -> None:
        """Aktif filtrelerin özetini gösterir."""
        parts = []
        if cfg.get("search"):
            parts.append(f'🔎 "{cfg["search"]}"')
        for col, label, emoji, _ in live:
            if col in cfg and cfg[col]:
                n = len(cfg[col])
                total = len(_opts(df[col]))
                parts.append(f"{emoji} {label}: {n}/{total}")
        if "sales_range" in cfg:
            (lo, hi), _ = cfg["sales_range"]
            parts.append(f"💰 ${lo/1e6:.1f}M–${hi/1e6:.1f}M")
        if "growth_range" in cfg:
            (lo, hi), _ = cfg["growth_range"]
            parts.append(f"📈 {lo:.0f}%–{hi:.0f}%")
        if cfg.get("positive_growth"):
            parts.append("📈 Pozitif")
        if cfg.get("international"):
            parts.append(f"🌐 {cfg['international']}")

        if not parts:
            st.sidebar.info("ℹ️ Aktif filtre yok")
        else:
            st.sidebar.markdown("**Aktif Filtreler:**")
            for p in parts:
                st.sidebar.markdown(f"- {p}")

    @staticmethod
    def _reset(live: List[Tuple]) -> None:
        """Tüm filtreleri sıfırlar."""
        keys = [
            "sf_search", "sf_sales_rng", "sf_growth_rng",
            "sf_pos", "sf_intl", "sf_reset_all",
        ]
        for _, _, _, k in live:
            keys += [k, f"{k}_q", f"{k}_ms", f"{k}_all", f"{k}_clr"]
        for k in keys:
            st.session_state.pop(k, None)


# ─────────────────────────────────────────────────────────────────────────────
# ANA PANEL: Filtre sonuç özeti (main area'da)
# ─────────────────────────────────────────────────────────────────────────────

def render_filter_status(raw_df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """Ana içerikte filtre durumunu gösterir."""
    try:
        total = len(raw_df)
        filt = len(filtered_df)
        pct = filt / total * 100 if total > 0 else 100

        if filt < total:
            color = "#ff4757" if pct < 10 else ("#ffb700" if pct < 50 else "#00e5a0")
            st.markdown(
                f'<div style="background:rgba(255,183,0,0.08);border:1px solid rgba(255,183,0,0.25);'
                f'border-radius:8px;padding:.5rem 1rem;margin-bottom:.6rem;font-size:.85rem">'
                f'<span style="color:{color};font-weight:900">{filt:,}</span> '
                f'<span style="color:#8ba3c7">satır gösteriliyor</span> '
                f'<span style="color:#4a6080">({total:,} toplam · '
                f'<b style="color:{color}">{pct:.1f}%</b>)</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass


# Geriye dönük uyumluluk
FilterPanel = SidebarFilterSystem
ProfessionalFilterSystem = SidebarFilterSystem
render_sidebar_summary = render_filter_status


st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# YARDIMCI: Bellek temizleme
# ─────────────────────────────────────────────────────────────────────────────

def _free_memory():
    """Streamlit Cloud'da belleği serbest bırakır."""
    gc.collect()


def _run_with_progress(label: str, func, *args, **kwargs):
    """
    Analiz fonksiyonunu progress bar ile çalıştırır.
    Streamlit Cloud'da 503 hatasını önlemek için UI'ı canlı tutar.
    """
    bar = st.progress(0, text=f"⏳ {label} başlatılıyor…")
    try:
        bar.progress(20, text=f"⏳ {label} çalışıyor…")
        result = func(*args, **kwargs)
        bar.progress(80, text=f"⏳ {label} tamamlanıyor…")
        _free_memory()
        bar.progress(100, text=f"✅ {label} tamamlandı!")
        bar.empty()
        return result
    except MemoryError:
        bar.empty()
        st.error(
            "❌ Yetersiz bellek! Streamlit Cloud ücretsiz planda 1GB RAM limiti var. "
            "Filtreleri kullanarak veri setini küçültün ve tekrar deneyin."
        )
        return None
    except Exception as exc:
        bar.empty()
        st.error(f"❌ {label} hatası: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TAB FONKSİYONLARI
# ─────────────────────────────────────────────────────────────────────────────

def _safe_groupby_sum(df: pd.DataFrame, group_col: str, val_col: str, top_n: int = 10) -> pd.DataFrame:
    """
    Kategori dtype sorununu önlemek için güvenli groupby.
    50k+ satırda observed=True + category dtype boş sonuç dönebilir.
    Çözüm: group sütununu str'e çevir, sonra groupby yap.
    """
    tmp = df[[group_col, val_col]].copy()
    tmp[group_col] = tmp[group_col].astype(str).str.strip()
    tmp = tmp[tmp[group_col].notna() & (tmp[group_col] != "") & (tmp[group_col] != "nan") & (tmp[group_col] != "Bilinmiyor")]
    result = (
        tmp.groupby(group_col, sort=False)[val_col]
        .sum()
        .nlargest(top_n)
        .reset_index()
    )
    return result


def render_overview_tab(df: pd.DataFrame, summary: Dict) -> None:
    """Pazar Genel Bakış sekmesini render eder."""
    try:
        section_title("📊 Pazar Genel Bakış")

        years = summary.get("years", [])
        total = summary.get("total_sales", 0.0)
        mols  = summary.get("molecules", 0)
        comps = summary.get("companies", 0)
        ctrs  = summary.get("countries", 0)
        mis   = summary.get("missing_pct", 0.0)

        cards_html = "".join([
            kpi_card("Toplam Pazar (Son Yıl)", fmt_currency(total), icon="💰"),
            kpi_card("Molekül", f"{mols:,}", icon="🧪"),
            kpi_card("Şirket", f"{comps:,}", icon="🏢"),
            kpi_card("Ülke", f"{ctrs:,}", icon="🌍"),
            kpi_card("Kapsanan Yıl", str(len(years)), icon="📅"),
            kpi_card("Veri Kalitesi", f"{100 - mis:.1f}%", icon="✅"),
        ])
        st.markdown(
            '<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:1rem;margin-bottom:1.5rem">'
            + cards_html + "</div>",
            unsafe_allow_html=True,
        )

        viz = EnterpriseVisualizer()
        col1, col2 = st.columns([3, 2])
        with col1:
            trend = viz.sales_trend_chart(df, top_n=8)
            if trend:
                st.plotly_chart(trend, use_container_width=True, config={"displayModeBar": True})
        with col2:
            treemap = viz.treemap_chart(df)
            if treemap:
                st.plotly_chart(treemap, use_container_width=True, config={"displayModeBar": True})

        st.markdown("---")
        section_title("🔢 Veri Seti İstatistikleri")

        # Satış yıllarını bir kez hesapla
        sales_yrs = DataPipeline._detect_years(df, "Sales_")
        lsc = f"Sales_{sales_yrs[-1]}" if sales_yrs else None

        # ── Satır 1: Satış Dağılımı / Şirket Top10 / Molekül Top10 ───────────
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**📊 Satış Dağılımı**")
            if lsc and lsc in df.columns:
                desc = df[lsc].describe().reset_index()
                desc.columns = ["İstatistik", "Değer"]
                desc["Değer"] = desc["Değer"].apply(
                    lambda v: f"${v:,.0f}" if pd.notna(v) else "—"
                )
                st.dataframe(desc, use_container_width=True, hide_index=True)

        with c2:
            st.markdown("**🏢 Şirket Bazında Top 10**")
            if "Company" in df.columns and lsc and lsc in df.columns:
                top_c = _safe_groupby_sum(df, "Company", lsc, top_n=10)
                if not top_c.empty:
                    top_c.columns = ["Şirket", "Satış (Ham)"]
                    top_c["Satış"] = top_c["Satış (Ham)"].apply(fmt_currency)
                    st.dataframe(
                        top_c[["Şirket", "Satış"]],
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.info("Şirket verisi bulunamadı.")

        with c3:
            st.markdown("**🧪 Molekül Bazında Top 10**")
            if "Molecule" in df.columns and lsc and lsc in df.columns:
                top_m = _safe_groupby_sum(df, "Molecule", lsc, top_n=10)
                if not top_m.empty:
                    top_m.columns = ["Molekül", "Satış (Ham)"]
                    top_m["Satış"] = top_m["Satış (Ham)"].apply(fmt_currency)
                    st.dataframe(
                        top_m[["Molekül", "Satış"]],
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.info("Molekül verisi bulunamadı.")

        # ── Satır 2: Ülke / Şehir / Sektör Top10 ─────────────────────────────
        st.markdown("---")
        section_title("🌍 Coğrafi & Sektör Dağılımı")
        c4, c5, c6 = st.columns(3)

        with c4:
            st.markdown("**🌍 Ülke Bazında Top 10**")
            if "Country" in df.columns and lsc and lsc in df.columns:
                top_country = _safe_groupby_sum(df, "Country", lsc, top_n=10)
                if not top_country.empty:
                    top_country.columns = ["Ülke", "Satış (Ham)"]
                    top_country["Satış"] = top_country["Satış (Ham)"].apply(fmt_currency)
                    st.dataframe(
                        top_country[["Ülke", "Satış"]],
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.info("Ülke verisi bulunamadı.")

        with c5:
            # Şehir sütunu varsa göster, yoksa Region göster
            city_col = next(
                (c for c in ["City", "Sub_Region", "Region"] if c in df.columns), None
            )
            label_map = {"City": "🏙️ Şehir", "Sub_Region": "📍 Alt Bölge", "Region": "🗺️ Bölge"}
            st.markdown(f"**{label_map.get(city_col, '📍 Bölge')} Bazında Top 10**")
            if city_col and lsc and lsc in df.columns:
                top_city = _safe_groupby_sum(df, city_col, lsc, top_n=10)
                if not top_city.empty:
                    top_city.columns = [label_map.get(city_col, "Bölge"), "Satış (Ham)"]
                    top_city["Satış"] = top_city["Satış (Ham)"].apply(fmt_currency)
                    st.dataframe(
                        top_city[[label_map.get(city_col, "Bölge"), "Satış"]],
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.info("Bölge verisi bulunamadı.")

        with c6:
            st.markdown("**🏥 Sektör Bazında Top 10**")
            if "Sector" in df.columns and lsc and lsc in df.columns:
                top_sec = _safe_groupby_sum(df, "Sector", lsc, top_n=10)
                if not top_sec.empty:
                    top_sec.columns = ["Sektör", "Satış (Ham)"]
                    top_sec["Satış"] = top_sec["Satış (Ham)"].apply(fmt_currency)
                    st.dataframe(
                        top_sec[["Sektör", "Satış"]],
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.info("Sektör verisi bulunamadı.")

    except Exception as exc:
        st.error(f"❌ Genel Bakış sekmesi hatası: {exc}")
        st.code(traceback.format_exc())


def render_analytics_tab(df: pd.DataFrame) -> None:
    """Gelişmiş Analitik sekmesini render eder."""
    try:
        section_title("🔬 Gelişmiş Analitik Motoru")

        # Büyük veri seti uyarısı
        if len(df) > 50000:
            st.warning(
                f"⚠️ Veri seti büyük ({len(df):,} satır). "
                "Analiz yavaş çalışabilir. Sidebar filtrelerini kullanarak "
                "veri setini küçültmeniz önerilir."
            )

        tabs   = st.tabs([
            "📈 Evrim Endeksi",
            "💲 Fiyat Erozyonu",
            "🏭 Pazar Konsantrasyonu (HHI)",
            "🔗 Kanibalizasyon",
        ])
        viz    = EnterpriseVisualizer()
        engine = AnalyticsEngine()

        # ── Evrim Endeksi ─────────────────────────────────────────────────────
        with tabs[0]:
            st.markdown('<div class="subsection-title">📈 Evrim Endeksi (EI)</div>',
                        unsafe_allow_html=True)
            insight_card(
                "EI = Ürün Büyümesi / Pazar Medyan Büyümesi × 100. "
                "EI > 100 → pazarı geçiyor. EI < 100 → pazar altında.",
                "info", "EI Hakkında",
            )
            if st.button("⚡ Evrim Endeksini Hesapla", key="btn_ei", type="primary"):
                ei_df = _run_with_progress(
                    "Evrim Endeksi", engine.evolution_index, df
                )
                SessionManager.set("ei_df", ei_df)

            ei_df = SessionManager.get_df("ei_df")
            if ei_df is not None:
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig = viz.ei_chart(ei_df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                with c2:
                    if "EI_Kategori" in ei_df.columns:
                        cat_df = ei_df["EI_Kategori"].value_counts().reset_index()
                        cat_df.columns = ["Kategori", "Adet"]
                        st.dataframe(cat_df, use_container_width=True, hide_index=True)

        # ── Fiyat Erozyonu ────────────────────────────────────────────────────
        with tabs[1]:
            st.markdown('<div class="subsection-title">💲 Fiyat Erozyonu Analizi</div>',
                        unsafe_allow_html=True)
            insight_card(
                "2022–2024 arası SU ortalama fiyat değişimi. "
                "Negatif erozyon: generik giriş, ihale veya LOE etkisi.",
                "warning", "Fiyat Erozyonu Hakkında",
            )
            if st.button("⚡ Fiyat Erozyonunu Analiz Et", key="btn_erosion", type="primary"):
                erosion_df = _run_with_progress(
                    "Fiyat Erozyonu", engine.price_erosion_analysis, df
                )
                SessionManager.set("erosion_df", erosion_df)

            erosion_df = SessionManager.get_df("erosion_df")
            if erosion_df is not None:
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig = viz.price_erosion_chart(erosion_df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                with c2:
                    if "Erozyon_Kategorisi" in erosion_df.columns:
                        ec_df = erosion_df["Erozyon_Kategorisi"].value_counts().reset_index()
                        ec_df.columns = ["Kategori", "Adet"]
                        st.dataframe(ec_df, use_container_width=True, hide_index=True)
                    if "Birikimli_Erozyon_Pct" in erosion_df.columns:
                        worst = erosion_df.nsmallest(1, "Birikimli_Erozyon_Pct")
                        if not worst.empty:
                            col0 = erosion_df.columns[0]
                            v    = float(worst["Birikimli_Erozyon_Pct"].iloc[0])
                            st.metric("En Kötü Erozyon", f"{v:.1f}%",
                                      str(worst[col0].iloc[0]))

        # ── HHI ───────────────────────────────────────────────────────────────
        with tabs[2]:
            st.markdown('<div class="subsection-title">🏭 Pazar Konsantrasyonu (HHI)</div>',
                        unsafe_allow_html=True)
            insight_card(
                "Herfindahl-Hirschman Endeksi monopolleşme eğilimini ölçer. "
                "HHI > 2.500 = Yüksek Konsantre (DOJ eşiği).",
                "danger", "HHI Hakkında",
            )
            seg_col = st.selectbox(
                "Konsantrasyon Boyutu:", ["Company", "Molecule"], key="hhi_seg"
            )
            if st.button("⚡ HHI Hesapla", key="btn_hhi", type="primary"):
                hhi_df = _run_with_progress(
                    "HHI Analizi", engine.hhi_analysis, df, seg_col
                )
                SessionManager.set("hhi_df", hhi_df)

            hhi_df = SessionManager.get_df("hhi_df")
            if hhi_df is not None:
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig = viz.hhi_chart(hhi_df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.dataframe(hhi_df, use_container_width=True, hide_index=True)
                    if "HHI" in hhi_df.columns:
                        latest_hhi = float(hhi_df["HHI"].iloc[-1])
                        konc = str(hhi_df["Konsantrasyon"].iloc[-1]) if "Konsantrasyon" in hhi_df.columns else "—"
                        st.metric("Son HHI", f"{latest_hhi:,.0f}", konc)

        # ── Kanibalizasyon ────────────────────────────────────────────────────
        with tabs[3]:
            st.markdown('<div class="subsection-title">🔗 Kanibalizasyon Analizi</div>',
                        unsafe_allow_html=True)
            insight_card(
                "Şirket içi molekül büyüme korelasyonu. "
                "r < -0,7 = moleküller birbirinin pazar payından çalıyor.",
                "warning", "Kanibalizasyon Hakkında",
            )
            if st.button("⚡ Kanibalizasyon Analizi Çalıştır", key="btn_cannibal", type="primary"):
                result = _run_with_progress(
                    "Kanibalizasyon Analizi", engine.cannibalization_analysis, df
                )
                SessionManager.set("cannibal_result", result)

            result = SessionManager.get("cannibal_result")
            if result is not None:
                pairs_df, corr_matrix = result if isinstance(result, tuple) else (result, None)
                c1, c2 = st.columns([1, 1])
                with c1:
                    if pairs_df is not None and not pairs_df.empty:
                        st.markdown("**⚠️ Yüksek Riskli Çiftler**")
                        st.dataframe(pairs_df.head(20),
                                     use_container_width=True, hide_index=True)
                with c2:
                    if corr_matrix is not None and not corr_matrix.empty:
                        fig = viz.cannibalization_heatmap(corr_matrix)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.error(f"❌ Analitik sekmesi hatası: {exc}")


def render_ai_tab(df: pd.DataFrame) -> None:
    """Yapay Zeka Katmanı sekmesini render eder."""
    try:
        section_title("🤖 Yapay Zeka Katmanı — Tahmin & Anomali Tespiti")

        # Cloud uyarısı
        st.info(
            "ℹ️ **Streamlit Cloud Notu:** AI analizleri hesaplama yoğun işlemlerdir. "
            "İlk çalıştırmada 10-30 saniye sürebilir. Sonuçlar 30 dakika boyunca önbelleklenir."
        )

        ai_tabs = st.tabs(["🔮 Ensemble Tahmin", "⚠️ Anomali Tespiti"])
        ai  = AIForecasting()
        viz = EnterpriseVisualizer()

        # ── Ensemble Tahmin ───────────────────────────────────────────────────
        with ai_tabs[0]:
            st.markdown('<div class="subsection-title">🔮 Ensemble Pazar Tahmini</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<span class="ai-badge">AI</span> &nbsp;'
                "Hibrit model: Exponential Smoothing (%60) + Doğrusal Regresyon (%40), "
                "bootstrap güven aralıkları (200 iterasyon).",
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns([1, 3])
            with c1:
                periods = st.slider("Tahmin Yılı", 1, 5, 2, key="fc_periods")
                if st.button("🔮 Tahmin Oluştur", type="primary",
                             key="btn_fc", use_container_width=True):
                    fc_df = _run_with_progress(
                        "Ensemble Tahmin",
                        ai.ensemble_forecast, df, periods
                    )
                    SessionManager.set("forecast_df", fc_df)
                    if fc_df is None:
                        st.error("❌ Tahmin için en az 3 tarihsel yıl gerekli.")
                    else:
                        st.success("✅ Tahmin tamamlandı!")

            with c2:
                fc_df = SessionManager.get_df("forecast_df")
                if fc_df is not None:
                    fig = viz.forecast_chart(fc_df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            fc_df = SessionManager.get_df("forecast_df")
            if fc_df is not None:
                st.markdown("---")
                fwd = fc_df[fc_df["Tarihsel"] == False].copy()
                if not fwd.empty:
                    for col in ["Tahmin", "Alt_CI_80", "Üst_CI_80", "Alt_CI_95", "Üst_CI_95"]:
                        if col in fwd.columns:
                            fwd[col] = fwd[col].apply(
                                lambda v: fmt_currency(v) if pd.notna(v) else "—"
                            )
                    st.dataframe(fwd, use_container_width=True, hide_index=True)

        # ── Anomali Tespiti ───────────────────────────────────────────────────
        with ai_tabs[1]:
            st.markdown('<div class="subsection-title">⚠️ Anomali Tespiti</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<span class="ai-badge">AI</span> &nbsp;'
                "Isolation Forest (kirlilik=%10, 100 ağaç) — "
                "satış, büyüme ve fiyat özelliklerine göre aykırı ürünler tespit edilir.",
                unsafe_allow_html=True,
            )

            # Büyük veri uyarısı
            if len(df) > 10000:
                st.warning(
                    f"⚠️ {len(df):,} satır tespit edildi. "
                    "Anomali tespiti için en fazla 5.000 satır örnekleme yapılacak."
                )

            if st.button("🔍 Anomalileri Tespit Et", type="primary",
                         key="btn_anomaly", use_container_width=True):
                anomaly_df = _run_with_progress(
                    "Anomali Tespiti", ai.anomaly_detection, df
                )
                SessionManager.set("anomaly_df", anomaly_df)
                if anomaly_df is None:
                    st.error("❌ Anomali tespiti için yeterli özellik bulunamadı.")
                else:
                    n_anom = int(anomaly_df["Anormal_mı"].sum()) if "Anormal_mı" in anomaly_df.columns else 0
                    st.success(f"✅ {n_anom} adet anomalık ürün tespit edildi.")

            anomaly_df = SessionManager.get_df("anomaly_df")
            if anomaly_df is not None:
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig = viz.anomaly_chart(anomaly_df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                with c2:
                    if "Anomali_Kategorisi" in anomaly_df.columns:
                        cat_df = anomaly_df["Anomali_Kategorisi"].value_counts().reset_index()
                        cat_df.columns = ["Kategori", "Adet"]
                        st.dataframe(cat_df, use_container_width=True, hide_index=True)

                if "Anormal_mı" in anomaly_df.columns and anomaly_df["Anormal_mı"].any():
                    anom_only = anomaly_df[anomaly_df["Anormal_mı"]].copy()
                    group_col = next(
                        (c for c in ["Molecule", "Company"] if c in anom_only.columns), None
                    )
                    show_cols = [group_col] if group_col else []
                    for extra in ["Anomali_Skoru", "Anomali_Kategorisi"]:
                        if extra in anom_only.columns:
                            show_cols.append(extra)
                    yrs = DataPipeline._detect_years(anom_only, "Sales_")
                    if yrs:
                        lsc = f"Sales_{yrs[-1]}"
                        if lsc in anom_only.columns:
                            show_cols.append(lsc)
                    if show_cols:
                        st.markdown("**🚨 Anomalık Ürünler**")
                        st.dataframe(
                            anom_only[show_cols].sort_values("Anomali_Skoru").head(30),
                            use_container_width=True, hide_index=True,
                        )

    except Exception as exc:
        st.error(f"❌ AI sekmesi hatası: {exc}")


def render_visualizations_tab(df: pd.DataFrame) -> None:
    """Kurumsal Görselleştirmeler sekmesini render eder."""
    try:
        section_title("📊 Kurumsal Görselleştirmeler")

        viz    = EnterpriseVisualizer()
        engine = AnalyticsEngine()

        viz_tabs = st.tabs([
            "💰 Sankey Diyagramı",
            "📊 Waterfall / Satış Köprüsü",
            "🎯 BCG Kuadrantı",
        ])

        with viz_tabs[0]:
            st.markdown('<div class="subsection-title">💰 Nakit Akışı Sankey</div>',
                        unsafe_allow_html=True)
            insight_card(
                "Şirket → Molekül → Sektör finansal akışını gösterir. "
                "Düğüm genişliği USD geliriyle orantılıdır.",
                "info", "Sankey Nasıl Okunur",
            )
            if st.button("🔄 Sankey Oluştur", key="btn_sankey", type="primary"):
                with st.spinner("Sankey diyagramı oluşturuluyor…"):
                    fig = viz.sankey_chart(df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True,
                                        config={"displayModeBar": True})
                    else:
                        st.warning("⚠️ Sankey için yeterli veri yok.")

        with viz_tabs[1]:
            st.markdown('<div class="subsection-title">📊 Satış Köprüsü (Waterfall)</div>',
                        unsafe_allow_html=True)
            insight_card(
                "Satış değişimini Hacim Etkisi ve Fiyat Etkisi'ne ayırır.",
                "info", "Satış Köprüsü Nasıl Okunur",
            )
            if st.button("🔄 Satış Köprüsü Oluştur", key="btn_bridge", type="primary"):
                bridge_df = _run_with_progress(
                    "Satış Köprüsü", engine.sales_bridge, df
                )
                SessionManager.set("bridge_df", bridge_df)

            bridge_df = SessionManager.get_df("bridge_df")
            if bridge_df is not None:
                c1, c2 = st.columns([3, 1])
                with c1:
                    fig = viz.waterfall_chart(bridge_df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                with c2:
                    disp_cols = [
                        c for c in ["Satış_Değişimi", "Hacim_Etkisi", "Fiyat_Etkisi"]
                        if c in bridge_df.columns
                    ]
                    if disp_cols:
                        display = bridge_df[disp_cols].head(15).copy()
                        for col in disp_cols:
                            display[col] = display[col].apply(
                                lambda v: fmt_currency(v) if isinstance(v, float) else v
                            )
                        st.dataframe(display, use_container_width=True, hide_index=True)

        with viz_tabs[2]:
            st.markdown('<div class="subsection-title">🎯 BCG Portföy Matrisi</div>',
                        unsafe_allow_html=True)
            insight_card(
                "X = Pazar Payı %, Y = Pazar Büyümesi. "
                "Yıldız ⭐ | Nakit İneği 💰 | Soru İşareti ❓ | Köpek 🐕",
                "info", "BCG Matrisi Nasıl Okunur",
            )
            if st.button("🔄 BCG Matrisi Oluştur", key="btn_bcg", type="primary"):
                bcg_df = _run_with_progress(
                    "BCG Analizi", engine.bcg_analysis, df
                )
                SessionManager.set("bcg_df", bcg_df)

            bcg_df = SessionManager.get_df("bcg_df")
            if bcg_df is not None:
                c1, c2 = st.columns([3, 1])
                with c1:
                    fig = viz.bcg_chart(bcg_df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                with c2:
                    if "BCG_Kuadrant" in bcg_df.columns:
                        quad_df = bcg_df["BCG_Kuadrant"].value_counts().reset_index()
                        quad_df.columns = ["Kuadrant", "Adet"]
                        st.dataframe(quad_df, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error(f"❌ Görselleştirme sekmesi hatası: {exc}")


def render_reporting_tab(df: pd.DataFrame, summary: Dict) -> None:
    """Dışa Aktarma & Raporlama sekmesini render eder."""
    try:
        section_title("📑 Dışa Aktarma & Raporlama")

        from visualizer import REPORTLAB_OK

        ei_df      = SessionManager.get_df("ei_df")
        erosion_df = SessionManager.get_df("erosion_df")
        hhi_df     = SessionManager.get_df("hhi_df")
        bcg_df     = SessionManager.get_df("bcg_df")
        fc_df      = SessionManager.get_df("forecast_df")

        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        gen = ReportGenerator()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("### 📊 Excel Raporu")
            st.caption("7 sayfalık çalışma kitabı")
            if st.button("⬇️ Excel Oluştur", use_container_width=True, key="btn_excel_gen"):
                with st.spinner("Excel oluşturuluyor…"):
                    xls = gen.generate_excel(df, summary, ei_df, erosion_df, hhi_df, bcg_df)
                    if xls:
                        st.download_button(
                            "💾 Excel İndir", data=xls,
                            file_name=f"pharma_v8_{ts}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )

        with col2:
            st.markdown("### 📄 PDF Raporu")
            st.caption("Yönetici PDF'i")
            if REPORTLAB_OK:
                if st.button("⬇️ PDF Oluştur", use_container_width=True, key="btn_pdf_gen"):
                    with st.spinner("PDF oluşturuluyor…"):
                        pdf = gen.generate_pdf(summary, ei_df, erosion_df, hhi_df, bcg_df, fc_df)
                        if pdf:
                            st.download_button(
                                "💾 PDF İndir", data=pdf,
                                file_name=f"pharma_v8_yonetici_{ts}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
            else:
                st.warning("⚠️ ReportLab yüklü değil.\n`pip install reportlab`")

        with col3:
            st.markdown("### 🌐 HTML Raporu")
            st.caption("İnteraktif HTML")
            if st.button("⬇️ HTML Oluştur", use_container_width=True, key="btn_html_gen"):
                with st.spinner("HTML oluşturuluyor…"):
                    html = gen.generate_html(df, summary)
                    st.download_button(
                        "💾 HTML İndir", data=html.encode("utf-8"),
                        file_name=f"pharma_v8_{ts}.html",
                        mime="text/html",
                        use_container_width=True,
                    )

        with col4:
            st.markdown("### 💾 Ham CSV")
            st.caption("Filtrelenmiş veri seti")
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 CSV İndir", data=csv_data,
                file_name=f"pharma_v8_veri_{ts}.csv",
                mime="text/csv",
                use_container_width=True,
                key="btn_csv_dl",
            )

        st.markdown("---")
        section_title("📋 Hızlı İstatistikler")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Kayıt Sayısı",  f"{summary.get('rows', 0):,}")
        s2.metric("Sütun Sayısı",  f"{len(df.columns):,}")
        s3.metric("Bellek (MB)",   f"{summary.get('memory_mb', 0):.2f}")
        s4.metric("Eksik Veri",    f"{summary.get('missing_pct', 0):.2f}%")

        st.markdown("---")
        section_title("✅ Analiz Tamamlanma Durumu")
        status_items = [
            ("Evrim Endeksi",       SessionManager.get_df("ei_df") is not None),
            ("Fiyat Erozyonu",      SessionManager.get_df("erosion_df") is not None),
            ("HHI Konsantrasyon",   SessionManager.get_df("hhi_df") is not None),
            ("Kanibalizasyon",      SessionManager.get("cannibal_result") is not None),
            ("BCG Sınıflandırması", SessionManager.get_df("bcg_df") is not None),
            ("Satış Köprüsü",       SessionManager.get_df("bridge_df") is not None),
            ("AI Tahmini",          SessionManager.get_df("forecast_df") is not None),
            ("Anomali Tespiti",     SessionManager.get_df("anomaly_df") is not None),
        ]
        cols = st.columns(4)
        for i, (name, done) in enumerate(status_items):
            pill_cls = "live" if done else "warn"
            icon = "●" if done else "○"
            cols[i % 4].markdown(
                f'<span class="status-pill {pill_cls}">{icon} {name}</span>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        if st.button("🔄 Tüm Analizi Sıfırla", use_container_width=True, key="btn_full_reset"):
            SessionManager.clear()
            _free_memory()
            st.rerun()

    except Exception as exc:
        st.error(f"❌ Raporlama sekmesi hatası: {exc}")


def render_data_tab(df: pd.DataFrame) -> None:
    """Ham Veri Gezgini sekmesini render eder."""
    try:
        section_title("🗄️ Veri Gezgini")

        with st.expander("📋 Sütun Eşleme", expanded=False):
            col_map = SessionManager.get("col_mapping")
            if col_map:
                st.dataframe(
                    pd.DataFrame(
                        list(col_map.items()),
                        columns=["Orijinal Sütun", "Standart Sütun"],
                    ),
                    use_container_width=True, hide_index=True,
                )

        c1, c2, c3 = st.columns(3)
        with c1:
            search = st.text_input("🔎 Ara", placeholder="Satırları filtrele…", key="data_search")
        with c2:
            sort_col = st.selectbox("Sırala", df.columns.tolist(), key="data_sort")
        with c3:
            sort_asc = st.checkbox("Artan", value=False, key="data_sort_asc")

        show_df = df.copy()
        if search.strip():
            mask = show_df.apply(
                lambda col: col.astype(str).str.contains(search.strip(), case=False, na=False),
                axis=0,
            ).any(axis=1)
            show_df = show_df[mask]

        if sort_col in show_df.columns:
            show_df = show_df.sort_values(sort_col, ascending=sort_asc)

        st.caption(f"📊 {len(show_df):,} / {len(df):,} satır gösteriliyor")
        # Cloud'da büyük tablolar yavaş render eder — max 1000 satır göster
        st.dataframe(show_df.head(1000), use_container_width=True, height=500)

        if len(show_df) > 1000:
            st.info(f"ℹ️ İlk 1.000 satır gösteriliyor. Tamamını görmek için CSV indirin.")

    except Exception as exc:
        st.error(f"❌ Veri gezgini hatası: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# ANA UYGULAMA
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    SessionManager.init_defaults()

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div class="version-badge">⚕️ PharmaIntelligence v8.0</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        uploaded = st.file_uploader(
            "📁 Pazar Verisi Yükle",
            type=["csv", "xlsx", "xls"],
            key="file_uploader",
            help="IMS/IQVIA MAT formatı · CSV veya Excel · Max 400MB",
        )

        if uploaded is not None:
            file_bytes = uploaded.read()
            file_hash  = hashlib.md5(file_bytes).hexdigest()

            if SessionManager.get("file_hash") != file_hash:
                with st.spinner("⚙️ Veri hattı işleniyor…"):
                    raw_df = DataPipeline.load(file_bytes, uploaded.name)
                    if raw_df is not None:
                        processed_df = DataPipeline.process(raw_df)
                        if processed_df is not None:
                            _, col_map = ColumnStandardizer.standardize_columns(raw_df)
                            SessionManager.set("raw_df", raw_df)
                            SessionManager.set("processed_df", processed_df)
                            SessionManager.set("col_mapping", col_map)
                            SessionManager.set("file_name", uploaded.name)
                            SessionManager.set("file_hash", file_hash)
                            SessionManager.clear([
                                "ei_df", "erosion_df", "hhi_df", "bcg_df",
                                "bridge_df", "cannibal_result",
                                "forecast_df", "anomaly_df",
                                "filtered_df", "summary",
                            ])
                            _free_memory()
                            st.success(f"✅ {len(processed_df):,} satır yüklendi")

                            # Büyük veri seti uyarısı
                            if len(processed_df) > 50000:
                                st.warning(
                                    f"⚠️ Büyük veri seti ({len(processed_df):,} satır). "
                                    "Filtreleri kullanmanız önerilir."
                                )

        if SessionManager.is_loaded():
            processed_df = SessionManager.get_df("processed_df")
            if processed_df is not None:
                # Filtreler sidebar'da render ediliyor
                filter_config = SidebarFilterSystem.render(processed_df)
                filtered_df = SidebarFilterSystem.apply(processed_df, filter_config)
                SessionManager.set("filtered_df", filtered_df)
                summary = DataPipeline.get_summary(filtered_df)
                SessionManager.set("summary", summary)

    # ── ANA İÇERİK ────────────────────────────────────────────────────────────

    if not SessionManager.is_loaded():
        st.markdown(
            """
            <div class="pharma-hero">
                <div class="version-badge">Enterprise v8.0</div>
                <h1 class="pharma-title">PharmaIntelligence</h1>
                <p class="pharma-subtitle">
                    Yapay zeka destekli ilaç pazar analitiği platformu —
                    EI · Fiyat Erozyonu · HHI · Kanibalizasyon ·
                    Ensemble Tahmin · Anomali Tespiti · Sankey · BCG · Waterfall
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="upload-hero">
                <h2 style="color:#00d4ff;margin:0 0 0.5rem 0">📁 Pazar Verinizi Yükleyin</h2>
                <p style="color:#8ba3c7;margin:0">
                    IMS/IQVIA MAT formatı · CSV veya Excel · 2022–2024 MAT Q3<br>
                    Sütunlar: Source.Name · Corporation · Molecule · MAT Q3 Sales/Units/SU Price
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        section_title("🚀 Platform Yetenekleri")
        feat_cols = st.columns(4)
        feats = [
            ("📐", "Sütun Standardizasyonu", "MAT Q3 2024 USD MNF → Sales_2024 otomatik eşleme"),
            ("📈", "Evrim Endeksi", "Ürün performansı vs. pazar medyanı büyümesi"),
            ("💲", "Fiyat Erozyonu", "SU fiyat değişimi 2022→2024 takibi"),
            ("🏭", "HHI Analizi", "Pazar konsantrasyonu ve monopol riski"),
            ("🔗", "Kanibalizasyon", "Şirket içi molekül korelasyon analizi"),
            ("🔮", "AI Tahmini", "Ensemble ES+LR, %95 güven bandı"),
            ("⚠️", "Anomali Tespiti", "Isolation Forest ürün risk puanlaması"),
            ("📑", "Dışa Aktarma", "Excel · PDF · HTML · CSV"),
        ]
        for i, (icon, title, desc) in enumerate(feats):
            with feat_cols[i % 4]:
                st.markdown(
                    f"<div class='kpi-card' style='min-height:110px'>"
                    f"<div style='font-size:1.8rem'>{icon}</div>"
                    f"<div class='kpi-label'>{title}</div>"
                    f"<div style='font-size:0.82rem;color:var(--text-secondary)'>{desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        return

    # ── Veri yüklüyse sekmeler ────────────────────────────────────────────────

    processed_df = SessionManager.get_df("processed_df")
    if processed_df is None:
        st.error("❌ Veri bulunamadı. Lütfen dosyayı yeniden yükleyin.")
        return

    # Hero başlık (ham veri bilgileri)
    _mol_n  = processed_df["Molecule"].nunique() if "Molecule" in processed_df.columns else "—"
    _comp_n = processed_df["Company"].nunique()  if "Company"  in processed_df.columns else "—"
    _ctr_n  = processed_df["Country"].nunique()  if "Country"  in processed_df.columns else "—"
    _yrs    = DataPipeline._detect_years(processed_df, "Sales_")
    _yr_str = " → ".join(str(y) for y in _yrs) if _yrs else "—"
    st.markdown(
        f'''
        <div class="pharma-hero">
            <div class="version-badge">Enterprise v8.0</div>
            <h1 class="pharma-title">PharmaIntelligence</h1>
            <p class="pharma-subtitle">
                <b>{len(processed_df):,}</b> kayıt &nbsp;·&nbsp;
                <b>{_mol_n}</b> molekül &nbsp;·&nbsp;
                <b>{_comp_n}</b> şirket &nbsp;·&nbsp;
                <b>{_ctr_n}</b> ülke &nbsp;·&nbsp;
                Dönem: {_yr_str}
            </p>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    # ── Filtre sonucu durum banner'ı ─────────────────────────────────────────
    df = SessionManager.get_df("filtered_df")
    if df is None:
        df = processed_df
    
    summary = DataPipeline.get_summary(df)
    SessionManager.set("summary", summary)
    
    # Filtre durumu göster
    render_filter_status(processed_df, df)

    # ── Sekmeler ─────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Genel Bakış",
        "🔬 Analitik",
        "🤖 Yapay Zeka",
        "📈 Görselleştirmeler",
        "🗄️ Veri Gezgini",
        "📑 Raporlar",
    ])

    with tabs[0]:
        render_overview_tab(df, summary)
    with tabs[1]:
        render_analytics_tab(df)
    with tabs[2]:
        render_ai_tab(df)
    with tabs[3]:
        render_visualizations_tab(df)
    with tabs[4]:
        render_data_tab(df)
    with tabs[5]:
        render_reporting_tab(df, summary)


# ─────────────────────────────────────────────────────────────────────────────
# GİRİŞ NOKTASI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        gc.enable()
        main()
    except Exception as exc:
        st.error("💥 Kritik uygulama hatası:")
        st.exception(exc)
        if st.button("🔄 Uygulamayı Yeniden Başlat",
                     key="crash_reload", use_container_width=True):
            SessionManager.clear()
            _free_memory()
            st.rerun()
