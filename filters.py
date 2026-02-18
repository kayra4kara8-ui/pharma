"""
PharmaIntelligence Enterprise v8.0 — filters.py v3.0
────────────────────────────────────────────────────
Sidebar filtre sistemi — sol panel, tam seçenek listesi

Özellikler:
  ✅ Tüm filtreler sidebar'da (sol tarafta)
  ✅ Her boyut için arama kutusu — sınırsız seçenek
  ✅ Daraltılabilir gruplar — yer tasarrufu
  ✅ Tüm moleküller / şehirler / ülkeler görünür
  ✅ 50k+ satırda hızlı, güvenli
  ✅ Streamlit Cloud uyumlu
"""

import re
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

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
