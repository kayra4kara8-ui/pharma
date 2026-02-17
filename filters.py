"""
PharmaIntelligence Enterprise v8.0 — filters.py  v2.0
──────────────────────────────────────────────────────
Tamamen yeniden yazıldı.

Mimari kararlar:
  • st.expander KULLANILMADI  → expander içi button/multiselect Cloud'da
                                 session state döngüsü oluşturur
  • Buton→multiselect default KULLANILMADI → widget key çakışması
  • Tümü Seç / Temizle → st.checkbox ile (button değil)
  • apply() tamamen vektörel, 50k+ satırda hızlı
  • Her adımda try/except → uygulama asla çökmez
"""

import re
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# TANIMLAR
# ─────────────────────────────────────────────────────────────────────────────

DIMS: List[Tuple[str, str, str, str]] = [
    ("Country",    "Ulke",      "🌍", "pf_country"),
    ("City",       "Sehir",     "🏙️","pf_city"),
    ("Company",    "Sirket",    "🏢", "pf_company"),
    ("Molecule",   "Molekul",   "🧪", "pf_molecule"),
    ("Sector",     "Sektor",    "🏥", "pf_sector"),
    ("Region",     "Bolge",     "🗺️","pf_region"),
    ("Sub_Region", "Alt Bolge", "📍", "pf_subregion"),
    ("Specialty",  "Uzmanlik",  "💊", "pf_specialty"),
    ("NFC123",     "NFC123",    "🔬", "pf_nfc123"),
]

_JUNK = {"", "nan", "none", "bilinmiyor", "unknown", "n/a", "-", "null", "na"}

_CSS = """<style>
.pf-title{font-size:1rem;font-weight:800;color:#00d4ff;margin:0 0 .8rem 0;letter-spacing:.4px;}
.pf-dim-lbl{font-size:.67rem;font-weight:800;text-transform:uppercase;letter-spacing:1.2px;color:#8ba3c7;margin-bottom:.3rem;}
.pf-chips{display:flex;flex-wrap:wrap;gap:.22rem;margin-top:.35rem;min-height:20px;}
.pf-chip{display:inline-block;background:rgba(0,112,224,.2);border:1px solid rgba(0,212,255,.3);
  color:#00d4ff;font-size:.67rem;font-weight:700;padding:.08rem .4rem;border-radius:20px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:120px;}
.pf-chip-all{background:rgba(0,229,160,.1);border-color:rgba(0,229,160,.3);color:#00e5a0;}
.pf-sep{height:1px;background:rgba(0,212,255,.1);margin:.9rem 0;}
.pf-active{background:linear-gradient(135deg,rgba(0,212,255,.06),rgba(123,47,255,.06));
  border:1px solid rgba(0,212,255,.14);border-radius:8px;padding:.5rem .9rem;
  font-size:.8rem;color:#8ba3c7;margin-top:.4rem;}
.pf-tag{display:inline-block;background:rgba(255,183,0,.1);border:1px solid rgba(255,183,0,.28);
  color:#ffb700;font-size:.67rem;font-weight:700;padding:.06rem .35rem;border-radius:20px;margin:.08rem .12rem;}
</style>"""


# ─────────────────────────────────────────────────────────────────────────────
# YARDIMCI
# ─────────────────────────────────────────────────────────────────────────────

def _opts(s: pd.Series) -> List[str]:
    try:
        v = s.astype(str).str.strip()
        return sorted(v[~v.str.lower().isin(_JUNK)].unique().tolist())
    except Exception:
        return []

def _ss_list(key: str) -> List[str]:
    v = st.session_state.get(key, [])
    return v if isinstance(v, list) else []

def _chips(sel: List[str], total: int) -> str:
    if not sel or len(sel) == total:
        return '<div class="pf-chips"><span class="pf-chip pf-chip-all">✓ Tümü</span></div>'
    html = "".join(
        f'<span class="pf-chip" title="{v}">{v[:15]}{"…" if len(v)>15 else ""}</span>'
        for v in sel[:5]
    )
    if len(sel) > 5:
        html += f'<span class="pf-chip">+{len(sel)-5}</span>'
    return f'<div class="pf-chips">{html}</div>'

def _detect_years(df: pd.DataFrame, prefix: str) -> List[int]:
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
# ANA SINIF
# ─────────────────────────────────────────────────────────────────────────────

class FilterPanel:

    @classmethod
    def render(cls, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            st.markdown(_CSS, unsafe_allow_html=True)
            return cls._inner(df)
        except Exception as e:
            st.warning(f"⚠️ Filtre paneli yüklenemedi: {e}")
            return {}

    @classmethod
    def _inner(cls, df: pd.DataFrame) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        live = [(c,l,e,k) for c,l,e,k in DIMS if c in df.columns]

        st.markdown('<div class="pf-title">🎛️ Veri Filtreleme Paneli</div>',
                    unsafe_allow_html=True)

        # ── Üst çubuk: global arama + sıfırla ────────────────────────────
        gc1, gc2, gc3 = st.columns([5, 1, 1])
        with gc1:
            search = st.text_input(
                "_srch_", key="pf_search",
                placeholder="🔎  Tüm alanlarda ara — molekül, şirket, ülke, şehir…",
                label_visibility="collapsed",
            )
        with gc2:
            if st.button("🗑️ Sıfırla", key="pf_rst", use_container_width=True):
                cls._reset(live); st.rerun()
        with gc3:
            st.markdown(
                f'<div style="text-align:center;padding:.4rem 0">'
                f'<div style="color:#00d4ff;font-size:1rem;font-weight:900">{len(df):,}</div>'
                f'<div style="color:#4a6080;font-size:.68rem">satır</div></div>',
                unsafe_allow_html=True,
            )
        if search.strip():
            cfg["search"] = search.strip()

        st.markdown('<div class="pf-sep"></div>', unsafe_allow_html=True)

        # ── Boyut sekmeleri ────────────────────────────────────────────────
        if not live:
            st.info("Filtrelenebilir boyut bulunamadı.")
            return cfg

        tabs = st.tabs([f"{e} {l}" for _,l,e,_ in live])
        for tab, (col, label, emoji, key) in zip(tabs, live):
            with tab:
                try:
                    cls._dim_tab(df, col, label, key, cfg)
                except Exception as ex:
                    st.warning(f"⚠️ {label}: {ex}")

        # ── Sayısal filtreler ──────────────────────────────────────────────
        st.markdown('<div class="pf-sep"></div>', unsafe_allow_html=True)
        cls._numeric(df, cfg)

        # ── Aktif filtre özeti ─────────────────────────────────────────────
        cls._summary(cfg, live, df)
        return cfg

    # ── Boyut sekmesi ─────────────────────────────────────────────────────────

    @classmethod
    def _dim_tab(cls, df, col, label, key, cfg):
        all_opts = _opts(df[col])
        total    = len(all_opts)
        if total == 0:
            st.info(f"{label} için veri bulunamadı.")
            return

        # Arama + checkbox satırı
        sc1, sc2, sc3 = st.columns([4, 1, 1])
        with sc1:
            q = st.text_input(
                f"_q_{key}_", key=f"{key}_q",
                placeholder=f"{label} içinde ara… ({total} seçenek mevcut)",
                label_visibility="collapsed",
            )
        with sc2:
            chk_all = st.checkbox("✓ Tümü", key=f"{key}_ca")
        with sc3:
            chk_clr = st.checkbox("✗ Temizle", key=f"{key}_cc")

        # Aramayla filtrele
        visible = (
            [o for o in all_opts if q.strip().lower() in o.lower()]
            if q.strip() else all_opts
        )

        # Mevcut seçim — sadece geçerli değerler
        cur = [v for v in _ss_list(key) if v in all_opts]

        # Checkbox → state güncelle
        if chk_all and not chk_clr:
            cur = visible[:]
            st.session_state[key] = cur
        elif chk_clr:
            cur = []
            st.session_state[key] = cur

        # Multiselect — default yalnızca görünür ∩ mevcut seçim
        default = [v for v in cur if v in visible]
        selected = st.multiselect(
            f"_ms_{label}_", options=visible, default=default,
            key=f"{key}_ms", label_visibility="collapsed",
            placeholder=f"Seçin veya yazın… ({len(visible)} seçenek görünüyor)",
        )
        st.session_state[key] = selected
        if selected:
            cfg[col] = selected

        # Özet satırı
        oc1, oc2 = st.columns([3, 1])
        with oc1:
            st.markdown(_chips(selected, total), unsafe_allow_html=True)
        with oc2:
            n_sel = len(selected) if selected else total
            st.markdown(
                f'<div style="text-align:right;color:#4a6080;font-size:.73rem;padding-top:.25rem">'
                f'<b style="color:#e8f0fe">{n_sel}</b>/{total}</div>',
                unsafe_allow_html=True,
            )

        # Dağılım önizleme
        yrs = _detect_years(df, "Sales_")
        if yrs:
            lsc = f"Sales_{yrs[-1]}"
            if lsc in df.columns:
                with st.expander(f"📊 Top 10 — {label} bazında satış", expanded=False):
                    try:
                        tmp = df[[col, lsc]].copy()
                        tmp[col] = tmp[col].astype(str).str.strip()
                        tmp[lsc] = pd.to_numeric(tmp[lsc], errors="coerce").fillna(0)
                        top = tmp.groupby(col, sort=False)[lsc].sum().nlargest(10).reset_index()
                        top.columns = [label, f"Satış {yrs[-1]}"]
                        top[f"Satış {yrs[-1]}"] = top[f"Satış {yrs[-1]}"].apply(
                            lambda v: f"${v/1e6:.2f}M" if v >= 1e6 else f"${v:,.0f}"
                        )
                        st.dataframe(top, use_container_width=True, hide_index=True)
                    except Exception:
                        pass

    # ── Sayısal filtreler ─────────────────────────────────────────────────────

    @staticmethod
    def _numeric(df, cfg):
        yrs         = _detect_years(df, "Sales_")
        growth_cols = [c for c in df.columns if re.match(r"Growth_\d{4}_\d{4}", c)]
        if not yrs and not growth_cols:
            return

        st.markdown(
            '<div style="font-size:.67rem;font-weight:800;text-transform:uppercase;'
            'letter-spacing:1.2px;color:#8ba3c7;margin-bottom:.5rem">📊 Sayısal Aralıklar</div>',
            unsafe_allow_html=True,
        )
        nc1, nc2 = st.columns(2)

        with nc1:
            if yrs:
                lsc = f"Sales_{yrs[-1]}"
                if lsc in df.columns:
                    try:
                        vals = pd.to_numeric(df[lsc], errors="coerce").dropna()
                        lo_a, hi_a = float(vals.min()), float(vals.max())
                        if lo_a < hi_a:
                            st.markdown(
                                f'<div class="pf-dim-lbl">💰 Satış {yrs[-1]} (USD)</div>',
                                unsafe_allow_html=True,
                            )
                            rng = st.slider(
                                "_srng_", min_value=lo_a, max_value=hi_a,
                                value=(lo_a, hi_a), key="pf_srng",
                                label_visibility="collapsed", format="$%.0f",
                            )
                            if rng[0] > lo_a or rng[1] < hi_a:
                                cfg["sales_range"] = (rng, lsc)
                    except Exception:
                        pass

        with nc2:
            if growth_cols:
                gc = growth_cols[-1]
                try:
                    gv = pd.to_numeric(df[gc], errors="coerce").dropna()
                    glo = float(max(gv.quantile(0.01), -500.0))
                    ghi = float(min(gv.quantile(0.99),  500.0))
                    glo, ghi = min(glo, -50.0), max(ghi, 50.0)
                    if glo < ghi:
                        st.markdown(
                            '<div class="pf-dim-lbl">📈 Büyüme % Aralığı</div>',
                            unsafe_allow_html=True,
                        )
                        grng = st.slider(
                            "_grng_", min_value=glo, max_value=ghi,
                            value=(glo, ghi), key="pf_grng",
                            label_visibility="collapsed", format="%.1f%%",
                        )
                        if grng[0] > glo or grng[1] < ghi:
                            cfg["growth_range"] = (grng, gc)
                except Exception:
                    pass

        bc1, bc2 = st.columns(2)
        with bc1:
            if growth_cols and st.checkbox("📈 Sadece pozitif büyüme", key="pf_pos"):
                cfg["positive_growth"] = growth_cols[-1]
        with bc2:
            if "International_Product" in df.columns:
                intl = st.selectbox(
                    "🌐 Ürün tipi", ["Tümü", "Sadece Uluslararası", "Sadece Yerel"],
                    key="pf_intl", label_visibility="collapsed",
                )
                if intl != "Tümü":
                    cfg["international"] = intl

    # ── Özet ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _summary(cfg, live, df):
        parts = []
        if cfg.get("search"):
            parts.append(f'🔎 "{cfg["search"]}"')
        for col, label, emoji, _ in live:
            if col in cfg and cfg[col]:
                n = len(cfg[col]); tot = len(_opts(df[col]))
                parts.append(f"{emoji} {label}: {n}/{tot}")
        if "sales_range" in cfg:
            (lo, hi), _ = cfg["sales_range"]
            parts.append(f"💰 ${lo/1e6:.1f}M–${hi/1e6:.1f}M")
        if "growth_range" in cfg:
            (lo, hi), _ = cfg["growth_range"]
            parts.append(f"📈 {lo:.0f}%–{hi:.0f}%")
        if cfg.get("positive_growth"): parts.append("📈 Pozitif")
        if cfg.get("international"):   parts.append(f"🌐 {cfg['international']}")

        if not parts:
            st.markdown(
                '<div class="pf-active" style="color:#4a6080">'
                'ℹ️ Aktif filtre yok — tüm veri gösteriliyor</div>',
                unsafe_allow_html=True,
            )
        else:
            tags = "".join(f'<span class="pf-tag">{p}</span>' for p in parts)
            st.markdown(
                f'<div class="pf-active"><b style="color:#e8f0fe">Aktif filtreler:</b> {tags}</div>',
                unsafe_allow_html=True,
            )

    # ── Apply ─────────────────────────────────────────────────────────────────

    @classmethod
    def apply(cls, df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
        if not cfg:
            return df
        try:
            mask = pd.Series(True, index=df.index)

            if cfg.get("search"):
                term  = cfg["search"].lower()
                smask = pd.Series(False, index=df.index)
                for c in df.select_dtypes(include="object").columns:
                    try:
                        smask |= df[c].astype(str).str.lower().str.contains(
                            term, na=False, regex=False
                        )
                    except Exception:
                        continue
                mask &= smask

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

            if "sales_range" in cfg:
                try:
                    (lo, hi), c = cfg["sales_range"]
                    if c in df.columns:
                        mask &= pd.to_numeric(df[c], errors="coerce").between(lo, hi)
                except Exception:
                    pass

            if "growth_range" in cfg:
                try:
                    (lo, hi), c = cfg["growth_range"]
                    if c in df.columns:
                        mask &= pd.to_numeric(df[c], errors="coerce").fillna(0).between(lo, hi)
                except Exception:
                    pass

            if "positive_growth" in cfg:
                try:
                    c = cfg["positive_growth"]
                    if c in df.columns:
                        mask &= pd.to_numeric(df[c], errors="coerce").fillna(0) > 0
                except Exception:
                    pass

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

    # ── Sıfırla ───────────────────────────────────────────────────────────────

    @staticmethod
    def _reset(live):
        keys = ["pf_search", "pf_srng", "pf_grng", "pf_pos", "pf_intl", "pf_rst"]
        for _, _, _, k in live:
            keys += [k, f"{k}_q", f"{k}_ms", f"{k}_ca", f"{k}_cc"]
        for k in keys:
            st.session_state.pop(k, None)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR ÖZET (hafif — widget yok)
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar_summary(raw_df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    try:
        total = len(raw_df); filt = len(filtered_df)
        pct   = filt / total * 100 if total > 0 else 100
        col   = "#ff4757" if pct < 10 else ("#ffb700" if pct < 50 else "#00e5a0")
        st.sidebar.markdown(
            f'<div style="background:rgba(13,31,60,.8);border:1px solid rgba(0,212,255,.15);'
            f'border-radius:10px;padding:.75rem 1rem;margin-top:.5rem">'
            f'<div style="color:#8ba3c7;font-size:.67rem;text-transform:uppercase;'
            f'letter-spacing:1px;margin-bottom:.25rem">📊 Filtre Durumu</div>'
            f'<div style="color:{col};font-size:1.3rem;font-weight:900">{filt:,}'
            f'<span style="font-size:.78rem;color:#8ba3c7;font-weight:400"> satır</span></div>'
            f'<div style="color:#4a6080;font-size:.7rem">{total:,} içinden '
            f'<b style="color:{col}">{pct:.1f}%</b></div></div>',
            unsafe_allow_html=True,
        )
        lines = []
        for col_name, label, emoji, _ in DIMS:
            if col_name in filtered_df.columns:
                n = filtered_df[col_name].astype(str).nunique()
                lines.append(f"{emoji} **{n:,}** {label}")
        if lines:
            st.sidebar.markdown("  \n".join(lines))
    except Exception:
        pass


# Geriye dönük uyumluluk takma adı
ProfessionalFilterSystem = FilterPanel
