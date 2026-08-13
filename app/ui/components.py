
"""
EstateRAG AI - Professional Streamlit UI Components

UI-only layer.
Retrieval, RAG, dataset, FAISS and business logic are intentionally unchanged.
"""

import streamlit as st
import pandas as pd
from typing import List

from app.data.schema import PropertyItem
from app.retrieval.hybrid_retriever import PropertyMatch
from app.utils.image_config import DEFAULT_PLACEHOLDER_IMAGE, get_property_image


CUSTOM_CSS = """
<style>

/* =========================================================
   ESTATERAG AI - PREMIUM APPLICATION SHELL
   ========================================================= */

:root {
    --bg: #07111f;
    --bg-soft: #0b1626;
    --card: #101d30;
    --card-2: #132238;
    --border: rgba(148, 163, 184, 0.16);
    --border-blue: rgba(56, 189, 248, 0.32);
    --text: #f8fafc;
    --muted: #94a3b8;
    --muted-2: #64748b;
    --blue: #38bdf8;
    --blue-2: #60a5fa;
    --green: #34d399;
}

/* Main background */
.stApp {
    background:
        radial-gradient(circle at 12% 0%, rgba(37, 99, 235, 0.14), transparent 27%),
        radial-gradient(circle at 88% 8%, rgba(14, 165, 233, 0.10), transparent 25%),
        linear-gradient(180deg, #07111f 0%, #08121f 48%, #060e19 100%);
    color: var(--text);
}

/* Prevent the default bright Streamlit header from dominating */
[data-testid="stHeader"] {
    background: rgba(7, 17, 31, 0.80);
}

/* Main content */
.block-container {
    max-width: 1480px;
    padding: 2.2rem 2.5rem 4rem;
}

/* Typography */
html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                 BlinkMacSystemFont, "Segoe UI", sans-serif;
}

h1, h2, h3, h4 {
    color: #f8fafc !important;
    letter-spacing: -0.025em;
}

h1 {
    font-size: 2.55rem !important;
    font-weight: 800 !important;
}

h2 {
    font-size: 1.85rem !important;
    font-weight: 780 !important;
}

h3 {
    font-size: 1.35rem !important;
    font-weight: 760 !important;
}

/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #081321 0%, #07101c 55%, #050c16 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.25rem;
}

[data-testid="stSidebar"] .stRadio label {
    border-radius: 12px;
    padding: 9px 11px;
    margin: 2px 0;
    transition: all .18s ease;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(56, 189, 248, 0.08);
}

.sidebar-brand {
    padding: 8px 5px 20px;
}

.sidebar-brand-title {
    font-size: 1.55rem;
    font-weight: 850;
    color: #f8fafc;
    letter-spacing: -0.04em;
}

.sidebar-brand-title span {
    color: #38bdf8;
}

.sidebar-brand-subtitle {
    color: #64748b;
    font-size: .70rem;
    margin-top: 5px;
    letter-spacing: .10em;
    text-transform: uppercase;
    font-weight: 750;
}

.sidebar-engine {
    padding: 15px;
    border-radius: 16px;
    border: 1px solid rgba(56, 189, 248, .18);
    background:
        linear-gradient(145deg,
            rgba(15, 38, 65, .72),
            rgba(8, 22, 38, .72));
    box-shadow: 0 14px 35px rgba(0,0,0,.16);
    margin-bottom: 22px;
}

.sidebar-engine-label {
    color: #64748b;
    font-size: .66rem;
    letter-spacing: .12em;
    font-weight: 800;
}

.sidebar-engine-title {
    color: #e0f2fe;
    font-weight: 760;
    margin-top: 6px;
}

.sidebar-engine-text {
    color: #64748b;
    font-size: .74rem;
    margin-top: 3px;
}

.sidebar-stat {
    border: 1px solid rgba(148,163,184,.10);
    background: rgba(15, 23, 42, .50);
    border-radius: 14px;
    padding: 12px;
}

/* =========================================================
   HERO
   ========================================================= */

.estate-hero {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(56, 189, 248, .20);
    border-radius: 26px;
    padding: 42px 44px;
    margin: 2px 0 26px;
    background:
        linear-gradient(135deg,
            rgba(13, 31, 53, .98),
            rgba(9, 22, 38, .96));
    box-shadow:
        0 25px 80px rgba(0,0,0,.25),
        inset 0 1px 0 rgba(255,255,255,.025);
}

.estate-hero:before {
    content: "";
    position: absolute;
    width: 420px;
    height: 420px;
    right: -160px;
    top: -220px;
    border-radius: 50%;
    background: rgba(14,165,233,.10);
    filter: blur(4px);
}

.hero-kicker {
    color: #38bdf8;
    text-transform: uppercase;
    font-size: .70rem;
    font-weight: 850;
    letter-spacing: .16em;
}

.hero-title {
    color: #f8fafc;
    font-size: 2.75rem;
    line-height: 1.06;
    font-weight: 850;
    margin-top: 9px;
    max-width: 800px;
}

.hero-subtitle {
    color: #94a3b8;
    max-width: 790px;
    margin-top: 14px;
    font-size: .98rem;
    line-height: 1.7;
}

/* =========================================================
   SEARCH INPUT / BUTTONS
   ========================================================= */

div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div,
div[data-baseweb="select"] > div {
    background: #0d192a !important;
    border-color: rgba(148,163,184,.18) !important;
    border-radius: 13px !important;
}

div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    color: #f8fafc !important;
}

div[data-baseweb="input"]:focus-within > div,
div[data-baseweb="textarea"]:focus-within > div,
div[data-baseweb="select"]:focus-within > div {
    border-color: rgba(56,189,248,.55) !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,.08) !important;
}

.stButton > button {
    border-radius: 11px !important;
    border: 1px solid rgba(96,165,250,.20) !important;
    background: linear-gradient(180deg, #14243a, #0f1d30) !important;
    color: #e2e8f0 !important;
    font-weight: 700 !important;
    min-height: 42px !important;
    transition: all .18s ease !important;
}

.stButton > button:hover {
    border-color: rgba(56,189,248,.60) !important;
    background: linear-gradient(180deg, #183452, #11243a) !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 22px rgba(0,0,0,.18);
}

/* =========================================================
   SECTION / STATUS
   ========================================================= */

.section-header {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 16px;
    margin: 25px 0 14px;
}

.section-kicker {
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: .12em;
    font-size: .67rem;
    font-weight: 850;
}

.section-title {
    color: #f8fafc;
    font-size: 1.55rem;
    font-weight: 800;
    margin-top: 4px;
}

.results-count {
    color: #64748b;
    font-size: .78rem;
}

.ai-response-box {
    border-radius: 20px;
    padding: 24px 26px;
    margin: 15px 0 24px;
    border: 1px solid rgba(56,189,248,.22);
    background:
        linear-gradient(135deg,
            rgba(15, 34, 58, .98),
            rgba(10, 23, 40, .98));
    box-shadow: 0 18px 55px rgba(0,0,0,.20);
}

.ai-label {
    color: #60a5fa;
    font-size: .68rem;
    text-transform: uppercase;
    letter-spacing: .13em;
    font-weight: 850;
}

.ai-answer {
    color: #e2e8f0;
    font-size: 1rem;
    line-height: 1.72;
    margin-top: 10px;
}

.ai-meta {
    color: #64748b;
    font-size: .74rem;
    margin-top: 14px;
}

/* =========================================================
   NATIVE PROPERTY CARD
   We use Streamlit columns/components instead of raw HTML
   for the actual card. This prevents HTML being displayed
   as code in the browser.
   ========================================================= */

.property-shell {
    border: 1px solid rgba(148,163,184,.15);
    border-radius: 20px;
    background:
        linear-gradient(145deg,
            rgba(17, 30, 49, .98),
            rgba(10, 21, 36, .98));
    padding: 8px;
    margin: 0 0 16px;
    box-shadow: 0 16px 45px rgba(0,0,0,.18);
}

.property-image-wrap {
    border-radius: 15px;
    overflow: hidden;
    background: #0b1524;
    border: 1px solid rgba(148,163,184,.10);
}

.property-id-pill {
    color: #7dd3fc;
    background: rgba(14,165,233,.09);
    border: 1px solid rgba(56,189,248,.20);
    border-radius: 999px;
    padding: 4px 9px;
    font-size: .66rem;
    font-weight: 800;
    letter-spacing: .06em;
}

.property-match-pill {
    color: #6ee7b7;
    background: rgba(16,185,129,.09);
    border: 1px solid rgba(52,211,153,.22);
    border-radius: 999px;
    padding: 4px 9px;
    font-size: .66rem;
    font-weight: 800;
}

.property-title-native {
    color: #f8fafc;
    font-size: 1.25rem;
    line-height: 1.25;
    font-weight: 800;
    margin: 10px 0 5px;
}

.property-location-native {
    color: #94a3b8;
    font-size: .82rem;
    margin-bottom: 14px;
}

.property-price-native {
    color: #f8fafc;
    font-size: 1.38rem;
    font-weight: 850;
}

.property-stat-label-native {
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-size: .62rem;
    font-weight: 800;
}

.property-stat-value-native {
    color: #dbeafe;
    font-size: .87rem;
    font-weight: 720;
    margin-top: 3px;
}

.feature-badge {
    display: inline-block;
    background: rgba(51,65,85,.55);
    border: 1px solid rgba(148,163,184,.12);
    color: #cbd5e1;
    padding: 5px 8px;
    border-radius: 8px;
    font-size: .69rem;
    margin: 3px 4px 3px 0;
}

.property-description-native {
    color: #94a3b8;
    font-size: .80rem;
    line-height: 1.62;
    margin-top: 10px;
}

/* =========================================================
   METRICS / DASHBOARD
   ========================================================= */

.metric-card {
    min-height: 126px;
    padding: 18px;
    border-radius: 18px;
    border: 1px solid rgba(148,163,184,.13);
    background: linear-gradient(145deg, #101d30, #0c1727);
    box-shadow: 0 14px 35px rgba(0,0,0,.17);
    transition: transform .18s ease, border-color .18s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(56,189,248,.35);
}

.metric-icon {
    font-size: 1.25rem;
}

.metric-label {
    color: #64748b;
    font-size: .66rem;
    text-transform: uppercase;
    letter-spacing: .10em;
    font-weight: 800;
    margin-top: 7px;
}

.metric-value {
    color: #f8fafc;
    font-size: 1.72rem;
    font-weight: 850;
    margin-top: 4px;
}

/* Plotly / dataframe containers */
[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
}

[data-testid="stExpander"] {
    border: 1px solid rgba(148,163,184,.13) !important;
    border-radius: 13px !important;
    background: rgba(10, 20, 34, .45);
}

/* Details */
.detail-panel {
    border: 1px solid rgba(148,163,184,.13);
    border-radius: 18px;
    padding: 20px;
    background: linear-gradient(145deg, #101d30, #0c1727);
}

.detail-title {
    color: #f8fafc;
    font-size: 1.32rem;
    font-weight: 820;
}

.detail-location {
    color: #94a3b8;
    margin-top: 5px;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 14px !important;
}

/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 900px) {
    .block-container {
        padding: 1.3rem 1rem 3rem;
    }

    .estate-hero {
        padding: 30px 24px;
    }

    .hero-title {
        font-size: 2rem;
    }
}

</style>
"""


def apply_custom_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, icon: str = "🏢"):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe_text(value, fallback="Not specified"):
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def render_property_card(match: PropertyMatch, key: str = ""):
    """
    Professional native Streamlit property card.

    IMPORTANT:
    This deliberately avoids rendering the complete property card as one
    giant HTML string. Streamlit renders the content through native widgets,
    so HTML source code cannot appear on screen as it did in the previous UI.
    """
    p = match.property_item
    score_pct = max(0, min(100, int(match.combined_score * 100)))

    img_src = get_property_image(p)
    if not img_src:
        img_src = DEFAULT_PLACEHOLDER_IMAGE

    is_land = (
        str(p.property_type).strip().lower() == "land"
        or (p.bedrooms == 0 and p.bathrooms == 0)
    )

    config_text = (
        "Residential Plot"
        if is_land
        else f"{p.bedrooms} BHK · {p.bathrooms} Bath"
    )

    with st.container(border=True):
        image_col, info_col = st.columns([1.05, 2.55], gap="large")

        with image_col:
            st.image(
                img_src,
                use_container_width=True,
                output_format="auto",
            )

        with info_col:
            top_left, top_right = st.columns([3, 1])

            with top_left:
                st.markdown(
                    f'<span class="property-id-pill">{_safe_text(p.property_id)}</span>',
                    unsafe_allow_html=True,
                )

            with top_right:
                st.markdown(
                    f'<div style="text-align:right;"><span class="property-match-pill">AI Match {score_pct}%</span></div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="property-title-native">{_safe_text(p.title)}</div>',
                unsafe_allow_html=True,
            )

            location = _safe_text(p.locality or p.location)
            city = _safe_text(p.city)

            st.markdown(
                f'<div class="property-location-native">{location}, {city}</div>',
                unsafe_allow_html=True,
            )

            s1, s2, s3, s4 = st.columns(4)

            with s1:
                st.markdown(
                    '<div class="property-stat-label-native">Price</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="property-price-native">{_safe_text(p.formatted_price)}</div>',
                    unsafe_allow_html=True,
                )

            with s2:
                st.markdown(
                    '<div class="property-stat-label-native">Area</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="property-stat-value-native">{p.area_sqft:,} sq.ft</div>',
                    unsafe_allow_html=True,
                )

            with s3:
                st.markdown(
                    '<div class="property-stat-label-native">Configuration</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="property-stat-value-native">{config_text}</div>',
                    unsafe_allow_html=True,
                )

            with s4:
                st.markdown(
                    '<div class="property-stat-label-native">Type</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="property-stat-value-native">{_safe_text(p.property_type)}</div>',
                    unsafe_allow_html=True,
                )

            if p.amenities:
                badges = "".join(
                    f'<span class="feature-badge">{_safe_text(a)}</span>'
                    for a in p.amenities[:5]
                )
                st.markdown(
                    f'<div style="margin-top:10px;">{badges}</div>',
                    unsafe_allow_html=True,
                )

            if p.description:
                description = _safe_text(p.description)
                if len(description) > 300:
                    description = description[:300].rstrip() + "..."

                st.markdown(
                    f'<div class="property-description-native">{description}</div>',
                    unsafe_allow_html=True,
                )


def render_property_details(match: PropertyMatch):
    p = match.property_item

    st.markdown(
        f"""
        <div class="detail-panel">
            <div class="detail-title">{_safe_text(p.title)}</div>
            <div class="detail-location">
                {_safe_text(p.locality or p.location)}, {_safe_text(p.city)},
                {_safe_text(p.state)} · {_safe_text(p.pincode)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_property_images(p)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Listed Price", _safe_text(p.formatted_price))
    c2.metric("Carpet Area", f"{p.area_sqft:,} sq.ft")
    c3.metric("Rate / Sq.Ft", f"₹{p.price_per_sqft}/sq.ft")
    c4.metric("Property Type", _safe_text(p.property_type))

    st.markdown("### Why this property matches")

    if match.match_reasons:
        for reason in match.match_reasons:
            st.markdown(f"**[Match]** {reason}")
    else:
        st.markdown(
            "**[Match]** Matched using semantic similarity and structured property criteria."
        )

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Property Specifications")
        st.write(
            f"**Bedrooms:** "
            f"{p.bedrooms if p.bedrooms > 0 else 'N/A (Land Plot)'}"
        )
        st.write(
            f"**Bathrooms:** "
            f"{p.bathrooms if p.bathrooms > 0 else 'N/A'}"
        )
        st.write(f"**Furnishing:** {_safe_text(p.furnishing)}")
        st.write(f"**Age:** {_safe_text(p.age_years, 'N/A')} years")
        st.write(f"**Landmark:** {_safe_text(p.landmark)}")

    with col_b:
        st.markdown("#### Location")
        st.write(f"**Locality:** {_safe_text(p.locality or p.location)}")
        st.write(f"**Area:** {_safe_text(p.area)}")
        st.write(f"**District:** {_safe_text(p.district or p.city)}")
        st.write(f"**City:** {_safe_text(p.city)}")
        st.write(f"**State:** {_safe_text(p.state)}")
        st.write(f"**Pincode:** {_safe_text(p.pincode)}")

    st.markdown("#### Amenities & Facilities")

    if p.amenities:
        st.markdown(
            " ".join(f"`{a}`" for a in p.amenities)
        )
    else:
        st.info("No specific amenities listed.")

    st.markdown("#### Property Description")
    st.info(_safe_text(p.description))


def render_property_images(property_item: PropertyItem):
    """Responsive property gallery with safe image fallback."""

    if not property_item or not property_item.image_urls:
        st.image(
            DEFAULT_PLACEHOLDER_IMAGE,
            caption="Property image unavailable",
            use_container_width=True,
        )
        return

    urls = [
        u.strip()
        for u in property_item.image_urls
        if isinstance(u, str) and u.strip()
    ]

    if not urls:
        st.image(
            DEFAULT_PLACEHOLDER_IMAGE,
            caption="Property image unavailable",
            use_container_width=True,
        )
        return

    urls = urls[:4]
    cols = st.columns(len(urls))

    for idx, url in enumerate(urls):
        with cols[idx]:
            try:
                st.image(
                    url,
                    caption=f"{property_item.title} · Image {idx + 1}",
                    use_container_width=True,
                )
            except Exception:
                st.image(
                    DEFAULT_PLACEHOLDER_IMAGE,
                    caption="Image unavailable",
                    use_container_width=True,
                )


def render_comparison_table(selected_properties: List[PropertyItem]):
    if not selected_properties:
        st.info("Select 2 or 3 properties to compare.")
        return

    data = {
        "Attribute": [
            "Property ID",
            "Title",
            "Location",
            "City",
            "Price",
            "Carpet Area",
            "Price / Sq.Ft",
            "Bedrooms",
            "Bathrooms",
            "Property Type",
            "Furnishing",
            "Age",
            "Amenities",
        ]
    }

    for idx, p in enumerate(selected_properties, 1):
        col_name = f"Property {chr(64 + idx)} · {p.property_id}"

        data[col_name] = [
            p.property_id,
            p.title,
            p.location,
            p.city,
            p.formatted_price,
            f"{p.area_sqft:,} sq.ft",
            f"₹{p.price_per_sqft}/sq.ft",
            f"{p.bedrooms} BHK",
            f"{p.bathrooms} Bathrooms",
            p.property_type,
            p.furnishing,
            f"{p.age_years} Years",
            ", ".join(p.amenities[:6]) if p.amenities else "None",
        ]

    st.dataframe(
        pd.DataFrame(data),
        use_container_width=True,
        hide_index=True,
    )