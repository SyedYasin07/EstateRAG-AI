"""
EstateRAG AI — Intelligent Real-Estate Property Discovery Platform
Professional Streamlit application entrypoint.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

from app.data.dataset_loader import DatasetLoader
from app.data.preprocessor import PropertyPreprocessor
from app.data.schema import PropertyItem
from app.retrieval.vector_store import VectorStoreManager
from app.retrieval.hybrid_retriever import HybridRetriever
from app.services.rag_chain import RAGChain
from app.utils.image_validator import ImageValidator
from app.ui.components import (
    apply_custom_css,
    render_metric_card,
    render_property_card,
    render_property_details,
    render_comparison_table,
)


st.set_page_config(
    page_title="EstateRAG AI | Property Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()


@st.cache_resource(show_spinner=False)
def init_rag_system():
    loader = DatasetLoader(data_path="data/properties.csv")
    properties = loader.load_data()

    docs = PropertyPreprocessor.process_all(properties)

    vec_store = VectorStoreManager(vector_store_dir="vectorstore")
    if not vec_store.load_index():
        vec_store.build_index(docs)

    retriever = HybridRetriever(
        vector_store=vec_store,
        properties=properties,
    )

    return loader, vec_store, retriever


def get_next_property_id(properties):
    numeric_ids = []

    for p in properties:
        if p.property_id and "PROP-" in p.property_id:
            try:
                numeric_ids.append(int(p.property_id.split("-")[1]))
            except (IndexError, ValueError):
                pass

    return f"PROP-{max(numeric_ids) + 1 if numeric_ids else 1001}"


def reset_search_filters():
    st.session_state["filter_city"] = "All Cities"
    st.session_state["filter_type"] = "All Types"
    st.session_state["filter_budget"] = 500
    st.session_state["filter_area"] = 0


def sidebar_brand(summary, loader):
    st.sidebar.markdown(
        """
        <div style="padding:8px 4px 22px;">
            <div style="
                font-size:1.45rem;
                font-weight:850;
                color:#f8fafc;
                letter-spacing:-.04em;
            ">
                🏠 Estate<span style="color:#60a5fa;">RAG</span> AI
            </div>
            <div style="
                color:#64748b;
                font-size:.72rem;
                margin-top:5px;
                letter-spacing:.08em;
                text-transform:uppercase;
                font-weight:700;
            ">
                Property Intelligence Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div style="
            padding:13px;
            border-radius:14px;
            border:1px solid rgba(96,165,250,.18);
            background:rgba(30,58,95,.18);
            margin-bottom:18px;
        ">
            <div style="font-size:.68rem;color:#64748b;text-transform:uppercase;
                        letter-spacing:.1em;font-weight:800;">
                AI ENGINE
            </div>
            <div style="margin-top:5px;color:#dbeafe;font-weight:700;">
                Hybrid RAG · FAISS
            </div>
            <div style="color:#64748b;font-size:.75rem;margin-top:3px;">
                Grounded property intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Explore")

    nav_option = st.sidebar.radio(
        "Navigation",
        [
            "🔍 AI Property Search",
            "📊 Property Dashboard",
            "⚖️ Property Comparison",
            "💡 AI Recommendations",
            "⚙️ Admin Management",
            "ℹ️ Architecture & About",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Inventory Snapshot")

    localities = (
        len(loader.clean_df["locality"].unique())
        if "locality" in loader.clean_df.columns
        else 0
    )

    s1, s2 = st.sidebar.columns(2)
    s1.metric("Listings", summary.total_properties)
    s2.metric("Cities", len(summary.cities))

    st.sidebar.caption(
        f"🗺️ {localities} localities  •  "
        f"⚡ FAISS 384-d"
    )

    return nav_option


def render_search_hero():
    st.markdown(
        """
        <section class="estate-hero">
            <div class="hero-kicker">AI-POWERED PROPERTY INTELLIGENCE</div>
            <div class="hero-title">
                <h1>Discover Properties That Match Your Lifestyle.</h1>
            </div>
            <div class="hero-subtitle">
                <p>
    Search by location, budget, BHK, area, property type, and preferences.
    EstateRAG AI combines intelligent filtering with semantic retrieval
    to deliver relevant, dataset-grounded property results.
</p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_search_page(loader, retriever, summary):
    render_search_hero()

    if "filter_city" not in st.session_state:
        reset_search_filters()

    st.markdown("#### Popular searches")

    quick_cols = st.columns(5)

    sample_query = None

    quick_actions = [
        ("📍", "Properties in Tirupati", "Show properties in Tirupati"),
        ("🌳", "Land under ₹50L", "Show land properties in Tirupati under ₹50 lakhs"),
        ("🏘️", "Near Alipiri", "Properties near Alipiri"),
        ("🏠", "3 BHK Vijayawada", "3 BHK in Vijayawada"),
        ("🏡", "Villas Hyderabad", "Villas in Hyderabad"),
    ]

    for col, (icon, label, query) in zip(quick_cols, quick_actions):
        if col.button(f"{icon} {label}", use_container_width=True):
            sample_query = query
            reset_search_filters()

    st.markdown("")

    user_query = st.text_input(
        "Property search",
        value=sample_query if sample_query else "",
        placeholder=(
            "Try: 2 BHK in Tirupati under ₹70 lakhs with parking..."
        ),
        label_visibility="collapsed",
    )

    with st.expander("⚙️ Advanced filters", expanded=False):
        f1, f2, f3, f4 = st.columns(4)

        city_filter = f1.selectbox(
            "City",
            ["All Cities"] + summary.cities,
            key="filter_city",
        )

        type_filter = f2.selectbox(
            "Property Type",
            [
                "All Types",
                "Apartment",
                "Villa",
                "Independent House",
                "Land",
                "Penthouse",
            ],
            key="filter_type",
        )

        max_budget_filter = f3.slider(
            "Maximum Budget (₹ Lakhs)",
            min_value=10,
            max_value=500,
            value=500,
            step=10,
            key="filter_budget",
        )

        min_area_filter = f4.slider(
            "Minimum Area (sq.ft)",
            min_value=0,
            max_value=4000,
            value=0,
            step=100,
            key="filter_area",
        )

    should_search = (
        bool(user_query)
        or city_filter != "All Cities"
        or type_filter != "All Types"
        or max_budget_filter < 500
        or min_area_filter > 0
    )

    if not should_search:
        st.markdown(
            """
            <div style="
                margin-top:26px;
                padding:42px 25px;
                text-align:center;
                border:1px dashed rgba(148,163,184,.20);
                border-radius:20px;
                background:rgba(15,23,42,.42);
            ">
                <div style="font-size:2rem;">⌕</div>
                <div style="font-size:1.15rem;font-weight:750;color:#e2e8f0;">
                    Start your property search
                </div>
                <div style="color:#64748b;margin-top:6px;">
                    Describe what you need in plain English.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    active_query = user_query if user_query else "Show properties"

    with st.spinner("Searching property intelligence..."):
        parsed_query, matches = retriever.search(active_query, top_k=10)

        if not parsed_query.city and city_filter != "All Cities":
            matches = [
                m for m in matches
                if m.property_item.city.lower() == city_filter.lower()
            ]

        if not parsed_query.property_type and type_filter != "All Types":
            matches = [
                m for m in matches
                if m.property_item.property_type.lower()
                == type_filter.lower()
            ]

        if not parsed_query.max_price_lakhs and max_budget_filter < 500:
            matches = [
                m for m in matches
                if m.property_item.price_lakhs <= max_budget_filter
            ]

        if not parsed_query.min_area_sqft and min_area_filter > 0:
            matches = [
                m for m in matches
                if m.property_item.area_sqft >= min_area_filter
            ]

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        rag_response = RAGChain.generate_response(
            active_query,
            parsed_query,
            matches,
            st.session_state.chat_history,
        )

        st.session_state.chat_history.append(
            {
                "user": active_query,
                "assistant": rag_response["answer"],
            }
        )

    # Result summary strip
    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric("Matching Listings", len(matches))

    with r2:
        st.metric(
            "Location",
            parsed_query.city if parsed_query.city else "Any city",
        )

    with r3:
        st.metric(
            "Search Mode",
            "Hybrid AI",
        )

    st.markdown("### 🤖 AI property brief")

    st.markdown(
        f"""
        <div class="ai-response-box">
            <div class="ai-label">Grounded response</div>
            <div class="ai-answer">{rag_response["answer"]}</div>
            <div style="
                margin-top:14px;
                color:#64748b;
                font-size:.75rem;
            ">
                {"Gemini / Groq LLM" if rag_response["used_llm"]
                 else "Offline Grounded Synthesizer"}
                &nbsp; · &nbsp; Based only on retrieved listings
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔍 Search intelligence"):
        st.json(
            {
                "User Query": active_query,
                "Parsed Criteria": {
                    "City": parsed_query.city or "Any",
                    "Locality": parsed_query.location or "Any",
                    "Property Type": parsed_query.property_type or "Any",
                    "Bedrooms": parsed_query.bedrooms or "Any",
                    "Max Budget": (
                        f"₹{parsed_query.max_price_lakhs}L"
                        if parsed_query.max_price_lakhs
                        else "Any"
                    ),
                    "Min Area": (
                        f"{parsed_query.min_area_sqft} sqft"
                        if parsed_query.min_area_sqft
                        else "Any"
                    ),
                    "Amenities": parsed_query.amenities,
                },
                "UI Filters": {
                    "City": city_filter,
                    "Type": type_filter,
                    "Max Budget": (
                        max_budget_filter
                        if max_budget_filter < 500
                        else "No Limit"
                    ),
                    "Min Area": (
                        min_area_filter
                        if min_area_filter > 0
                        else "No Minimum"
                    ),
                },
            }
        )

    st.markdown(
        f"### 🏠 Property results "
        f"<span style='color:#64748b;font-size:.85rem;'>"
        f"{len(matches)} matching listings</span>",
        unsafe_allow_html=True,
    )

    if not matches:
        st.warning(
            "No matching properties were found for these criteria."
        )
        return

    for index, match in enumerate(matches):
        render_property_card(match, key=f"search_{index}")

        with st.expander(
            f"View details · {match.property_item.property_id}"
        ):
            render_property_details(match)


def render_dashboard(loader, summary):
    st.markdown("# Market Intelligence")
    st.caption(
        "A data-driven view of your real-estate inventory."
    )

    df = loader.clean_df
    total_localities = (
        len(df["locality"].unique())
        if "locality" in df.columns
        else len(df)
    )

    st.markdown("### Portfolio snapshot")

    cols = st.columns(5)

    metrics = [
        ("Total Properties", str(summary.total_properties), "🏢"),
        ("Cities Covered", str(len(summary.cities)), "📍"),
        ("Localities", str(total_localities), "🗺️"),
        ("Average Price", f"₹{summary.avg_price_lakhs:.1f} L", "💰"),
        ("Largest Area", f"{summary.max_area_sqft:,} sqft", "📐"),
    ]

    for col, (label, value, icon) in zip(cols, metrics):
        with col:
            render_metric_card(label, value, icon)

    st.markdown("### Market overview")

    c1, c2 = st.columns(2)

    with c1:
        city_counts = (
            df["city"]
            .value_counts()
            .reset_index()
        )
        city_counts.columns = ["City", "Listings"]

        fig = px.bar(
            city_counts,
            x="City",
            y="Listings",
            template="plotly_dark",
            text="Listings",
        )
        fig.update_layout(
            title="Inventory by city",
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.pie(
            df,
            names="property_type",
            hole=.55,
            template="plotly_dark",
        )
        fig.update_layout(
            title="Property type mix",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig = px.histogram(
            df,
            x="price_lakhs",
            nbins=20,
            color="property_type",
            template="plotly_dark",
            labels={"price_lakhs": "Price (₹ Lakhs)"},
        )
        fig.update_layout(
            title="Price distribution",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.scatter(
            df,
            x="area_sqft",
            y="price_lakhs",
            color="city",
            size="bedrooms",
            hover_data=[
                "property_id",
                "title",
                "locality",
                "city",
            ],
            template="plotly_dark",
            labels={
                "area_sqft": "Area (sq.ft)",
                "price_lakhs": "Price (₹ Lakhs)",
            },
        )
        fig.update_layout(
            title="Price vs area",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### City inventory")

    city_summary = (
        df.groupby("city")
        .agg(
            Listings=("property_id", "count"),
            Avg_Price_Lakhs=("price_lakhs", "mean"),
            Avg_Area_Sqft=("area_sqft", "mean"),
        )
        .reset_index()
    )

    city_summary["Avg_Price_Lakhs"] = (
        city_summary["Avg_Price_Lakhs"].round(1)
    )
    city_summary["Avg_Area_Sqft"] = (
        city_summary["Avg_Area_Sqft"].round(0).astype(int)
    )

    st.dataframe(
        city_summary,
        use_container_width=True,
        hide_index=True,
    )


def render_comparison(properties):
    st.markdown("# Compare Properties")
    st.caption(
        "Evaluate up to three listings side-by-side before making a decision."
    )

    prop_options = {
        f"{p.property_id} · {p.title} · {p.formatted_price}": p
        for p in properties
    }

    selected_keys = st.multiselect(
        "Choose properties",
        list(prop_options.keys()),
        max_selections=3,
        placeholder="Select 2 or 3 properties...",
    )

    selected_props = [
        prop_options[key]
        for key in selected_keys
    ]

    if selected_props:
        st.markdown("### Comparison matrix")
        render_comparison_table(selected_props)

    if len(selected_props) >= 2:
        st.markdown("### 🤖 AI comparison")

        with st.spinner("Analyzing property trade-offs..."):
            analysis = RAGChain.generate_comparison(
                selected_props
            )

        st.markdown(
            f"""
            <div class="ai-response-box">
                <div class="ai-label">AI comparative analysis</div>
                <div class="ai-answer">{analysis}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif selected_props:
        st.info("Select one more property to enable AI comparison.")


def render_recommendations(retriever):
    st.markdown("# AI Recommendations")
    st.caption(
        "Tell EstateRAG what kind of property fits your lifestyle."
    )

    st.markdown(
        """
        <div class="ai-response-box">
            <div class="ai-label">Recommendation engine</div>
            <div class="ai-answer">
                Describe your family, lifestyle, location, budget or
                property preferences. The system will find the closest
                matches from the available inventory.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pref_text = st.text_area(
        "Your requirements",
        placeholder=(
            "Example: I need a spacious family home in Tirupati, "
            "above 1500 sq.ft, with parking and good security."
        ),
        height=130,
        label_visibility="collapsed",
    )

    if st.button(
        "✨ Generate Recommendations",
        type="primary",
        use_container_width=True,
    ):
        if not pref_text.strip():
            st.warning("Describe your property requirements first.")
            return

        with st.spinner("Finding your best property matches..."):
            parsed_query, matches = retriever.search(
                pref_text,
                top_k=5,
            )
            response = RAGChain.generate_response(
                pref_text,
                parsed_query,
                matches,
            )

        st.markdown("### Your personalized property shortlist")

        st.markdown(
            f"""
            <div class="ai-response-box">
                <div class="ai-label">Recommendation reasoning</div>
                <div class="ai-answer">{response["answer"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for index, match in enumerate(matches):
            render_property_card(
                match,
                key=f"recommendation_{index}",
            )

            with st.expander(
                f"View recommendation details · "
                f"{match.property_item.property_id}"
            ):
                render_property_details(match)


def render_admin(loader, vec_store, properties, summary):
    st.markdown("# Inventory Administration")
    st.caption(
        "Manage listings, audit image quality and keep the retrieval index synchronized."
    )

    img_report = ImageValidator.dataset_image_report(properties)

    total_localities = (
        len(loader.clean_df["locality"].unique())
        if "locality" in loader.clean_df.columns
        else 0
    )

    st.markdown("### System health")

    cols = st.columns(5)
    health = [
        ("Listings", summary.total_properties),
        ("Cities", len(summary.cities)),
        ("Localities", total_localities),
        ("With Images", img_report["properties_with_images"]),
        ("Unique Images", img_report["unique_image_urls"]),
    ]

    for col, (label, value) in zip(cols, health):
        with col:
            st.metric(label, value)

    if img_report["duplicate_image_assignments"] > 0:
        st.warning(
            f"Image audit found "
            f"{img_report['duplicate_image_assignments']} duplicate image assignments."
        )
    else:
        st.success("Image audit passed — no duplicate image assignments detected.")

    st.markdown("### Inventory")

    inventory_cols = [
        "property_id",
        "title",
        "city",
        "locality",
        "property_type",
        "price_lakhs",
        "area_sqft",
        "bedrooms",
    ]

    st.dataframe(
        loader.clean_df[
            [c for c in inventory_cols if c in loader.clean_df.columns]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Add new property")

    with st.expander("➕ Create listing", expanded=False):
        default_auto_id = get_next_property_id(properties)

        with st.form("add_property_form"):
            c1, c2, c3 = st.columns(3)

            new_id = c1.text_input(
                "Property ID",
                value=default_auto_id,
            )
            new_title = c2.text_input(
                "Property Title",
                value="Spacious 3 BHK Apartment",
            )
            new_city = c3.text_input(
                "City",
                value="Tirupati",
            )

            c4, c5, c6 = st.columns(3)

            new_price = c4.number_input(
                "Price (₹ Lakhs)",
                value=55.0,
                step=1.0,
            )
            new_area = c5.number_input(
                "Area (sq.ft)",
                value=1450,
                step=50,
            )
            new_bhk = c6.number_input(
                "Bedrooms / BHK",
                value=3,
                step=1,
            )

            c7, c8, c9 = st.columns(3)

            new_baths = c7.number_input(
                "Bathrooms",
                value=3,
                step=1,
            )
            new_type = c8.selectbox(
                "Property Type",
                [
                    "Apartment",
                    "Villa",
                    "Independent House",
                    "Land",
                    "Penthouse",
                ],
            )
            new_furnish = c9.selectbox(
                "Furnishing",
                [
                    "Semi-Furnished",
                    "Furnished",
                    "Unfurnished",
                ],
            )

            l1, l2, l3 = st.columns(3)

            new_state = l1.text_input(
                "State",
                value="Andhra Pradesh",
            )
            new_district = l2.text_input(
                "District",
                value="Tirupati",
            )
            new_locality = l3.text_input(
                "Locality",
                value="MR Palli",
            )

            l4, l5, l6 = st.columns(3)

            new_area_name = l4.text_input(
                "Area Name",
                value="MR Palli Circle",
            )
            new_landmark = l5.text_input(
                "Landmark",
                value="SV University Gate",
            )
            new_pincode = l6.text_input(
                "Pincode",
                value="517502",
            )

            new_loc = st.text_input(
                "Location Summary",
                value="MR Palli, Tirupati",
            )

            new_img = st.text_input(
                "Image URLs (comma separated)",
                value="",
                placeholder="https://... , https://...",
            )

            new_amenities = st.text_input(
                "Amenities",
                value="Parking, Security, Power Backup, Gym",
            )

            new_desc = st.text_area(
                "Description",
                value=(
                    "Modern property with excellent connectivity "
                    "and convenient access to nearby facilities."
                ),
            )

            submitted = st.form_submit_button(
                "Add Property & Rebuild Index",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            clean_id = new_id.strip()

            if not clean_id:
                st.error("Property ID cannot be empty.")

            elif any(p.property_id == clean_id for p in properties):
                st.error(
                    f"Property ID '{clean_id}' already exists."
                )

            elif not new_title.strip() or not new_city.strip():
                st.error("Title and City are required.")

            else:
                new_item = PropertyItem(
                    property_id=clean_id,
                    title=new_title.strip(),
                    location=new_loc.strip(),
                    city=new_city.strip(),
                    price_lakhs=float(new_price),
                    area_sqft=int(new_area),
                    bedrooms=int(new_bhk),
                    bathrooms=int(new_baths),
                    property_type=new_type,
                    amenities=new_amenities,
                    furnishing=new_furnish,
                    age_years=0,
                    description=new_desc.strip(),
                    state=new_state.strip(),
                    district=new_district.strip(),
                    locality=new_locality.strip(),
                    area=new_area_name.strip(),
                    landmark=new_landmark.strip(),
                    pincode=new_pincode.strip(),
                    image_urls=new_img.strip(),
                )

                properties.append(new_item)

                records = [
                    p.model_dump()
                    if hasattr(p, "model_dump")
                    else p.dict()
                    for p in properties
                ]

                df_new = pd.DataFrame(records)

                for col in ["amenities", "image_urls"]:
                    if col in df_new.columns:
                        df_new[col] = df_new[col].apply(
                            lambda value: (
                                ", ".join(value)
                                if isinstance(value, list)
                                else str(value)
                            )
                        )

                df_new.to_csv(
                    "data/properties.csv",
                    index=False,
                )

                st.cache_resource.clear()

                docs = PropertyPreprocessor.process_all(properties)
                vec_store.build_index(docs)

                st.success(
                    f"{clean_id} added successfully. "
                    "CSV and FAISS index updated."
                )

                st.rerun()

    st.markdown("### Data maintenance")

    if st.button(
        "🔄 Refresh Dataset & FAISS Index",
        use_container_width=True,
    ):
        st.cache_resource.clear()

        docs = PropertyPreprocessor.process_all(properties)
        vec_store.build_index(docs)

        st.success(
            "Dataset and FAISS vector index refreshed successfully."
        )
        st.rerun()


def render_architecture():
    st.markdown("# Platform Architecture")
    st.caption(
        "How EstateRAG converts natural-language requirements into grounded property intelligence."
    )

    architecture = [
        ("01", "Property Dataset", "CSV property inventory"),
        ("02", "Data Processing", "Validation, normalization and document creation"),
        ("03", "Embeddings", "SentenceTransformer semantic representation"),
        ("04", "FAISS Vector Store", "Fast semantic similarity retrieval"),
        ("05", "Hybrid Retrieval", "Metadata filtering + vector similarity"),
        ("06", "Top-K Context", "Relevant property records"),
        ("07", "RAG Generator", "Gemini / Groq / offline grounded engine"),
        ("08", "Streamlit Experience", "Search, analytics, comparison and recommendations"),
    ]

    for number, title, description in architecture:
        st.markdown(
            f"""
            <div style="
                display:flex;
                gap:18px;
                align-items:center;
                padding:17px 20px;
                margin:8px 0;
                border:1px solid rgba(148,163,184,.13);
                border-radius:14px;
                background:linear-gradient(90deg,#101a2b,#0d1524);
            ">
                <div style="
                    min-width:40px;
                    height:40px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    border-radius:10px;
                    background:rgba(59,130,246,.12);
                    color:#60a5fa;
                    font-weight:850;
                ">{number}</div>
                <div>
                    <div style="color:#f8fafc;font-weight:800;">
                        {title}
                    </div>
                    <div style="color:#64748b;font-size:.82rem;margin-top:3px;">
                        {description}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Grounding principles")

    c1, c2, c3 = st.columns(3)

    cards = [
        (
            c1,
            "🎯 Strict grounding",
            "The response generator is instructed to use retrieved property records rather than inventing listing details.",
        ),
        (
            c2,
            "🔎 Hybrid retrieval",
            "Structured constraints handle exact requirements while vector search handles semantic preferences.",
        ),
        (
            c3,
            "🛡️ Zero-match handling",
            "When the dataset has no suitable record, the system reports that instead of fabricating a property.",
        ),
    ]

    for col, title, body in cards:
        with col:
            st.markdown(
                f"""
                <div class="metric-card" style="min-height:180px;">
                    <div style="font-size:1.35rem;">{title[:2]}</div>
                    <div style="
                        color:#f8fafc;
                        font-size:1rem;
                        font-weight:800;
                        margin-top:10px;
                    ">{title[3:]}</div>
                    <div style="
                        color:#64748b;
                        font-size:.82rem;
                        line-height:1.6;
                        margin-top:8px;
                    ">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main():
    loader, vec_store, retriever = init_rag_system()
    properties = loader.properties
    summary = loader.get_summary()

    nav_option = sidebar_brand(summary, loader)

    if nav_option == "🔍 AI Property Search":
        render_search_page(loader, retriever, summary)

    elif nav_option == "📊 Property Dashboard":
        render_dashboard(loader, summary)

    elif nav_option == "⚖️ Property Comparison":
        render_comparison(properties)

    elif nav_option == "💡 AI Recommendations":
        render_recommendations(retriever)

    elif nav_option == "⚙️ Admin Management":
        render_admin(loader, vec_store, properties, summary)

    elif nav_option == "ℹ️ Architecture & About":
        render_architecture()


if __name__ == "__main__":
    main()