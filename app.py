import streamlit as st
import torch
from PIL import Image
from transformers import pipeline


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ScrapOS",
    page_icon="♻️",
    layout="wide"
)


# =========================================================
# AI MODEL
# =========================================================

MODEL_ID = "google/siglip-base-patch16-224"


@st.cache_resource
def load_classifier():
    return pipeline(
        task="zero-shot-image-classification",
        model=MODEL_ID,
        device=-1
    )


try:
    classifier = load_classifier()

except Exception as e:
    st.error("❌ AI model could not be loaded.")
    st.exception(e)
    st.stop()


# =========================================================
# E-WASTE CATEGORIES
# =========================================================

CATEGORIES = [
    "an old mobile phone",
    "an old laptop computer",
    "an old desktop computer",
    "an old television",
    "an old refrigerator",
    "an old washing machine",
    "an old microwave oven",
    "an old printer",
    "an old keyboard",
    "an old computer mouse",
    "an electronic battery",
    "a printed circuit board"
]


# =========================================================
# AI LABEL → BUSINESS CATEGORY
# =========================================================

LABEL_MAP = {

    "an old mobile phone":
        "mobile phone",

    "an old laptop computer":
        "laptop",

    "an old desktop computer":
        "desktop computer",

    "an old television":
        "television",

    "an old refrigerator":
        "refrigerator",

    "an old washing machine":
        "washing machine",

    "an old microwave oven":
        "microwave oven",

    "an old printer":
        "printer",

    "an old keyboard":
        "keyboard",

    "an old computer mouse":
        "computer mouse",

    "an electronic battery":
        "battery",

    "a printed circuit board":
        "PCB"
}


# =========================================================
# DEMO PRICE DATABASE
# NOTE:
# These are prototype/demo values.
# =========================================================

PRICE_TABLE = {

    "mobile phone": {
        "low": 300,
        "high": 1200
    },

    "laptop": {
        "low": 800,
        "high": 2500
    },

    "desktop computer": {
        "low": 500,
        "high": 2200
    },

    "television": {
        "low": 300,
        "high": 1800
    },

    "refrigerator": {
        "low": 800,
        "high": 3000
    },

    "washing machine": {
        "low": 500,
        "high": 2500
    },

    "microwave oven": {
        "low": 200,
        "high": 1000
    },

    "printer": {
        "low": 150,
        "high": 800
    },

    "keyboard": {
        "low": 50,
        "high": 200
    },

    "computer mouse": {
        "low": 30,
        "high": 150
    },

    "battery": {
        "low": 100,
        "high": 1200
    },

    "PCB": {
        "low": 500,
        "high": 3000
    }
}


# =========================================================
# CONDITION FACTORS
# =========================================================

CONDITION_MULTIPLIER = {

    "Working": 1.00,

    "Partially Working": 0.80,

    "Non-working": 0.60
}


# =========================================================
# DEMO RECYCLERS
# =========================================================

RECYCLERS = [

    {
        "name": "Recycler A",

        "location": "Mumbai",

        "categories": [
            "mobile phone",
            "laptop",
            "desktop computer",
            "PCB"
        ],

        "price_factor": 1.00,

        "pickup": True
    },

    {
        "name": "Recycler B",

        "location": "Mumbai",

        "categories": [
            "laptop",
            "television",
            "printer",
            "washing machine"
        ],

        "price_factor": 1.10,

        "pickup": True
    },

    {
        "name": "Recycler C",

        "location": "Mumbai",

        "categories": [
            "mobile phone",
            "battery",
            "PCB",
            "desktop computer"
        ],

        "price_factor": 0.95,

        "pickup": False
    },

    {
        "name": "Recycler D",

        "location": "Mumbai",

        "categories": [
            "refrigerator",
            "washing machine",
            "microwave oven",
            "television"
        ],

        "price_factor": 1.05,

        "pickup": True
    }
]


# =========================================================
# AI DETECTION
# =========================================================

def detect_e_waste(image):

    if image is None:
        raise ValueError(
            "No image was provided."
        )

    results = classifier(
        image,
        candidate_labels=CATEGORIES
    )

    if not results:
        raise ValueError(
            "SigLIP returned no predictions."
        )

    best_result = results[0]

    predicted_label = best_result["label"]

    model_score = float(
        best_result["score"]
    )

    if predicted_label not in LABEL_MAP:
        raise ValueError(
            f"Unknown model label returned: "
            f"{predicted_label}"
        )

    clean_category = LABEL_MAP[
        predicted_label
    ]

    return (
        clean_category,
        model_score,
        results
    )


# =========================================================
# PRICE CALCULATION
# =========================================================

def calculate_valuation(
    category,
    weight,
    condition
):

    if category not in PRICE_TABLE:
        raise ValueError(
            f"No price data exists for "
            f"category: {category}"
        )

    weight = float(weight)

    if weight <= 0:
        raise ValueError(
            "Weight must be greater than zero."
        )

    if condition not in CONDITION_MULTIPLIER:
        raise ValueError(
            f"Unknown condition: {condition}"
        )

    base = PRICE_TABLE[
        category
    ]

    midpoint = (
        base["low"] +
        base["high"]
    ) / 2

    condition_factor = (
        CONDITION_MULTIPLIER[
            condition
        ]
    )

    estimated = (
        midpoint *
        condition_factor *
        (weight / 2)
    )

    return {

        "base_low":
            base["low"],

        "base_high":
            base["high"],

        "estimated":
            int(round(estimated))
    }


# =========================================================
# RECYCLER MATCHING
# =========================================================

def find_matching_recyclers(
    category,
    estimated_price,
    location
):

    matches = []

    normalized_location = (
        location
        .strip()
        .lower()
    )

    for recycler in RECYCLERS:

        if category not in recycler[
            "categories"
        ]:
            continue

        if (
            normalized_location
            != recycler[
                "location"
            ].lower()
        ):
            continue

        offer = int(
            round(
                estimated_price *
                recycler[
                    "price_factor"
                ]
            )
        )

        pickup = (

            "Pickup Available"

            if recycler["pickup"]

            else "Drop-off"
        )

        matches.append([

            recycler["name"],

            recycler["location"],

            f"₹{offer:,}",

            pickup
        ])

    return matches


# =========================================================
# MASTER ANALYSIS FUNCTION
#
# IMPORTANT:
# This returns EXACTLY 6 values.
# =========================================================

def analyze_e_waste(
    image,
    weight,
    condition,
    location
):

    if image is None:
        raise ValueError(
            "No image was uploaded."
        )

    # -----------------------------------------------------
    # 1. AI IDENTIFICATION
    # -----------------------------------------------------

    (
        category,
        model_score,
        raw_results
    ) = detect_e_waste(
        image
    )


    # -----------------------------------------------------
    # 2. VALUATION
    # -----------------------------------------------------

    valuation = calculate_valuation(

        category=category,

        weight=weight,

        condition=condition
    )

    estimated_price = (
        valuation["estimated"]
    )


    # -----------------------------------------------------
    # 3. RECYCLER MATCHING
    # -----------------------------------------------------

    recycler_rows = (
        find_matching_recyclers(

            category=category,

            estimated_price=
                estimated_price,

            location=location
        )
    )


    # -----------------------------------------------------
    # 4. RETURN EXACTLY 6 VALUES
    # -----------------------------------------------------

    return (

        category,

        model_score,

        estimated_price,

        {
            "low":
                valuation["base_low"],

            "high":
                valuation["base_high"]
        },

        raw_results,

        recycler_rows
    )


# =========================================================
# STREAMLIT UI
# =========================================================

st.title(
    "♻️ ScrapOS"
)

st.subheader(
    "AI-Powered E-Waste Identification "
    "& Recycler Matching"
)

st.write(
    "Upload an image of e-waste to identify "
    "the item, estimate its indicative value "
    "and discover matching recyclers."
)


# =========================================================
# INPUT SECTION
# =========================================================

col1, col2 = st.columns(2)


# ---------------------------------------------------------
# IMAGE
# ---------------------------------------------------------

with col1:

    uploaded_file = st.file_uploader(

        "📷 Upload E-Waste Image",

        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )


# ---------------------------------------------------------
# DETAILS
# ---------------------------------------------------------

with col2:

    weight = st.number_input(

        "⚖️ Approximate Weight (kg)",

        min_value=0.1,

        value=2.0,

        step=0.1
    )


    condition = st.radio(

        "🔧 Condition",

        [
            "Working",
            "Partially Working",
            "Non-working"
        ],

        index=0
    )


    location = st.text_input(

        "📍 Location",

        value="Mumbai"
    )


# =========================================================
# ONE AND ONLY ONE ANALYZE BUTTON
# =========================================================

analyze_button = st.button(

    "🔍 Analyze E-Waste",

    type="primary",

    key="analyze_e_waste_button"
)


# =========================================================
# ANALYSIS
# =========================================================

if analyze_button:

    # -----------------------------------------------------
    # CHECK IMAGE
    # -----------------------------------------------------

    if uploaded_file is None:

        st.error(
            "Please upload an e-waste image."
        )

        st.stop()


    try:

        # -------------------------------------------------
        # LOAD IMAGE
        # -------------------------------------------------

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        st.info(
            f"Image loaded: "
            f"{image.size[0]} × "
            f"{image.size[1]}"
        )


        # -------------------------------------------------
        # RUN COMPLETE PIPELINE
        # -------------------------------------------------

        with st.spinner(
            "🤖 AI analyzing e-waste..."
        ):

            (

                category,

                score,

                estimated_price,

                base,

                results,

                recyclers

            ) = analyze_e_waste(

                image=image,

                weight=weight,

                condition=condition,

                location=location
            )


        # =================================================
        # RESULT SECTION
        # =================================================

        st.divider()


        result_left, result_right = (
            st.columns(2)
        )


        # -------------------------------------------------
        # IMAGE
        # -------------------------------------------------

        with result_left:

            st.image(

                image,

                caption=
                    "Uploaded E-Waste",

                use_container_width=True
            )


        # -------------------------------------------------
        # AI RESULT
        # -------------------------------------------------

        with result_right:

            st.success(

                f"Detected: "
                f"{category.title()}"
            )


            st.metric(

                "AI Match Score",

                f"{score:.2%}"
            )


            st.metric(

                "Indicative Value",

                f"₹{estimated_price:,}"
            )


            st.caption(

                f"Prototype category range: "

                f"₹{base['low']:,} – "
                f"₹{base['high']:,}"
            )


        # =================================================
        # RECYCLER RESULTS
        # =================================================

        st.divider()

        st.subheader(
            "♻️ Matching Recyclers"
        )


        if recyclers:

            st.dataframe(

                recyclers,

                use_container_width=True,

                hide_index=True
            )

        else:

            st.warning(

                "No matching recyclers "
                "found for this location."
            )


        # =================================================
        # AI PREDICTION BREAKDOWN
        # =================================================

        st.divider()

        st.subheader(
            "🤖 AI Prediction Breakdown"
        )


        prediction_data = []


        for result in results[:5]:

            prediction_data.append({

                "Category":

                    LABEL_MAP.get(

                        result["label"],

                        result["label"]
                    ),

                "Match Score":

                    f"{result['score']:.2%}"
            })


        st.dataframe(

            prediction_data,

            use_container_width=True,

            hide_index=True
        )


        # -------------------------------------------------
        # DISCLAIMER
        # -------------------------------------------------

        st.caption(

            "Prototype valuation uses demo rates. "
            "Production ScrapOS should use live "
            "recycler-submitted rates."
        )


    except Exception as e:

        st.error(
            "❌ AI analysis failed."
        )

        st.exception(e)
