
import sys
import streamlit as st
import torch

from PIL import Image
from transformers import pipeline

st.write("Python version:", sys.version)


# =========================================================
# 1. LOAD AI MODEL
# =========================================================

MODEL_ID = "google/siglip-base-patch16-224"


@st.cache_resource(show_spinner="Loading AI model...")
def load_classifier():

    return pipeline(
        "zero-shot-image-classification",
        model=MODEL_ID,
        device=-1
    )


classifier = load_classifier()


# =========================================================
# 2. E-WASTE CATEGORIES
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
# 3. AI LABEL → BUSINESS CATEGORY
# =========================================================

LABEL_MAP = {
    "an old mobile phone": "mobile phone",
    "an old laptop computer": "laptop",
    "an old desktop computer": "desktop computer",
    "an old television": "television",
    "an old refrigerator": "refrigerator",
    "an old washing machine": "washing machine",
    "an old microwave oven": "microwave oven",
    "an old printer": "printer",
    "an old keyboard": "keyboard",
    "an old computer mouse": "computer mouse",
    "an electronic battery": "battery",
    "a printed circuit board": "PCB"
}


# =========================================================
# 4. DEMO PRICE DATABASE
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
# 5. CONDITION FACTORS
# =========================================================

CONDITION_MULTIPLIER = {
    "Working": 1.00,
    "Partially Working": 0.80,
    "Non-working": 0.60
}


# =========================================================
# 6. DEMO RECYCLERS
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
# 7. AI DETECTION
# =========================================================

def detect_e_waste(image):

    try:

        if image is None:
            raise ValueError("Image is empty.")

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
        model_score = best_result["score"]

        if predicted_label not in LABEL_MAP:
            raise ValueError(
                f"Unknown model label: {predicted_label}"
            )

        clean_category = LABEL_MAP[predicted_label]

        return (
            clean_category,
            model_score,
            results
        )

    except Exception as e:

        raise RuntimeError(
            f"SigLIP inference failed: "
            f"{type(e).__name__}: {str(e)}"
        ) from e


# =========================================================
# 8. PRICE CALCULATION
# =========================================================

def calculate_valuation(category, weight, condition):

    weight = float(weight)

    if weight <= 0:
        raise ValueError(
            "Weight must be greater than zero."
        )

    base = PRICE_TABLE[category]

    midpoint = (
        base["low"] +
        base["high"]
    ) / 2

    condition_factor = CONDITION_MULTIPLIER[
        condition
    ]

    estimated = (
        midpoint *
        condition_factor *
        (weight / 2)
    )

    return {
        "base_low": base["low"],
        "base_high": base["high"],
        "estimated": round(estimated)
    }


# =========================================================
# 9. RECYCLER MATCHING
# =========================================================

def find_matching_recyclers(
    category,
    estimated_price,
    location
):

    matches = []

    for recycler in RECYCLERS:

        if category not in recycler["categories"]:
            continue

        if (
            location.strip().lower()
            != recycler["location"].lower()
        ):
            continue

        offer = round(
            estimated_price *
            recycler["price_factor"]
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
# 10. MAIN AI FUNCTION
# =========================================================

def analyze_e_waste(
    image,
    weight,
    condition,
    location
):

    if image is None:
        return (
            "❌ Please upload an image.",
            [],
            []
        )

    # AI detection
    category, model_score, raw_results = (
        detect_e_waste(image)
    )

    # Valuation
    valuation = calculate_valuation(
        category,
        weight,
        condition
    )

    estimated_price = valuation["estimated"]

    # Recycler matching
    recycler_rows = find_matching_recyclers(
        category,
        estimated_price,
        location
    )

    # Top AI predictions
    prediction_rows = []

    for result in raw_results[:5]:

        prediction_rows.append([
            LABEL_MAP.get(
                result["label"],
                result["label"]
            ),
            round(
                result["score"] * 100,
                2
            )
        ])

    # Final text output
    analysis = f"""
# ♻️ ScrapOS AI Analysis

## 🔍 Detected E-Waste

### {category.title()}

**AI Match Score:** {model_score:.2%}

---

## 📋 Input Information

**Weight:** {weight} kg

**Condition:** {condition}

**Location:** {location}

---

# 💰 Indicative Value

## ₹{estimated_price:,}

Base category range:

**₹{valuation["base_low"]:,} – ₹{valuation["base_high"]:,}**

> Prototype valuation using demo rates.
> Production ScrapOS will use live recycler-submitted rates.

---

## ♻️ Recycler Matching

Found **{len(recycler_rows)}** matching recycler(s).
"""

    return (
        analysis,
        prediction_rows,
        recycler_rows
    )


# =========================================================
# STREAMLIT UI
# =========================================================

st.title("♻️ ScrapOS")

st.subheader(
    "AI-Powered E-Waste Identification & Recycler Matching"
)

st.write(
    "Upload an image of e-waste to identify the item, "
    "estimate its indicative value and discover matching recyclers."
)


# =========================================================
# INPUT SECTION
# =========================================================

col1, col2 = st.columns(2)


with col1:

    uploaded_file = st.file_uploader(
        "📷 Upload E-Waste Image",
        type=["jpg", "jpeg", "png", "webp"]
    )


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
        ]
    )

    location = st.text_input(
        "📍 Location",
        value="Mumbai"
    )


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze_button = st.button(
    "🔍 Analyze E-Waste",
    type="primary"
)


# =========================================================
# ANALYSIS
# =========================================================

if analyze_button:

    if uploaded_file is None:

        st.error(
            "Please upload an e-waste image."
        )

    else:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        with st.spinner(
            "🤖 AI analyzing e-waste..."
        ):

            if analyze_button:

    if uploaded_file is None:

        st.error(
            "Please upload an e-waste image."
        )

    else:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.info(
                f"Image loaded: {image.size[0]} × {image.size[1]}"
            )

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
                    image,
                    weight,
                    condition,
                    location
                )

            st.success(
                f"Detected: {category.title()}"
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
                    "No matching recyclers found."
                )

        except Exception as e:

            st.error(
                "❌ AI analysis failed."
            )

            st.exception(e)


        # =====================================================
        # RESULT SECTION
        # =====================================================

        st.divider()

        left, right = st.columns(2)


        # -----------------------------
        # IMAGE
        # -----------------------------

        with left:

            st.image(
                image,
                caption="Uploaded E-Waste",
                use_container_width=True
            )


        # -----------------------------
        # AI RESULT
        # -----------------------------

        with right:

            st.success(
                f"Detected: {category.title()}"
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


        # =====================================================
        # RECYCLERS
        # =====================================================

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
                "No matching recyclers found "
                "for this location."
            )


        # =====================================================
        # AI PREDICTIONS
        # =====================================================

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


# =========================================================
# 12. LAUNCH
# ========================================================
