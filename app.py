
import torch
import gradio as gr
import spaces

from PIL import Image
from transformers import pipeline


# =========================================================
# 1. LOAD AI MODEL
# =========================================================

MODEL_ID = "google/siglip-base-patch16-224"

device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "zero-shot-image-classification",
    model=MODEL_ID,
    device=device
)


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

    results = classifier(
        image,
        candidate_labels=CATEGORIES
    )

    best_result = results[0]

    predicted_label = best_result["label"]
    model_score = best_result["score"]

    clean_category = LABEL_MAP[predicted_label]

    return clean_category, model_score, results


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

@spaces.GPU
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
# 11. GRADIO UI
# =========================================================

with gr.Blocks(
    title="ScrapOS AI E-Waste Valuation"
) as demo:

    gr.Markdown(
        """
        # ♻️ ScrapOS
        ### AI-Powered E-Waste Identification & Recycler Matching

        Upload or capture an image of e-waste,
        estimate its indicative value,
        and discover matching recyclers.
        """
    )

    with gr.Row():

        # -----------------------------
        # INPUT
        # -----------------------------

        with gr.Column():

            image_input = gr.Image(
                sources=["upload", "webcam"],
                type="pil",
                label="📷 Scan E-Waste"
            )

            weight_input = gr.Number(
                label="⚖️ Approximate Weight (kg)",
                value=2,
                minimum=0.1
            )

            condition_input = gr.Radio(
                choices=[
                    "Working",
                    "Partially Working",
                    "Non-working"
                ],
                value="Non-working",
                label="🔧 Condition"
            )

            location_input = gr.Textbox(
                label="📍 Location",
                value="Mumbai"
            )

            analyze_button = gr.Button(
                "🔍 Analyze E-Waste",
                variant="primary"
            )

        # -----------------------------
        # OUTPUT
        # -----------------------------

        with gr.Column():

            result_output = gr.Markdown()

            prediction_output = gr.Dataframe(
                headers=[
                    "Detected Category",
                    "AI Match Score (%)"
                ],
                datatype=[
                    "str",
                    "number"
                ],
                label="🤖 AI Predictions",
                interactive=False
            )

            recycler_output = gr.Dataframe(
                headers=[
                    "Recycler",
                    "Location",
                    "Offer",
                    "Pickup"
                ],
                datatype=[
                    "str",
                    "str",
                    "str",
                    "str"
                ],
                label="♻️ Matching Recyclers",
                interactive=False
            )

    analyze_button.click(
        fn=analyze_e_waste,
        inputs=[
            image_input,
            weight_input,
            condition_input,
            location_input
        ],
        outputs=[
            result_output,
            prediction_output,
            recycler_output
        ]
    )


# =========================================================
# 12. LAUNCH
# =========================================================

demo.launch()
