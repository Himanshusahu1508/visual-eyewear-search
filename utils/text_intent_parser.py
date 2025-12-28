import re

SHAPES = ["round", "rectangle", "aviator", "square"]
MATERIALS = ["metal", "acetate", "plastic"]
COLORS = ["black", "silver", "gold", "brown"]

def parse_text_intent(text: str):
    text = text.lower()

    intent = {
        "shape": None,
        "material": None,
        "color": None,
        "price_max": None
    }

    for s in SHAPES:
        if s in text:
            intent["shape"] = s.capitalize()

    for m in MATERIALS:
        if m in text:
            intent["material"] = m

    for c in COLORS:
        if c in text:
            intent["color"] = c

    # simple price intent
    if "cheap" in text or "budget" in text:
        intent["price_max"] = 1500

    return intent
