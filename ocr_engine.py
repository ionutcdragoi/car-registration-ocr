"""
OCR Engine for car registration document extraction.
Uses Tesseract OCR with OpenCV preprocessing to extract structured fields.
Optimized for European (especially Romanian) registration documents.
"""

import re
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pathlib import Path


# ---------------------------------------------------------------------------
# Image orientation helpers
# ---------------------------------------------------------------------------

def _detect_rotation_osd(gray: np.ndarray, tesseract_cmd: str | None = None) -> int:
    """Use Tesseract OSD to detect script orientation. Returns degrees to rotate."""
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    try:
        osd = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT)
        angle = osd.get("rotate", 0)
        return int(angle)
    except Exception:
        return 0


def _rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image by 0, 90, 180, or 270 degrees."""
    if angle == 0:
        return img
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _score_rotation_text(text: str) -> int:
    """Score OCR text for rotation selection using alphanumeric count + keyword bonuses."""
    score = sum(1 for c in text if c.isalnum())
    upper = text.upper()
    # Strong keywords that confirm correct orientation
    rotation_keywords = [
        "AUTOTURISM", "AUTOUTILITAR", "REMORCA", "MOTOCICL",
        "BENZINA", "MOTORINA", "DIESEL", "GPL", "HIBRID",
        "DACIA", "SANDERO", "LOGAN", "RENAULT", "VOLKSWAGEN", "OPEL",
        "BMW", "AUDI", "FORD", "FIAT", "SKODA", "TOYOTA", "HYUNDAI",
        "MERCEDES", "PEUGEOT", "CITROEN", "VOLVO", "SEAT", "KIA",
        "LEASING", "SECTORUL", "BUCURESTI", "ROMANIA",
        "LOCURI", "MASA", "CERTIFICAT", "INMATRICULARE",
        "ALB", "NEGRU", "GRI", "ALBASTRU", "ROSU", "VERDE",
        "ALEXANDRIEI", "PRECIZIEI",
    ]
    for kw in rotation_keywords:
        if kw in upper:
            score += 200
    # Bonus for EU field codes (A., B., D.1, P.1, etc.)
    eu_hits = len(re.findall(r'\b[ABDEFGIJKPRSVW]\.?\s*[123]?\s*\.?\s*[123]?\s*[\s:._\-]+\S', upper))
    score += eu_hits * 50
    return score


def _try_all_rotations(gray: np.ndarray, tesseract_cmd: str | None = None) -> np.ndarray:
    """
    Try 0, 90, 180, 270 rotations and pick the one with the most
    recognized text. Uses a downscaled preview for speed.
    """
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    # Use a downscaled version for fast rotation detection
    h, w = gray.shape
    max_dim = max(h, w)
    if max_dim > 2000:
        scale = 2000 / max_dim
        preview = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        preview = gray

    lang = _detect_languages(tesseract_cmd)
    best_angle = 0
    best_score = 0
    for angle in [0, 90, 180, 270]:
        candidate = _rotate_image(preview, angle)
        try:
            txt = pytesseract.image_to_string(
                candidate, config="--oem 3 --psm 6", lang=lang
            )
            score = _score_rotation_text(txt)
            if score > best_score:
                best_score = score
                best_angle = angle
        except Exception:
            continue

    # Apply the winning rotation to the full-size image
    return _rotate_image(gray, best_angle)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str, tesseract_cmd: str | None = None) -> list[np.ndarray]:
    """
    Apply multiple preprocessing techniques to improve OCR accuracy.
    Includes automatic rotation detection.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize if image is too small
    h, w = gray.shape
    if w < 1500:
        scale = 1500 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Cap very large images (e.g. 3840x2160) to speed up OCR
    h, w = gray.shape
    max_dim = max(h, w)
    if max_dim > 3000:
        scale = 3000 / max_dim
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Auto-rotation
    gray = _try_all_rotations(gray, tesseract_cmd)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=12)

    variants = []
    variants.append(denoised)

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    variants.append(adaptive)

    # Otsu threshold
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    # CLAHE + Otsu (light)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    _, enhanced_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(enhanced_otsu)

    # Sharpen + threshold
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    _, sharp_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(sharp_thresh)

    return variants


# ---------------------------------------------------------------------------
# OCR execution
# ---------------------------------------------------------------------------

def run_ocr(image_variants: list[np.ndarray], tesseract_cmd: str | None = None) -> str:
    """
    Run Tesseract on multiple image variants and return the best result.
    Scores by alphanumeric char count + known keyword bonus.
    """
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    best_text = ""
    best_score = 0

    lang = _detect_languages(tesseract_cmd)

    # psm 6 is best for structured docs, psm 4 for columns, psm 3 as auto
    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 4",
        "--oem 3 --psm 3",
    ]

    # Keywords that indicate good registration document OCR
    bonus_keywords = [
        "AUTOTURISM", "AUTOUTILITAR", "SANDERO", "DACIA", "BENZINA",
        "MOTORINA", "DIESEL", "LEASING", "SECTORUL", "BUCURESTI",
        "ALB", "NEGRU", "GRI", "LOCURI", "MASA", "VIN",
        "MERCEDES", "OPEL", "BMW", "FORD", "RENAULT", "HIBRID",
    ]

    for variant in image_variants:
        for config in configs:
            try:
                text = pytesseract.image_to_string(variant, config=config, lang=lang)
            except Exception:
                try:
                    text = pytesseract.image_to_string(variant, config=config, lang="eng")
                except Exception:
                    text = pytesseract.image_to_string(variant, config=config)

            score = sum(1 for c in text if c.isalnum())
            upper = text.upper()
            score += sum(100 for kw in bonus_keywords if kw in upper)

            if score > best_score:
                best_score = score
                best_text = text

    return best_text


def _detect_languages(tesseract_cmd: str | None = None) -> str:
    """Return the best lang string based on installed Tesseract languages."""
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    try:
        available = pytesseract.get_languages(config="")
        desired = ["ron", "eng", "deu", "fra", "spa", "ita"]
        langs = [l for l in desired if l in available]
        if langs:
            return "+".join(langs)
    except Exception:
        pass
    return "eng"


# ---------------------------------------------------------------------------
# EU field code parsing
# ---------------------------------------------------------------------------

def _parse_eu_field_codes(text: str) -> dict[str, str]:
    """
    Parse EU-style field codes from OCR text.
    Handles OCR noise like: D1 DACIA, P1999, F11618, C.3.1 PIZZA...
    """
    result: dict[str, str] = {}

    # Normalize common OCR misreads
    cleaned = text
    cleaned = re.sub(r'C€', 'C', cleaned)

    # Specific patterns for each EU field code (ordered by specificity)
    eu_codes = [
        # Multi-level codes first
        ("C.1.3", r"C\.?\s*1\.?\s*3[\s:._\-]+(.{5,120})"),
        ("C.1.1", r"C\.?\s*1\.?\s*1[\s:._\-]+(.{3,80})"),
        ("C.3.3", r"C\.?\s*3\.?\s*3[\s:._\-]*((?:SOS|STR|BD|CALEA|SPLAIUL|ALEEA)[^\n]{5,100})"),
        ("C.3.1", r"C\.?\s*3\.?\s*1[\s:._\-]+(.{3,80})"),
        ("V.7",   r"V\.?\s*7[\s:._\-]+(.{2,30})"),
        ("I.1",   r"I\.?\s*1[\s:._\-]+(.{5,30})"),
        ("D.1",   r"D\.?\s*1[\s:._\-]*([A-Z][A-Z\s\-]{1,30})"),
        ("D.2",   r"D\.?\s*2[\s:._\-]*([A-Z0-9][A-Z0-9\s\-/]{1,30})"),
        ("D.3",   r"D\.?\s*3[\s:._\-]*([A-Z][A-Z0-9\s\-]{1,30})"),
        ("P.1",   r"P\.?\s*1[\s:._\-]*(\d{3,4})"),
        ("P.2",   r"P\.?\s*2[\s:._\-]*(\d{2,4})"),
        ("P.3",   r"P\.?\s*3[\s:._\-]*([A-Z][A-Z\s+/\-]{2,30})"),
        ("F.1",   r"F\.?\s*1[\s:._\-]*(\d{3,5})"),
        ("F.2",   r"F\.?\s*2[\s:._\-]*(\d{3,5})"),
        ("S.1",   r"S\.?\s*[1I][\s:._\-]*(\d)"),
        ("S.2",   r"S\.?\s*[2][\s:._\-]*(\d{1,3})"),
        # Single-letter: be specific to avoid false positives
        ("A",     r"(?:^|[\s|(\[{])A[\s:._\-]+(B[\s\-]?\d{2,3}[\s\-]?[A-Z]{3})"),
        ("B",     r"(?:^|[\s|])B[\s:._\-]+(\d{2}[./\-]\d{2}[./\-]\d{4})"),
        ("E",     r"(?:^|[\s|])E[\s:._\-]+([A-HJ-NPR-Z0-9]{17})"),
        ("G",     r"(?:^|[\s|])G[\s:._\-]*(\d{3,5})"),
        ("I",     r"(?:^|[\s|])I[\s:._\-]+(\d{2}[./\-]\d{2}[./\-]\d{4})"),
        ("J",     r"(?:^|[\s|])J[\s:._\-]+([A-Z0-9][A-Z0-9\s]{1,30})"),
        ("K",     r"(?:^|[\s|])K[CK]*[\s:._\-]+(e\d{1,2}\*[^\s]{5,40})"),
        ("R",     r"(?:^|[\s|])R[\s:._\-]+([A-Z]{2,15})"),
    ]

    for code, pattern in eu_codes:
        if code not in result:
            m = re.search(pattern, cleaned, re.MULTILINE | re.IGNORECASE)
            if m:
                value = m.group(1).strip()
                if value:
                    result[code] = value

    # Also try to extract C.2 / CE C2 holder name
    if "C.2.1" not in result:
        m = re.search(
            r"C\.?\s*[E2][\s:._\-]*(?:C\.?2[\s:._\-]*)?"
            r"((?:RCI|[A-Z]{2,})[A-Z\s\.]{3,50}(?:IFN|IEN|SRL|SA|SNC|SCA)?)",
            cleaned, re.IGNORECASE
        )
        if m:
            result["C.2.1"] = m.group(1).strip()

    return result


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

KNOWN_MAKES = [
    "TOYOTA", "VOLKSWAGEN", "VW", "FORD", "BMW", "MERCEDES", "MERCEDES-BENZ",
    "AUDI", "OPEL", "RENAULT", "PEUGEOT", "CITROEN", "CITROËN", "FIAT", "SKODA",
    "ŠKODA", "SEAT", "HYUNDAI", "KIA", "NISSAN", "HONDA", "MAZDA", "SUZUKI",
    "VOLVO", "DACIA", "CHEVROLET", "MITSUBISHI", "SUBARU", "LEXUS", "PORSCHE",
    "LAND ROVER", "JAGUAR", "ALFA ROMEO", "JEEP", "MINI", "SMART", "TESLA",
    "FERRARI", "LAMBORGHINI", "MASERATI", "BENTLEY", "ROLLS ROYCE", "ASTON MARTIN",
    "CHRYSLER", "DODGE", "RAM", "BUICK", "CADILLAC", "GMC", "LINCOLN",
    "INFINITI", "ACURA", "GENESIS", "CUPRA", "MG", "BYD", "NIO", "LUCID",
    "RIVIAN", "POLESTAR", "SAAB", "LANCIA", "SSANGYONG", "ISUZU", "DAEWOO",
]

KNOWN_MODELS = [
    "SANDERO", "LOGAN", "DUSTER", "SPRING", "JOGGER", "STEPWAY",
    "GOLF", "POLO", "PASSAT", "TIGUAN", "T-ROC", "TOUAREG",
    "CLIO", "MEGANE", "CAPTUR", "KADJAR", "ARKANA",
    "CORSA", "ASTRA", "MOKKA", "CROSSLAND", "GRANDLAND",
    "FOCUS", "FIESTA", "KUGA", "PUMA", "MONDEO",
    "FABIA", "OCTAVIA", "SUPERB", "KODIAQ", "KAMIQ",
    "TUCSON", "KONA", "I30", "I20", "IONIQ",
    "SPORTAGE", "CEED", "NIRO", "PICANTO", "SORENTO",
    "SERIE", "CLASS", "X1", "X3", "X5",
    "YARIS", "COROLLA", "RAV4", "C-HR",
    "VIVARO", "COMBO", "ZAFIRA",
]

FUEL_KEYWORDS = {
    "BENZINA+GPL": "Petrol + LPG",
    "BENZINA + GPL": "Petrol + LPG",
    "BENZINA/GPL": "Petrol + LPG",
    "BENZINA+GPI": "Petrol + LPG",  # OCR misread of GPL
    "DIESEL": "Diesel",
    "MOTORINA": "Diesel",
    "GAZOLE": "Diesel",
    "BENZINA": "Petrol",
    "PETROL": "Petrol",
    "ESSENCE": "Petrol",
    "BENZIN": "Petrol",
    "GASOLINE": "Petrol",
    "ELECTRIC": "Electric",
    "HYBRID": "Hybrid",
    "HIBRID": "Hybrid",
    "GPL": "LPG",
    "LPG": "LPG",
    "GNC": "CNG",
    "CNG": "CNG",
}

COLORS = {
    "ALB": "White", "WHITE": "White", "WEISS": "White", "BLANC": "White", "BIANCO": "White",
    "NEGRU": "Black", "BLACK": "Black", "SCHWARZ": "Black", "NOIR": "Black", "NERO": "Black",
    "GRI": "Grey", "GREY": "Grey", "GRAY": "Grey", "GRAU": "Grey", "GRIS": "Grey",
    "ROSU": "Red", "RED": "Red", "ROT": "Red", "ROUGE": "Red", "ROSSO": "Red",
    "ALBASTRU": "Blue", "BLUE": "Blue", "BLAU": "Blue", "BLEU": "Blue", "BLU": "Blue",
    "VERDE": "Green", "GREEN": "Green", "GRÜN": "Green", "VERT": "Green",
    "GALBEN": "Yellow", "YELLOW": "Yellow", "GELB": "Yellow", "JAUNE": "Yellow",
    "ARGINTIU": "Silver", "SILVER": "Silver", "SILBER": "Silver", "ARGENT": "Silver",
    "MARO": "Brown", "BROWN": "Brown", "BRAUN": "Brown", "MARRON": "Brown",
    "PORTOCALIU": "Orange", "ORANGE": "Orange",
}


def extract_fields(raw_text: str) -> dict:
    """
    Parse the raw OCR text and extract structured car registration fields.
    Uses EU field codes first, then falls back to direct regex patterns.
    """
    fields = {
        "license_plate": "",
        "owner_name": "",
        "owner_address": "",
        "vehicle_make": "",
        "vehicle_model": "",
        "vehicle_type": "",
        "vin": "",
        "registration_date": "",
        "first_registration_date": "",
        "engine_capacity_cc": "",
        "engine_power_kw": "",
        "engine_power_hp": "",
        "fuel_type": "",
        "color": "",
        "num_seats": "",
        "max_weight_kg": "",
        "co2_emissions": "",
        "mileage": "",
        "raw_text": raw_text.strip(),
    }

    text = raw_text.upper()

    # Parse EU field codes
    eu = _parse_eu_field_codes(text)

    # ---- License Plate (field A) ----
    if "A" in eu:
        fields["license_plate"] = eu["A"]
    if not fields["license_plate"]:
        m = re.search(r"\b(B[\s\-_.]*\d{2,3}[\s\-_.]*[A-Z]{3})\b", text)
        if m:
            fields["license_plate"] = m.group(1)
        else:
            m = re.search(r"\b([A-Z]{1,2}[\s\-_.]*\d{2,3}[\s\-_.]*[A-Z]{3})\b", text)
            if m:
                fields["license_plate"] = m.group(1)

    # ---- VIN (field E) ----
    if "E" in eu:
        fields["vin"] = eu["E"]
    if not fields["vin"]:
        m = re.search(r"\b([A-HJ-NPR-Z0-9]{17})\b", text)
        if m:
            fields["vin"] = m.group(1)
        else:
            m = re.search(r"(?:^|[\s|])E\s+([A-Z0-9]{17})", text, re.MULTILINE)
            if m:
                fields["vin"] = m.group(1)

    # ---- Vehicle Make (field D.1) ----
    if "D.1" in eu:
        val = eu["D.1"].upper()
        for make in KNOWN_MAKES:
            if make in val:
                fields["vehicle_make"] = make
                break
        if not fields["vehicle_make"]:
            fields["vehicle_make"] = val.split()[0]
    if not fields["vehicle_make"]:
        m = re.search(r"D\.?\s*1[\s:._\-]*([A-Z]{2,20})", text)
        if m:
            candidate = m.group(1).strip()
            for make in KNOWN_MAKES:
                if make in candidate:
                    fields["vehicle_make"] = make
                    break
            if not fields["vehicle_make"]:
                fields["vehicle_make"] = candidate
    if not fields["vehicle_make"]:
        for make in KNOWN_MAKES:
            if make in text:
                fields["vehicle_make"] = make
                break

    # ---- Vehicle Model (field D.3 or text scan) ----
    # Words to strip from model field (fuel types, noise)
    _model_noise = {"BENZINA", "MOTORINA", "DIESEL", "GPL", "ELECTRIC", "HYBRID",
                    "BENZINA+GPL", "ALB", "NEGRU", "GRI", "ALBASTRU", "ROSU", "VERDE",
                    "@", "PA", "NARA"}

    def _clean_model(raw: str) -> str:
        words = raw.upper().split("\n")[0].strip().split()
        cleaned = [w for w in words if w not in _model_noise and not re.match(r'^\d{1,2}$', w)]
        return " ".join(cleaned[:3])

    if "D.3" in eu:
        fields["vehicle_model"] = _clean_model(eu["D.3"])

    if not fields["vehicle_model"]:
        m = re.search(r"D\.?\s*3[\s:._\-]*([A-Z][A-Z0-9\s\-]{1,20})", text)
        if m:
            fields["vehicle_model"] = _clean_model(m.group(1))

    if not fields["vehicle_model"]:
        for model in KNOWN_MODELS:
            if model in text:
                fields["vehicle_model"] = model
                break

    # ---- Vehicle Type ----
    # Match full vehicle type words including suffixes like AUTOUTILITARA N1
    vtype_pat = r"(AUTOTURISM(?:E)?|AUTOUTILITAR(?:A|E)?|MOTOCICLET(?:A|E)?|REMORCA)\s*([A-Z]?\s*\d{0,2})"
    m_vtype = re.search(vtype_pat, text)
    if m_vtype:
        base = m_vtype.group(1).strip()
        qualifier = m_vtype.group(2).strip() if m_vtype.group(2) else ""
        fields["vehicle_type"] = (base + " " + qualifier).strip()
    if not fields["vehicle_type"] and "D.2" in eu:
        fields["vehicle_type"] = eu["D.2"].strip()
    if not fields["vehicle_type"] and "J" in eu:
        fields["vehicle_type"] = eu["J"].strip()

    # ---- Registration Date ----
    date_pat = r"(\d{2}[./\-]\d{2}[./\-]\d{4})"

    if "I" in eu:
        m = re.search(date_pat, eu["I"])
        if m:
            fields["registration_date"] = m.group(1)
    if "I.1" in eu and not fields["registration_date"]:
        m = re.search(date_pat, eu["I.1"])
        if m:
            fields["registration_date"] = m.group(1)

    if "B" in eu:
        m = re.search(date_pat, eu["B"])
        if m:
            fields["first_registration_date"] = m.group(1)

    if not fields["registration_date"]:
        all_dates = re.findall(date_pat, text)
        if all_dates:
            fields["registration_date"] = all_dates[0]
            if len(all_dates) > 1:
                fields["first_registration_date"] = all_dates[1]

    if not fields["first_registration_date"]:
        fields["first_registration_date"] = fields["registration_date"]

    # ---- Engine Capacity (P.1) ----
    if "P.1" in eu:
        fields["engine_capacity_cc"] = eu["P.1"]
    if not fields["engine_capacity_cc"]:
        m = re.search(r"P\.?\s*1[\s:._\-]*(\d{3,4})", text)
        if m:
            fields["engine_capacity_cc"] = m.group(1)
    if not fields["engine_capacity_cc"]:
        m = re.search(r"(\d{3,4})\s*(?:CC|CM[³3]|CCM|CMC)", text)
        if m:
            fields["engine_capacity_cc"] = m.group(1)
    # Fallback: look for 3-4 digit number near "B" field code (OCR often reads P1 as Bi/B1)
    if not fields["engine_capacity_cc"]:
        m = re.search(r"B[iI1]?\s*(\d{3,4})(?:\s|$)", text)
        if m:
            val = int(m.group(1))
            # Engine capacities typically 500-9999, exclude years and weights
            if 500 <= val <= 9999 and val != int(fields.get("max_weight_kg") or 0):
                fields["engine_capacity_cc"] = str(val)

    # ---- Engine Power kW (P.2) ----
    if "P.2" in eu:
        fields["engine_power_kw"] = eu["P.2"]
    if not fields["engine_power_kw"]:
        m = re.search(r"P\.?\s*2[\s:._\-]*(\d{2,4})", text)
        if m:
            fields["engine_power_kw"] = m.group(1)
    if not fields["engine_power_kw"]:
        m = re.search(r"(\d{2,4})\s*KW", text)
        if m:
            fields["engine_power_kw"] = m.group(1)

    # ---- Engine Power HP ----
    m = re.search(r"(\d{2,4})\s*(?:HP|CP|PS|CV|CH)", text)
    if m:
        fields["engine_power_hp"] = m.group(1)

    # ---- Fuel Type (P.3) ----
    # Scan full text first (more reliable than EU field which may be misread)
    for keyword, fuel in FUEL_KEYWORDS.items():
        if keyword in text:
            fields["fuel_type"] = fuel
            break

    if not fields["fuel_type"] and "P.3" in eu:
        p3 = eu["P.3"].upper()
        for keyword, fuel in FUEL_KEYWORDS.items():
            if keyword in p3:
                fields["fuel_type"] = fuel
                break

    # ---- Color (R) ----
    if "R" in eu:
        for keyword, color in COLORS.items():
            if keyword in eu["R"].upper():
                fields["color"] = color
                break
    if not fields["color"]:
        # Look for R_ALB, R ALB, R.ALB, SALB (OCR merges S+ALB)
        m = re.search(r"(?:^|[\s|])R[\s:._\-]+([A-Z]{2,15})", text, re.MULTILINE)
        if m:
            for keyword, color in COLORS.items():
                if keyword in m.group(1):
                    fields["color"] = color
                    break
    if not fields["color"]:
        # OCR sometimes merges: "SALBOO" or "| SALBOO" — extract ALB from it
        m = re.search(r"S?ALB(?:OO?|\b)", text)
        if m:
            fields["color"] = "White"
    if not fields["color"]:
        for keyword, color in COLORS.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", text):
                fields["color"] = color
                break

    # ---- Number of Seats (S.1) ----
    if "S.1" in eu:
        fields["num_seats"] = eu["S.1"]
    if not fields["num_seats"]:
        # S.1, S1, S 1 — also handle OCR reading S1 as SI (I/1 confusion)
        m = re.search(r"S\.?\s*[1I][\s:._\-]*(\d{1,2})", text)
        if m:
            fields["num_seats"] = m.group(1)

    # ---- Max Weight (F.1) ----
    if "F.1" in eu:
        fields["max_weight_kg"] = eu["F.1"]
    if not fields["max_weight_kg"]:
        m = re.search(r"F\.?\s*1[\s:._\-]*(\d{3,5})", text)
        if m:
            fields["max_weight_kg"] = m.group(1)
    if not fields["max_weight_kg"] and "G" in eu:
        m = re.search(r"(\d{3,5})", eu["G"])
        if m:
            fields["max_weight_kg"] = m.group(1)

    # ---- CO2 Emissions (V.7) ----
    if "V.7" in eu:
        m = re.search(r"(\d{2,4})", eu["V.7"])
        if m:
            fields["co2_emissions"] = m.group(1) + " g/km"
    if not fields["co2_emissions"]:
        m = re.search(r"(?:CO2|CO₂|EMISII)[\s:]*(\d{2,4})\s*(?:G/KM|G)", text)
        if m:
            fields["co2_emissions"] = m.group(1) + " g/km"

    # ---- Owner Name ----
    if "C.2.1" in eu:
        name = eu["C.2.1"].strip()
        name = re.split(r"\s+[A-Z]\.\d", name)[0].strip()
        if len(name) > 2:
            fields["owner_name"] = name.title()

    if not fields["owner_name"]:
        m = re.search(
            r"C\.?\s*[E2][\s*:._\-]*(?:C\.?2[\s:._\-]*)?"
            r"([A-Z][A-Z\s\.]{3,50}(?:IFN|IEN|SRL|SA|SNC|SCA))",
            text
        )
        if m:
            fields["owner_name"] = m.group(1).strip().title()

    if not fields["owner_name"]:
        for key in ["C.1.1", "C.1", "C.3.1"]:
            if key in eu:
                name = eu[key].split("\n")[0].strip()
                name = re.split(r"\s+[A-Z]\.\d", name)[0].strip()
                if len(name) > 2:
                    fields["owner_name"] = name.title()
                    break

    if not fields["owner_name"]:
        for label in ["TITULAR", "PROPRIETAR", "OWNER", "INHABER"]:
            pattern = label + r"[\s:]+([A-ZĂÂÎȘȚÉÈÊËÀÙÛÜÔÖÏÎÇÑ][A-ZĂÂÎȘȚÉÈÊËÀÙÛÜÔÖÏÎÇÑ\s\.\-]{2,50})"
            m = re.search(pattern, text)
            if m:
                name = m.group(1).strip().split("\n")[0].strip()
                if len(name) > 3:
                    fields["owner_name"] = name.title()
                    break

    # ---- Owner Address ----
    if "C.3.3" in eu:
        fields["owner_address"] = eu["C.3.3"].strip().title()
    elif "C.1.3" in eu:
        fields["owner_address"] = eu["C.1.3"].strip().title()

    if not fields["owner_address"]:
        m = re.search(
            r"C\.?\s*3\.?\s*3[\s:._\-]*((?:SOS|STR|BD|CALEA|SPLAIUL|ALEEA)[^\n]{5,100})",
            text, re.IGNORECASE
        )
        if m:
            fields["owner_address"] = m.group(1).strip().title()

    if not fields["owner_address"]:
        for label in ["ADRESA", "ADDRESS", "ADRESSE", "DOMICILIU"]:
            m = re.search(label + r"[\s:]+(.{5,100})", text)
            if m:
                fields["owner_address"] = m.group(1).strip().title()
                break

    return fields


def process_image(image_path: str, tesseract_cmd: str | None = None) -> dict:
    """
    Full pipeline: preprocess image -> OCR -> extract fields.
    Returns a dict of extracted fields.
    """
    variants = preprocess_image(image_path, tesseract_cmd)
    raw_text = run_ocr(variants, tesseract_cmd)
    fields = extract_fields(raw_text)
    return fields
