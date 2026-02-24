import os
import re
import cv2
import pytesseract
import pandas as pd
from PIL import Image
from difflib import get_close_matches

# Configuration
IMAGE_FOLDER = "images"
SNACK_DB = "snack_data.csv"

# ---------------- OCR UTILITIES ---------------- #

def clean_ocr_text(text):
    """Extract meaningful words from noisy OCR text"""
    text = re.sub(r'[^a-zA-Z0-9\s&]', ' ', text)
    text = re.sub(r'\b[a-zA-Z]\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    meaningful_words = [word for word in words if len(word) >= 3]
    return ' '.join(meaningful_words)


def adaptive_ocr(image_path):
    """Adaptive OCR that tries multiple strategies"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if max(h, w) < 1500:
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    all_text = ""
    strategies = []

    # Strategy 1: Simple Otsu threshold
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strategies.append(thresh1)

    # Strategy 2: Adaptive Gaussian threshold
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 10)
    strategies.append(thresh2)

    # Strategy 3: Denoising + Otsu
    denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
    _, thresh3 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strategies.append(thresh3)

    for processed in strategies:
        pil_img = Image.fromarray(processed)
        for psm in [6, 7, 8, 13]:
            config = f"--psm {psm}"
            try:
                text = pytesseract.image_to_string(pil_img, config=config)
                if text.strip():
                    cleaned = clean_ocr_text(text)
                    if cleaned:
                        all_text += f" {cleaned}"
            except:
                continue

    return clean_ocr_text(all_text)


def extract_product_keywords(ocr_text):
    """Extract potential product keywords from OCR text"""
    if not ocr_text:
        return []

    product_keywords = []
    brands = [
        'karachi', 'kurkure', 'lays', 'britannia', 'sunfeast', 'unibic',
        'cadbury', 'bikano', 'beyond', 'cornitos', 'haldiram', 'urban',
        'kettle', 'chef', 'cookie', 'bingo', 'too yumm'
    ]
    product_types = [
        'osmania', 'coconut', 'butter', 'choc', 'chunk', 'choco', 'oreo',
        'oats', 'almond', 'cashew', 'digestive', 'bourbon', 'sugar free',
        'masala', 'munch', 'puffcorn', 'salted', 'cream', 'onion', 'pepper',
        'tikka', 'sour', 'ragi', 'kimchi', 'banana', 'chips', 'cookies', 'biscuits'
    ]
    flavors = [
        'hot', 'spicy', 'classic', 'indian', 'chatak', 'yummy', 'cheese',
        'rock salt', 'vinegar', 'cream', 'onion'
    ]

    text_lower = ocr_text.lower()

    for brand in brands:
        if brand in text_lower:
            product_keywords.append(brand)
    for p_type in product_types:
        if p_type in text_lower:
            product_keywords.append(p_type)
    for flavor in flavors:
        if flavor in text_lower:
            product_keywords.append(flavor)

    return list(set(product_keywords))


def smart_product_matcher(ocr_text, known_products):
    """Smart product matching using keyword extraction"""
    if not ocr_text:
        return "Unknown"

    text_lower = ocr_text.lower()
    keywords = extract_product_keywords(ocr_text)

    # Step 1: Keyword-based scoring
    if keywords:
        best_match = None
        best_score = 0

        for product in known_products:
            product_lower = product.lower()
            score = sum(3 for kw in keywords if kw in product_lower)
            if score > best_score:
                best_match = product
                best_score = score

        if best_match and best_score >= 3:
            return best_match

    # Step 2: Direct fallback matching
    direct_matches = {
        'cookie man': 'Cookie Man Choc Chunk Cookies',
        'choc filled': 'Cadbury Chocobakes Choc-Filled Cookies',
        'masala munch': 'Kurkure Masala Munch',
        'cream onion': "Lay's Cream & Onion",
        'sour cream onion': "Haldiram's Sour Cream & Onion",
        'tikka masala': 'Cornitos Nacho Crisps Tikka Masala',
        'rock salt': 'Kettle Studio Rock Salt & English Vinegar',
        'kimchi': 'Urban Platter Kimchi Potato Chips',
        'ragi': 'Chef Urbano Ragi Chips Indian Masala',
        'digestive': 'Britannia NutriChoice Digestive High Fibre Biscuits',
        'oats almonds': 'Sunfeast Farmlite Oats & Almonds Cookies',
        'cashew almond': "Sunfeast Mom's Magic Cashew Almond Biscuits",
        'protein cookie': 'Beyond Food Ultimate Protein Cookies Chocolate'
    }

    for pattern, product in direct_matches.items():
        if pattern in text_lower:
            return product

    return "Unknown"


# ---------------- FLASK-INTEGRATED FUNCTION ---------------- #

def process_single_image(image_path):
    """Used by Flask app to extract product name from a single image"""
    if not os.path.exists(SNACK_DB):
        raise FileNotFoundError("Snack database not found.")

    snack_db = pd.read_csv(SNACK_DB)
    known_products = snack_db['Product Name'].tolist()

    try:
        ocr_text = adaptive_ocr(image_path)
        product_name = smart_product_matcher(ocr_text, known_products)
        return {
            "ocr_text": ocr_text,
            "product_name": product_name
        }
    except Exception as e:
        print(f"❌ OCR error on {image_path}: {e}")
        return {
            "ocr_text": "",
            "product_name": "Unknown"
        }
