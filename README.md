🥗 Smart Food Label Guidance System

Scan. Verify. Choose Better.
A smartphone-based food label analysis system that uses OCR and NLP to extract nutrition information, verify FSSAI compliance, detect hidden allergens and expiry risks, and recommend healthier product alternatives — all in real time.

🚨 The Problem

Food labels today are confusing — small fonts, complex nutritional details, and hidden allergens make it nearly impossible for the average consumer to make informed decisions. There is also no simple way to verify FSSAI compliance or compare products for healthier options.

💡 Our Solution

Scan → OCR + NLP → Compliance Check → "Good to Go" or "Better Option Available"

⚙️ How It Works


Capture — User scans food label via mobile camera

Preprocessing — De-noising, skew correction, ROI detection

OCR Engine — Tesseract / PaddleOCR extracts expiry, nutrition, allergens

NLP Layer — Regex + Named Entity Recognition parses ingredients

Compliance Engine — Rule-based FSSAI validation

Recommendation System — Suggests safer and healthier alternatives



🏗️ Tech Stack


Python
Tesseract / PaddleOCR
Regex + NER (NLP)
Random Forest (ML Model)
NumPy, Pandas


📊 Sample Output

python{

  'CURRENT_PRODUCT': "LAY'S CLASSIC SALTED",
  
  'CURRENT_SCORE': 44.8,
  
  'CURRENT_GRADE': 'B',
  
  'RECOMMENDED_PRODUCT': "LAY'S AMERICAN STYLE CREAM & ONION",
  
  'RECOMMENDED_SCORE': 1.4,
  
  'RECOMMENDED_GRADE': 'A',
  
  'CATEGORY': 'POTATO CHIPS',
  
  'CALORIES_REDUCTION': 31,
  
  'FAT_REDUCTION': 8
  
}

🚀 Future Scope


Barcode / QR code scanning

Personalized diet alerts

Integration with e-commerce platforms

Expansion to FDA and EU food regulations

