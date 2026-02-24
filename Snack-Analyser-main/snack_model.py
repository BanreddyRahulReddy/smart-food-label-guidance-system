# health_model.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

MODEL_PATH = "health_model.pkl"

def detect_category(name: str) -> str:
    """Detect basic snack category."""
    if not name:
        return "Unknown"
    name = name.lower()
    chips = ["lay", "kurkure", "chips", "too yumm", "bingo", "cornitos"]
    biscuits = ["biscuit", "cookie", "unibic", "britannia", "sunfeast", "parle"]
    if any(k in name for k in chips):
        return "Chips"
    elif any(k in name for k in biscuits):
        return "Biscuits"
    else:
        return "Other"

def load_snack_db(path="snack_data.csv") -> pd.DataFrame:
    """Load snack data and compute basic health score."""
    df = pd.read_csv(path)
    df["Category"] = df["Product Name"].apply(detect_category)
    df["Health Score"] = df.apply(
        lambda r: (float(r.get("Protein (g)", 0))*2) -
                  (float(r.get("Sugar (g)", 0))*1.5) -
                  (float(r.get("Energy (kcal)", 0))/50), axis=1
    )
    return df

def train_health_model(snack_df: pd.DataFrame):
    """Train ML model to predict health score based on nutrition data."""
    features = ["Energy (kcal)", "Carbohydrates (g)", "Sugar (g)", "Protein (g)"]
    snack_df = snack_df.dropna(subset=features + ["Health Score"])
    X = snack_df[features]
    y = snack_df["Health Score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"Model trained. R² score: {score:.3f}")
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline

def load_health_model():
    """Load trained ML model, train if missing."""
    try:
        return joblib.load(MODEL_PATH)
    except:
        print("Model not found, training a new one...")
        df = load_snack_db()
        return train_health_model(df)

def get_health_score(product_name: str, db: pd.DataFrame):
    """Return predicted health score for a product."""
    model = load_health_model()
    features = ["Energy (kcal)", "Carbohydrates (g)", "Sugar (g)", "Protein (g)"]
    row = db[db['Product Name'].str.lower() == product_name.lower()]
    if row.empty:
        return None
    return round(float(model.predict(row[features])[0]), 2)

def recommend_alternative(product_name: str, db: pd.DataFrame):
    """Recommend a healthier snack alternative."""
    model = load_health_model()
    category = detect_category(product_name)
    subset = db[db["Category"] == category]
    if subset.empty:
        return None

    # Predict health scores
    features = ["Energy (kcal)", "Carbohydrates (g)", "Sugar (g)", "Protein (g)"]
    subset["Predicted Score"] = model.predict(subset[features])

    current_row = subset[subset["Product Name"].str.lower() == product_name.lower()]
    if current_row.empty:
        return None
    current_score = current_row["Predicted Score"].values[0]

    better = subset[subset["Predicted Score"] > current_score]
    if better.empty:
        return None

    best = better.sort_values(by="Predicted Score", ascending=False).iloc[0]
    reason = []
    if best["Sugar (g)"] < current_row["Sugar (g)"].values[0]:
        reason.append("less sugar")
    if best["Protein (g)"] > current_row["Protein (g)"].values[0]:
        reason.append("more protein")
    if best["Energy (kcal)"] < current_row["Energy (kcal)"].values[0]:
        reason.append("fewer calories")

    return {
        "category": category.title(),
        "recommended_product": best["Product Name"],
        "reason": ", ".join(reason),
        "predicted_health_gain": round(best["Predicted Score"] - current_score, 2)
    }

if __name__ == "__main__":
    df = load_snack_db("snack_data.csv")
    train_health_model(df)
