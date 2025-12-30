from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import json

app = Flask(__name__)

MODEL_PATH = "heart_model.pkl"
MODEL_COLUMNS_NPY = "model_columns.npy"     # optional: numpy array of column names / order
MODEL_COLUMNS_PKL = "model_columns.pkl"     # optional: pickle with list of column names
# ----------------------------

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Train model and place it next to app.py")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Try to load saved feature order (optional but recommended)
feature_order = None
if os.path.exists(MODEL_COLUMNS_NPY):
    try:
        feature_order = list(np.load(MODEL_COLUMNS_NPY, allow_pickle=True))
    except Exception:
        feature_order = None

if feature_order is None and os.path.exists(MODEL_COLUMNS_PKL):
    try:
        with open(MODEL_COLUMNS_PKL, "rb") as f:
            feature_order = pickle.load(f)
    except Exception:
        feature_order = None

# Helper to get the expected number of features of the model
def model_expected_features():
    # scikit-learn estimators generally have `n_features_in_`
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    # fallback: try coef_ shape
    if hasattr(model, "coef_"):
        return int(model.coef_.shape[1])
    return None

# ---------- PAGES ----------
@app.route('/')
def welcome():
    return render_template("welcome.html")

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/dashboard', methods=['POST'])
def dashboard():
    name = request.form.get('name')
    email = request.form.get('email')
    phone = request.form.get('phone')
    address = request.form.get('address')

    if name and email and phone:
        return redirect(url_for('predict_form'))
    return "Please fill all details!"

@app.route('/predict_form')
def predict_form():
    return render_template("index.html")


# generate the user graph
def generate_user_graph(values, labels):
    os.makedirs("static", exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.bar(labels, values, color="skyblue")
    plt.xticks(rotation=70)
    plt.title("User Input Feature Values")
    plt.tight_layout()
    plt.savefig("static/user_graph.png")
    plt.close()


# ---------- Prediction route (handles 11 form fields) ----------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ------------- 1) Read inputs from form (11 features) -------------
        # NOTE: these names must match the `name` attributes in your index.html
        raw_inputs = {
            "age": request.form.get("age"),
            "sex": request.form.get("sex"),
            "cp": request.form.get("cp"),
            "trestbps": request.form.get("trestbps"),
            "chol": request.form.get("chol"),
            "fbs": request.form.get("fbs"),
            "restecg": request.form.get("restecg"),
            "thalach": request.form.get("thalach"),
            "exang": request.form.get("exang"),
            "oldpeak": request.form.get("oldpeak"),
            "slope": request.form.get("slope")
        }

        # Ensure all exist
        missing = [k for k, v in raw_inputs.items() if v is None or v == ""]
        if missing:
            return f"Error: missing form fields: {missing}"

        # Convert to float values (and handle typical string items)
        features_list = []
        for k in ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope"]:
            val = raw_inputs[k]
            try:
                features_list.append(float(val))
            except ValueError:
                # If user sent strings like 'M' or 'F' by mistake, return a clear error
                return f"Error: could not convert form value for '{k}' to number: '{val}'. Please ensure the form uses numeric values."

        # ------------- 2) Prepare input vector expected by the model -------------
        expected = model_expected_features()

        if feature_order:
            # If we have a saved feature order, build vector respecting that order
            # feature_order should be a list of strings, e.g. ["Age", "Sex", "CP", ...]
            # We'll try to map names from our form to that feature order.
            # Mapping heuristics (lowercase comparisons)
            map_form_to_feature = {
                "age": ["age"],
                "sex": ["sex"],
                "cp": ["cp","chestpain","chest_pain","chestpain_type"],
                "trestbps": ["trestbps","restingbp","resting_blood_pressure","restbp"],
                "chol": ["chol","cholesterol"],
                "fbs": ["fbs","fastingbs","fasting_blood_sugar"],
                "restecg": ["restecg","rest_ecg","restingecg"],
                "thalach": ["thalach","maxhr","max_heart_rate","thalach"],
                "exang": ["exang","exerciseangina","exercise_angina"],
                "oldpeak": ["oldpeak","st_depression","old_peak"],
                "slope": ["slope","st_slope","st-slope"]
            }

            # create a dict from lower-name -> value
            form_values = {
                "age": features_list[0],
                "sex": features_list[1],
                "cp": features_list[2],
                "trestbps": features_list[3],
                "chol": features_list[4],
                "fbs": features_list[5],
                "restecg": features_list[6],
                "thalach": features_list[7],
                "exang": features_list[8],
                "oldpeak": features_list[9],
                "slope": features_list[10]
            }

            # Build ordered vector
            x = []
            for col in feature_order:
                lower = col.strip().lower()
                placed = False
                for form_key, aliases in map_form_to_feature.items():
                    for alias in aliases:
                        if alias in lower:
                            x.append(form_values[form_key])
                            placed = True
                            break
                    if placed: break
                if not placed:
                    # Unknown column — fill with 0 (or you can choose np.nan)
                    x.append(0.0)

            x = np.array([x], dtype=float)
            # check dims
            if expected is not None and x.shape[1] != expected:
                return (f"Error: model expects {expected} features but feature file provided produced {x.shape[1]}. "
                        f"Please retrain model or supply matching feature order file.")
        else:
            # No feature_order saved:
            if expected is None or expected == len(features_list):
                # model expects same number as we provided
                x = np.array([features_list], dtype=float)
            elif expected > len(features_list):
                # Model expects more — we will pad zeros to the right and warn.
                # This is a fallback; best practice: retrain model for 11 features or save feature_order.
                pad_len = expected - len(features_list)
                x = np.array([features_list + [0.0] * pad_len], dtype=float)
            else:
                # Model expects fewer features than provided: drop extras (unlikely)
                x = np.array([features_list[:expected]], dtype=float)

        # ------------- 3) Predict -------------
        prediction = int(model.predict(x)[0])
        result_text = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        # ------------- 4) Generate graph for the 11 visible features -------------
        # We generate graph using form values (labels for display)
        labels = ["Age","Sex","CP","RestBP","Chol","FBS","RestECG","MaxHR","ExAngina","Oldpeak","ST Slope"]
        generate_user_graph(features_list, labels)

        return render_template("result.html", result=result_text, graph_path="static/user_graph.png")

    except Exception as e:
        # Return friendly error message (instead of raw sklearn traceback)
        return f"Error: {str(e)}"


@app.route('/graphs')
def graphs():
    if os.path.exists("static/user_graph.png"):
        return render_template("graphs.html", graph="static/user_graph.png")
    return "No graph available."


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)


