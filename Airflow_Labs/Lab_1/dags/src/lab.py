import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
import pickle
import os
import base64

# Mall dataset feature columns (3 numeric columns)
FEATURE_COLS = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    print("We are here")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    serialized_data = pickle.dumps(df)                         # bytes
    return base64.b64encode(serialized_data).decode("ascii")   # JSON-safe string


def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.

    NOTE:
    To keep your DAG unchanged, we return a single base64 string,
    but inside it we store a tuple: (scaled_data, fitted_scaler).
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()

    # Select Mall dataset columns
    clustering_data = df[FEATURE_COLS]

    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)

    # Store both (data, scaler) so downstream tasks can scale test.csv consistently
    payload = (clustering_data_minmax, min_max_scaler)

    clustering_serialized_data = pickle.dumps(payload)
    return base64.b64encode(clustering_serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Builds a Gaussian Mixture Model (GMM) on the preprocessed data and saves it.
    Returns a list of BIC values (JSON-serializable).

    NOTE:
    We keep the same function name/signature as your KMeans version.
    """
    data_bytes = base64.b64decode(data_b64)
    payload = pickle.loads(data_bytes)

    X_scaled, scaler = payload

    bics = []
    best_model = None
    best_bic = None
    best_k = None

    # Try k=1..10 (faster than 1..49, and enough for Mall dataset)
    for k in range(1, 11):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=42,
            n_init=5
        )
        gmm.fit(X_scaled)

        bic = float(gmm.bic(X_scaled))
        bics.append(bic)

        if (best_bic is None) or (bic < best_bic):
            best_bic = bic
            best_model = gmm
            best_k = k

    print(f"Best k (by min BIC): {best_k}")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Save model + scaler + feature cols together
    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "model": best_model,
                "scaler": scaler,
                "feature_cols": FEATURE_COLS,
                "best_k": best_k
            },
            f
        )

    return bics  # list is JSON-safe


def load_model_elbow(filename: str, bic_values: list):
    """
    Loads the saved model and reports best k (chosen by minimum BIC).
    Returns the first prediction (as a plain int) for test.csv.
    """
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    bundle = pickle.load(open(output_path, "rb"))

    loaded_model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]
    best_k = bundle.get("best_k", None)

    # Optional: show best k from saved file + confirm BIC list length
    print(f"Optimal no. of clusters (by BIC): {best_k}")
    print(f"BIC values received: {len(bic_values)}")

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))

    X_test = df[feature_cols]
    X_test_scaled = scaler.transform(X_test)

    pred = loaded_model.predict(X_test_scaled)[0]

    try:
        return int(pred)
    except Exception:
        return pred.item() if hasattr(pred, "item") else pred