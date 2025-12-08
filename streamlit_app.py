import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------
# Global constants
# ---------------------------------------------------
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

POS_URL = "https://raw.githubusercontent.com/alydhicks/Protein-Files/main/positive_protein_sequences.csv"
NEG_URL = "https://raw.githubusercontent.com/alydhicks/Protein-Files/main/negative_protein_sequences.csv"


# ---------------------------------------------------
# 1. Data loading
# ---------------------------------------------------
@st.cache_data(show_spinner="Loading protein sequence data…")
def load_data(sample_size=None):
    """
    Load positive and negative protein CSVs from GitHub.
    Adds a 'label' column and optionally returns a random sample.
    """
    pos_df = pd.read_csv(POS_URL)
    neg_df = pd.read_csv(NEG_URL)

    pos_df["label"] = 1
    neg_df["label"] = 0

    all_df = pd.concat([pos_df, neg_df], ignore_index=True)

    if sample_size is not None and sample_size > 0:
        sample_size = min(sample_size, len(all_df))
        all_df = all_df.sample(n=sample_size, random_state=42)

    return pos_df, neg_df, all_df


# ---------------------------------------------------
# 2. Feature engineering for pairs
# ---------------------------------------------------
def featurize_pair_df(df: pd.DataFrame, seq1_col: str, seq2_col: str) -> pd.DataFrame:
    """
    Turn a DataFrame with two sequence columns into numeric features:
    - Length of each sequence
    - Amino acid frequencies for each sequence
    - Length difference
    """
    out = pd.DataFrame(index=df.index)

    def featurize_series(series: pd.Series, prefix: str):
        s = series.astype(str).fillna("")
        out[f"{prefix}len"] = s.str.len()
        for aa in AMINO_ACIDS:
            out[f"{prefix}freq_{aa}"] = s.apply(
                lambda x: x.count(aa) / len(x) if len(x) > 0 else 0.0
            )

    featurize_series(df[seq1_col], "s1_")
    featurize_series(df[seq2_col], "s2_")
    out["len_diff"] = out["s1_len"] - out["s2_len"]

    return out


def featurize_single_pair(
    seq1: str,
    seq2: str,
    seq1_col: str,
    seq2_col: str,
    feature_cols_template: pd.Index,
) -> pd.DataFrame:
    """
    Featurize a single (seq1, seq2) pair into the same feature space
    as the training data.
    """
    temp_df = pd.DataFrame({seq1_col: [seq1], seq2_col: [seq2]})
    feat_df = featurize_pair_df(temp_df, seq1_col, seq2_col)

    # Ensure same columns / order as training data
    feat_df = feat_df.reindex(columns=feature_cols_template, fill_value=0.0)
    return feat_df


# ---------------------------------------------------
# 3. Main Streamlit app
# ---------------------------------------------------
def main():
    st.set_page_config(
        page_title="Protein Sequence Dataset Explorer",
        layout="wide",
    )

    st.title("👩‍🎓 MSDS545Project – Protein–Protein Interaction Predictor")
    st.write(
        "This app loads protein–protein interaction data, trains a simple model, "
        "and lets you input two protein sequences to predict the likelihood of interaction."
    )

    # ---------------- Sidebar: data / model options ----------------
    st.sidebar.header("Data & Model Options")
    sample_size = st.sidebar.number_input(
        "Sample N rows from combined dataset (0 = use all rows)",
        min_value=0,
        step=1000,
        value=0,
        help="Sampling can help keep things fast and memory-friendly.",
    )
    sample_size = sample_size if sample_size > 0 else None

    # ---------------- Load data ----------------
    try:
        pos_df, neg_df, all_df = load_data(sample_size=sample_size)
    except Exception as e:
        st.error("❌ There was an error loading the CSV files from GitHub.")
        st.write("- Check that the URLs are correct")
        st.write("- Repo and files are public")
        st.write("- Filenames and paths match exactly")
        st.exception(e)
        return

    st.success(
        f"Loaded {len(pos_df):,} positive and {len(neg_df):,} negative pairs "
        f"({len(all_df):,} rows in the working dataset)."
    )

    # ---------------- Raw data viewer ----------------
    with st.expander("📊 View raw data"):
        option = st.selectbox(
            "Choose which dataset to view:",
            ["Positive pairs", "Negative pairs", "Combined (all)"],
            key="raw_selector",
        )

        if option == "Positive pairs":
            st.write("**Positive pairs (label = 1)**")
            st.dataframe(pos_df)
        elif option == "Negative pairs":
            st.write("**Negative pairs (label = 0)**")
            st.dataframe(neg_df)
        else:
            st.write("**Combined dataset**")
            st.dataframe(all_df)

    # ---------------- Basic class distribution ----------------
    st.subheader("Class Distribution")
    class_counts = all_df["label"].value_counts().rename({1: "Positive", 0: "Negative"})
    st.write(class_counts)

    # ---------------- Detect sequence columns ----------------
    st.subheader("Sequence Columns")
    pair_candidates = [
        "protein_sequences_1",
        "protein_sequences_2",
        "seq1",
        "seq2",
        "protein1",
        "protein2",
    ]
    available_seq_cols = [c for c in pair_candidates if c in all_df.columns]

    if len(available_seq_cols) < 2:
        st.error(
            "Could not automatically find two sequence columns.\n\n"
            "Make sure your CSVs contain something like "
            "`protein_sequences_1` and `protein_sequences_2`, or update "
            "the `pair_candidates` list in the code."
        )
        st.stop()

    seq1_col, seq2_col = available_seq_cols[:2]
    st.write(f"Using `{seq1_col}` as **sequence 1** and `{seq2_col}` as **sequence 2**.")

    # ---------------- Train simple model ----------------
    st.markdown("---")
    st.header("🔬 Train Simple PPI Model")

    # Feature matrix
    X = featurize_pair_df(all_df, seq1_col, seq2_col)
    y = all_df["label"]

    # Clean NaN / inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    st.write("Feature matrix shape:", X.shape)

    # Train–test split
    test_size = st.sidebar.slider(
        "Test size (fraction)", 0.1, 0.5, 0.2, step=0.05, key="test_size_slider"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    st.write(f"Training rows: {len(X_train)}, Testing rows: {len(X_test)}")

    # Model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    st.success("🎉 Model training complete!")

    # ---------------- Evaluation: KEEP these ----------------
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    st.subheader("ROC Curve & AUC")
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr}).set_index("FPR")
    st.write(f"ROC–AUC: **{auc:.3f}**")
    st.line_chart(roc_df)

    st.subheader("Feature Importance")
    importance_df = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values(by="importance", ascending=False)
    st.dataframe(importance_df.head(20))
    st.bar_chart(importance_df.set_index("feature").head(20))

    # ---------------- New pair prediction ----------------
    st.markdown("---")
    st.header("🧪 Predict Interaction for New Protein Pair")

    st.write(
        "Enter two protein sequences below. The model will compute simple amino-acid "
        "features and predict the probability that the two proteins interact."
    )

    # Provide defaults from first row so user sees the format
    default_s1 = str(all_df[seq1_col].iloc[0]) if len(all_df) > 0 else ""
    default_s2 = str(all_df[seq2_col].iloc[0]) if len(all_df) > 0 else ""

    seq1_input = st.text_area("Protein sequence 1", value=default_s1, height=120)
    seq2_input = st.text_area("Protein sequence 2", value=default_s2, height=120)

    if st.button("Predict interaction"):
        # Clean: keep only letters and uppercase them
        seq1_clean = "".join(ch for ch in seq1_input.upper() if ch.isalpha())
        seq2_clean = "".join(ch for ch in seq2_input.upper() if ch.isalpha())

        if len(seq1_clean) == 0 or len(seq2_clean) == 0:
            st.error("Please enter valid (non-empty) amino-acid sequences for BOTH proteins.")
        else:
            new_X = featurize_single_pair(
                seq1_clean,
                seq2_clean,
                seq1_col=seq1_col,
                seq2_col=seq2_col,
                feature_cols_template=X.columns,
            )
            new_X = new_X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            prob = model.predict_proba(new_X)[0, 1]
            pred_label = "Interacting (1)" if prob >= 0.5 else "Non-interacting (0)"

            st.success(
                f"**Predicted probability of interaction:** {prob:.3f}  \n"
                f"**Predicted class:** {pred_label}"
            )


if __name__ == "__main__":
    main()
