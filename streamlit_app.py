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
# Constants
# ---------------------------------------------------
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

POS_URL = "https://raw.githubusercontent.com/alydhicks/Protein-Files/main/positive_protein_sequences.csv"
NEG_URL = "https://raw.githubusercontent.com/alydhicks/Protein-Files/main/negative_protein_sequences.csv"

# Candidate column names for protein pairs
PAIR_CANDIDATES = [
    "protein_sequences_1",
    "protein_sequences_2",
    "seq1",
    "seq2",
    "protein1",
    "protein2",
]

# ---------------------------------------------------
# Streamlit page config
# ---------------------------------------------------
st.set_page_config(
    page_title="Protein Sequence Dataset Explorer",
    layout="wide"
)


# ---------------------------------------------------
# Utility: amino-acid frequency for a single sequence
# ---------------------------------------------------
def amino_acid_frequency(seq: str) -> pd.DataFrame:
    """
    Given a single protein sequence string, return a DataFrame
    with amino acid frequencies.
    """
    if seq is None:
        seq = ""
    seq = str(seq)

    counts = {aa: seq.count(aa) for aa in AMINO_ACIDS}
    total = sum(counts.values())

    freqs = {
        aa: (counts[aa] / total) if total > 0 else 0.0
        for aa in AMINO_ACIDS
    }

    df = pd.DataFrame(
        {
            "amino_acid": AMINO_ACIDS,
            "frequency": [freqs[aa] for aa in AMINO_ACIDS],
        }
    )
    return df


# ---------------------------------------------------
# Load data
# ---------------------------------------------------
@st.cache_data(show_spinner="Loading protein sequence data…")
def load_data(sample_size: int | None = None):
    """
    Load positive and negative protein CSVs from GitHub,
    add label column, and return combined DataFrame.
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
# Feature engineering for PPI pairs
# ---------------------------------------------------
def build_pair_features(df: pd.DataFrame, seq1_col: str, seq2_col: str) -> pd.DataFrame:
    """
    Build numerical features from a DataFrame containing two sequence columns.
    Features:
    - len1, len2, len_diff
    - s1_freq_AA, s2_freq_AA for each amino acid
    """
    df = df.copy()

    df["len1"] = df[seq1_col].astype(str).str.len()
    df["len2"] = df[seq2_col].astype(str).str.len()
    df["len_diff"] = df["len1"] - df["len2"]

    # Frequencies for each amino acid in both sequences
    for aa in AMINO_ACIDS:
        df[f"s1_freq_{aa}"] = df[seq1_col].astype(str).apply(
            lambda s: s.count(aa) / len(s) if len(s) > 0 else 0.0
        )
        df[f"s2_freq_{aa}"] = df[seq2_col].astype(str).apply(
            lambda s: s.count(aa) / len(s) if len(s) > 0 else 0.0
        )

    return df


# ---------------------------------------------------
# Main app
# ---------------------------------------------------
def main():
    st.title("👩‍🎓 MSDS545Project – Protein Interaction Explorer")
    st.write("Welcome to our machine learning app for predicting protein–protein interactions.")

    # Sidebar controls
    st.sidebar.header("Data & Model Options")
    sample_size = st.sidebar.number_input(
        "Sample N rows from combined dataset (0 = use all rows)",
        min_value=0,
        step=1000,
        value=0,
        help="Sampling can keep training fast for large datasets.",
    )
    sample_size = sample_size if sample_size > 0 else None

    # Load data
    try:
        pos_df, neg_df, all_df = load_data(sample_size)
    except Exception as e:
        st.error("❌ Error loading CSV files from GitHub.")
        st.write("- Check URLs")
        st.write("- Make sure repo and files are public")
        st.write("- Verify filenames and paths")
        st.exception(e)
        return

    st.success(
        f"Loaded {len(pos_df):,} positive and {len(neg_df):,} negative examples "
        f"({len(all_df):,} total rows)."
    )

    # Raw data viewer
    with st.expander("📊 View raw data"):
        option = st.selectbox(
            "Choose which dataset to view:",
            ["Positive sequences", "Negative sequences", "Combined (all)"],
        )

        if option == "Positive sequences":
            st.write("**Positive sequences (label = 1)**")
            st.dataframe(pos_df)
        elif option == "Negative sequences":
            st.write("**Negative sequences (label = 0)**")
            st.dataframe(neg_df)
        else:
            st.write("**Combined dataset**")
            st.dataframe(all_df)

    # Detect sequence pair columns
    available = [c for c in PAIR_CANDIDATES if c in all_df.columns]
    if len(available) >= 2:
        seq1_col, seq2_col = available[:2]
    else:
        # Fall back: let the user choose two string columns
        string_cols = [c for c in all_df.columns if all_df[c].dtype == "object"]
        if len(string_cols) < 2:
            st.error(
                "Could not automatically detect two sequence columns. "
                "Please ensure your CSVs contain two sequence columns "
                "(e.g., 'protein_sequences_1' and 'protein_sequences_2')."
            )
            return

        st.sidebar.subheader("Sequence Columns")
        seq1_col = st.sidebar.selectbox("Sequence 1 column", options=string_cols, index=0)
        seq2_col = st.sidebar.selectbox("Sequence 2 column", options=string_cols, index=1)

    st.info(f"Using `{seq1_col}` and `{seq2_col}` as the two protein sequence columns.")

    # Tabs for organization
    tab1, tab2, tab3 = st.tabs(
        ["📈 Dataset & EDA", "🧠 Model Performance", "🔮 Predict New Interaction"]
    )

    # ---------------------------------------------------
    # Tab 1: Dataset & simple EDA
    # ---------------------------------------------------
    with tab1:
        st.subheader("Class Distribution")
        class_counts = all_df["label"].value_counts().rename(
            {1: "Positive (interacting)", 0: "Negative (non-interacting)"}
        )
        st.write(class_counts)

        # Sequence length stats
        all_df["len1"] = all_df[seq1_col].astype(str).str.len()
        all_df["len2"] = all_df[seq2_col].astype(str).str.len()

        st.subheader(f"Sequence Length Statistics – `{seq1_col}`")
        st.write(all_df["len1"].describe())
        st.bar_chart(
            all_df["len1"]
            .value_counts()
            .sort_index()
            .head(100)
        )

        st.subheader(f"Sequence Length Statistics – `{seq2_col}`")
        st.write(all_df["len2"].describe())
        st.bar_chart(
            all_df["len2"]
            .value_counts()
            .sort_index()
            .head(100)
        )

    # ---------------------------------------------------
    # Tab 2: Model performance
    # ---------------------------------------------------
    with tab2:
        st.subheader("Train Random Forest Model")

        # Build features
        feat_df = build_pair_features(all_df, seq1_col, seq2_col)
        numeric_cols = feat_df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c != "label"]

        X = feat_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = feat_df["label"]

        test_size = st.slider("Test size (fraction)", 0.1, 0.5, 0.2, step=0.05)
        random_state = 42

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=random_state,
        )
        model.fit(X_train, y_train)

        st.success("🎉 Model training complete!")

        # Evaluation
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Classification Report**")
            st.text(classification_report(y_test, y_pred))

        with colB:
            st.markdown("**Confusion Matrix**")
            st.write(confusion_matrix(y_test, y_pred))
            st.markdown("**ROC-AUC Score:**")
            st.write(roc_auc_score(y_test, y_proba))

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        st.subheader("ROC Curve")
        st.line_chart(roc_data.set_index("FPR"))

        # Feature importance
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame(
            {"feature": feature_cols, "importance": model.feature_importances_}
        ).sort_values(by="importance", ascending=False)

        st.dataframe(importance_df.head(30))

        st.bar_chart(
            importance_df.set_index("feature")["importance"].head(30)
        )

        # Save model & feature_cols in session_state for use in Tab 3
        st.session_state["trained_model"] = model
        st.session_state["feature_cols"] = feature_cols
        st.session_state["seq1_col"] = seq1_col
        st.session_state["seq2_col"] = seq2_col

    # ---------------------------------------------------
    # Tab 3: Predict new interaction from two sequences
    # ---------------------------------------------------
    with tab3:
        st.subheader("Predict Interaction Between Two New Protein Sequences")

        seq1_input = st.text_area(
            "Protein 1 sequence",
            height=120,
            help="Enter the amino-acid sequence for protein 1 (e.g., 'MKTFFV...')",
        )
        seq2_input = st.text_area(
            "Protein 2 sequence",
            height=120,
            help="Enter the amino-acid sequence for protein 2.",
        )

        if st.button("Predict interaction"):
            if "trained_model" not in st.session_state:
                st.error(
                    "Model is not trained yet. Please go to the **Model Performance** tab first."
                )
            elif not seq1_input or not seq2_input:
                st.error("Please enter both sequences before predicting.")
            else:
                model = st.session_state["trained_model"]
                feature_cols = st.session_state["feature_cols"]

                # Build a one-row DataFrame matching training schema
                new_df = pd.DataFrame(
                    {
                        seq1_col: [seq1_input],
                        seq2_col: [seq2_input],
                        "label": [0],  # dummy label
                    }
                )

                new_feat = build_pair_features(new_df, seq1_col, seq2_col)
                X_new = (
                    new_feat[feature_cols]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0)
                )

                prob = model.predict_proba(X_new)[0, 1]
                pred_label = model.predict(X_new)[0]

                st.markdown(
                    f"### 🧾 Predicted probability of interaction: **{prob:.3f}**"
                )
                st.markdown(
                    f"Predicted class: **{int(pred_label)}** "
                    "(1 = interacting, 0 = non-interacting)"
                )

                # Amino acid frequency plots for each input sequence
                st.markdown("---")
                st.subheader("📈 Amino Acid Frequency for Input Sequences")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Protein 1 amino acid frequency**")
                    freq1_df = amino_acid_frequency(seq1_input)
                    st.bar_chart(freq1_df.set_index("amino_acid")["frequency"])

                with col2:
                    st.markdown("**Protein 2 amino acid frequency**")
                    freq2_df = amino_acid_frequency(seq2_input)
                    st.bar_chart(freq2_df.set_index("amino_acid")["frequency"])


if __name__ == "__main__":
    main()

