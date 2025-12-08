import streamlit as st
import pandas as pd

# ---------------------------------------------------
# Streamlit page config
# ---------------------------------------------------
st.set_page_config(
    page_title="Protein Sequence Dataset Explorer",
    layout="wide"
)

st.title('👩‍🎓 MSDS545Project')
st.write('Welcome to our machine learning model building app') 

# ---------------------------------------------------
# 1. GitHub RAW URLs for your CSV files
#    Update these if your repo/path is different
# ---------------------------------------------------
POS_URL = "https://raw.githubusercontent.com/alydhicks/Protein-Files/main/positive_protein_sequences.csv"
NEG_URL = "https://raw.githubusercontent.com/alydhicks/Protein-Files/main/negative_protein_sequences.csv"
with st.expander("data"):
    st.write("rawdata")
# ---------------------------------------------------
# 2. Data loading function (with optional sampling)
# ---------------------------------------------------
@st.cache_data(show_spinner="Loading protein sequence data…")
def load_data(sample_size=None):
    """
    Load positive and negative protein CSVs from GitHub.
    Adds a 'label' column and optionally returns a random sample.
    """
    # Read both CSVs from GitHub
    pos_df = pd.read_csv(POS_URL)
    neg_df = pd.read_csv(NEG_URL)

    # Add labels: 1 = positive, 0 = negative
    pos_df["label"] = 1
    neg_df["label"] = 0

    # Combine into one DataFrame
    all_df = pd.concat([pos_df, neg_df], ignore_index=True)

    # Optional sampling to avoid memory issues with very large data
    if sample_size is not None and sample_size > 0:
        sample_size = min(sample_size, len(all_df))
        all_df = all_df.sample(n=sample_size, random_state=42)

    return pos_df, neg_df, all_df


# ---------------------------------------------------
# 3. Main Streamlit app
# ---------------------------------------------------
def main():
    st.title("Protein Sequence Dataset Explorer")

    st.markdown(
        """
        This app loads **positive** and **negative** protein sequence datasets
        directly from GitHub and prepares them for downstream analysis or modeling.
        """
    )

    # Sidebar controls
    st.sidebar.header("Data Loading Options")

    sample_size = st.sidebar.number_input(
        "Sample N rows from combined dataset (0 = use all rows)",
        min_value=0,
        step=1000,
        value=0,
        help="If the dataset is very large, sampling can help keep things fast and memory-friendly."
    )
    sample_size = sample_size if sample_size > 0 else None

    # Try loading data
    try:
        pos_df, neg_df, all_df = load_data(sample_size=sample_size)
    except Exception as e:
        st.error("❌ There was an error loading the CSV files from GitHub.")
        st.write("Please check that:")
        st.write("- The URLs are correct")
        st.write("- The repo and files are public")
        st.write("- The filenames and paths match exactly")
        st.exception(e)
        return

    # Summary
    st.success(
        f"Loaded {len(pos_df):,} positive and {len(neg_df):,} negative sequences "
        f"({len(all_df):,} rows in the working dataset)."
    )

    # ---------------------------------------------------
    # Data previews
    # ---------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Positive Sequences (label = 1)")
        st.dataframe(pos_df.head())

    with col2:
        st.subheader("Negative Sequences (label = 0)")
        st.dataframe(neg_df.head())

    st.subheader("Combined Dataset (with 'label' column)")
    st.dataframe(all_df.head())

    # ---------------------------------------------------
    # Class distribution
    # ---------------------------------------------------
    st.subheader("Class Distribution")
    class_counts = all_df["label"].value_counts().rename({1: "Positive", 0: "Negative"})
    st.write(class_counts)

    # ---------------------------------------------------
    # Sequence length statistics (if we can find a sequence column)
    # ---------------------------------------------------
    seq_col_candidates = ["sequence", "Sequence", "protein_sequence", "seq"]
    seq_col = None
    for col in seq_col_candidates:
        if col in all_df.columns:
            seq_col = col
            break

    if seq_col is not None:
        st.subheader(f"Sequence Length Statistics (using column: `{seq_col}`)")
        all_df["sequence_length"] = all_df[seq_col].astype(str).str.len()

        st.write(all_df["sequence_length"].describe())

        st.bar_chart(
            all_df["sequence_length"]
            .value_counts()
            .sort_index()
            .head(100)  # limit to avoid huge charts
        )
    else:
        st.info(
            "I couldn't automatically find a sequence column. "
            "Please check your CSV column names (e.g., 'sequence', 'protein_sequence_1')."
        )

    # ---------------------------------------------------
    # Placeholder for future modeling
   # ---------------------------------------------------
    # Sequence length statistics (if we can find a sequence column)
    # ---------------------------------------------------
    seq_col_candidates = ["sequence", "Sequence", "protein_sequence_1", "seq"]
    seq_col = None
    for col in seq_col_candidates:
        if col in all_df.columns:
            seq_col = col
            break
    # Let the user choose which column to use as the sequence for modeling
    string_cols = [c for c in all_df.columns if all_df[c].dtype == "object"]

    st.sidebar.subheader("Sequence Column for Modeling")
    seq_col = st.sidebar.selectbox(
        "Select the sequence column to use for modeling:",
         options=string_cols,
         index=0 if "protein_sequences_1" in string_cols else 0,
     )
    

    # ---------------------------------------------------
    # 4. Modeling & Prediction Section
    # ---------------------------------------------------
    # 4. Modeling & Prediction Section
    # ---------------------------------------------------
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        roc_curve
    )

    st.markdown("---")
    st.header("🔬 Modeling & Prediction")

    if seq_col is None:
        st.warning("⚠️ Cannot proceed with modeling because no sequence column was detected.")
    else:
        st.subheader("Feature Engineering")

        # ---- Feature Engineering ----
        def extract_features(df):
            """
            Convert protein sequences into simple numeric features:
            - Sequence length
            - Amino acid frequency (20 standard AAs)
            """
            df = df.copy()
            df["length"] = df[seq_col].astype(str).str.len()
            
            # Amino acid set
            amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
            for aa in amino_acids:
                df[f"freq_{aa}"] = df[seq_col].astype(str).apply(
                    lambda s: s.count(aa) / len(s) if len(s) > 0 else 0
                )

            return df

        # Extract features
        feat_df = extract_features(all_df)

        # ✅ 1) Keep ONLY numeric columns as features
        numeric_cols = feat_df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c != "label"]

        # Debug view (optional)
        # st.write("Feature columns:", feature_cols)
        # st.write(feat_df[feature_cols].dtypes)

        X = feat_df[feature_cols]
        y = feat_df["label"]

        # ✅ 2) Clean NaN / inf values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        # ---- Train/Test Split ----
        st.subheader("Train/Test Split")
        test_size = st.slider("Test size (fraction)", 0.1, 0.5, 0.2, step=0.05)
        random_state = 42

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        st.write(f"Training rows: {len(X_train)}")
        st.write(f"Testing rows: {len(X_test)}")

        # ---- Model Training ----
        st.subheader("Train Model")
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=42
        )
        model.fit(X_train, y_train)

        st.success("🎉 Model training complete!")

        # ---- Evaluation ----
        st.subheader("Model Evaluation")

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        colA, colB = st.columns(2)

        with colA:
            st.write("**Classification Report**")
            st.text(classification_report(y_test, y_pred))

        with colB:
            st.write("**Confusion Matrix**")
            st.write(confusion_matrix(y_test, y_pred))
            st.write("**ROC-AUC Score:**", roc_auc_score(y_test, y_proba))

        # ---- ROC Curve ----
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        st.line_chart(roc_data.set_index("FPR"))

        # ---- Feature Importance ----
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        st.dataframe(importance_df.head(20))
        st.bar_chart(importance_df.set_index("feature").head(20))

        # ---------------------------------------------------
        #  Generate Prediction File for Kaggle
        # ---------------------------------------------------
        st.subheader("📄 Generate Kaggle Submission File")

        if st.button("Create Kaggle Submission CSV"):
            preds_df = pd.DataFrame({
                "sequence": all_df[seq_col],
                "prediction": model.predict(X)  # X is feat_df[feature_cols] cleaned
            })

            csv_data = preds_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Kaggle Predictions CSV",
                data=csv_data,
                file_name="kaggle_ppi_predictions.csv",
                mime="text/csv"
            )
            st.success("Your Kaggle submission file is ready!")
            
       # ---------------------------------------------------
        #  Exploratory Boxplots for Feature Distributions
        # ---------------------------------------------------
        import seaborn as sns
        import matplotlib.pyplot as plt

        st.subheader("Feature Distributions by Label")

        # Start from the numeric feature columns
        cols = feature_cols.copy()

        # Safely remove columns if they exist
        for c in ["protein_sequences_1", "protein_sequences_2", "PPI"]:
            if c in cols:
                cols.remove(c)

        # Create a boxplot for each feature vs label
        for col in cols:
            fig, ax = plt.subplots()
            sns.boxplot(x="label", y=col, data=feat_df, ax=ax)
            ax.set_title(f"{col} vs Label")
            ax.set_xlabel("Label (0 = Negative, 1 = Positive)")
            ax.set_ylabel(col)
            st.pyplot(fig)
            plt.close(fig)

if __name__ == "__main__":
    main()
