import streamlit as st

st.title('👩‍🎓 MSDS545Project')

st.write('Welcome to our machine learning model building app')

import pandas as pd

# ---------------------------------------------------
# Streamlit page config
# ---------------------------------------------------
st.set_page_config(
    page_title="Protein Sequence Dataset Explorer",
    layout="wide"
)

# ---------------------------------------------------
# 1. GitHub RAW URLs for your CSV files
#    Update these if your repo/path is different
# ---------------------------------------------------
POS_URL = "https://raw.githubusercontent.com/alydhicks/Protein-Files/main/positive_protein_sequences.csv"
NEG_URL = "https://raw.githubusercontent.com/alydhicks/Protein-Files/main/negative_protein_sequences.csv"


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
            "Please check your CSV column names (e.g., 'sequence', 'protein_sequence')."
        )

    # ---------------------------------------------------
    # Placeholder for future modeling
    # ---------------------------------------------------
    st.markdown("---")
    st.markdown("### Next Step: Modeling / Prediction")
    st.markdown(
        "Now that the data is loading correctly, you can plug `all_df` into a model "
        "(e.g., train/test split, feature extraction from sequences, etc.)."
    )


if __name__ == "__main__":
    main() 
