import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt



if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

if "selected_drug" not in st.session_state:
    st.session_state.selected_drug = None

if "selected_effect" not in st.session_state:
    st.session_state.selected_effect = None

if "view" not in st.session_state:
    st.session_state.view = "main"

API_URL = "https://unsuited-commerce-footsore.ngrok-free.dev/predict"

st.set_page_config(
    page_title="Drug Side Effect Predictor",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
.block-container{
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

div[data-testid="metric-container"]{
    border:1px solid #E5E7EB;
    border-radius:12px;
    padding:15px;
    box-shadow:0 2px 4px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)


st.title("🧬 Drug Side Effect Prediction Dashboard")
st.caption("VKR-NMF-based adverse drug reaction prediction and molecular analysis")

st.sidebar.header("Prediction Controls")

drug_name = st.sidebar.text_input(
    "Drug Name",
    placeholder="e.g. Ibuprofen"
)

threshold = st.sidebar.slider(
    "Similarity Threshold",
    0.0,
    1.01,
    0.70,
    0.01
)

top_n = 20

run_btn = st.sidebar.button("Explore")

if run_btn or st.session_state.prediction_data is not None:

    if not run_btn:
        data = st.session_state.prediction_data
    else:
        if not drug_name:
            st.warning("Please enter a drug name")
            st.stop()

        with st.spinner("Running full prediction pipeline..."):

            response = requests.post(
                API_URL,
                json={
                    "drug_name": drug_name,
                    "threshold": threshold
                }
            )

        if response.status_code != 200:
            st.error(response.text)
            st.stop()

        st.session_state.prediction_data = response.json()
        data = st.session_state.prediction_data
        
        st.session_state.prediction_data = data

    if "error" in data:
        st.error(data["error"])
        st.stop()

    pred = np.array(data["predictions"])
    sim_scores = np.array(data["similar_scores"])
    y_true = data["true_labels"]
    if y_true is not None:
        y_true = np.array(y_true)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Drug", st.session_state.prediction_data["drug"])
    c2.metric("Matched Drugs", len(data["similar_drugs"]))
    c3.metric("Threshold", threshold)
    c4.metric("Max Prediction", round(pred.max(), 4))

    st.divider()


    with st.expander("📄 Drug Information", expanded=True):
        st.write("**Drug Name:**", data["drug"])
        st.write("**SMILES:**")
        st.code(data["smiles"])

        if "pubchem_url" in data:
            st.markdown(
                f"[Open in PubChem]({data['pubchem_url']})"
            )

    if st.session_state.view == "drug_profile":

        selected = st.session_state.selected_drug

        st.title(f"Drug Profile: {selected}")

        info = requests.post(
            API_URL,
            json={
                "drug_name": selected,
                "threshold": threshold
            }
        ).json()

        st.subheader("SMILES")
        st.code(info["smiles"])

        st.subheader("Top Side Effects")
        st.write(info["top_side_effects"][:10])

        st.markdown(
            f"[Open in PubChem]({info['pubchem_url']})"
        )

        if st.button("⬅ Back"):
            st.session_state.view = "main"
            st.rerun()

        st.stop()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Similar Drugs",
        "Predicted Side Effects",
        "Threshold Dashboard",
        "Molecular Information",
        "Drug Side Effect Analysis System"
    ])


    with tab1:

        st.subheader("Most Similar Drugs")

        sim_df = pd.DataFrame({
            "Similar Drugs": data["similar_drugs"]
        })

        st.dataframe(
            sim_df,
            width="stretch"
        )

        selected = st.selectbox(
            "🔍 Search / Select a Similar Drug",
            options=["-- select --"] + data["similar_drugs"],
            key="drug_selector"
        )

        if selected != "-- select --":

            st.divider()
            st.subheader(f"Drug Profile: {selected}")

            info = requests.post(
                API_URL,
                json={
                    "drug_name": selected,
                    "threshold": threshold
                }
            ).json()

            c1, c2 = st.columns(2)

            with c1:
                st.write("**Drug Name:**", selected)

                st.write("**SMILES:**")
                st.code(info["smiles"])

                st.markdown(
                    f"[Open in PubChem]({info['pubchem_url']})"
                )

            with c2:
                st.write("**Top Side Effects**")

                se_df = pd.DataFrame({
                    "Side Effects":
                    info["top_side_effects"][:10]
                })

                st.dataframe(
                    se_df,
                    width="stretch"
                )

            st.subheader("Drug Structure")

            st.image(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{selected}/PNG",
                width=300
            )

    with tab2:

        st.subheader("Top Predicted Side Effects")

        side_df = pd.DataFrame({
            "Side Effect Name":
            data["top_side_effects"][:top_n]
        })

        st.dataframe(
            side_df,
            width="stretch"
        )

        csv = side_df.to_csv(index=False).encode()

        st.download_button(
            "⬇ Download Side Effects",
            csv,
            "predicted_side_effects.csv",
            "text/csv"
        )

        st.divider()

        st.subheader("⚠ Top 5 Priority Alerts")

        for effect in data["top_side_effects"][:5]:
            st.warning(effect)

        st.divider()

        st.subheader("Side Effects Word Cloud")

        text = " ".join(data["top_side_effects"])

        wordcloud = WordCloud(
            width=900,
            height=400,
            background_color="white",
            colormap="viridis"
        ).generate(text)

        fig = plt.figure(figsize=(12,5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.subheader("Threshold Impact")

        thresholds = np.arange(0.1, 1.01, 0.05)
        counts = [np.sum(sim_scores >= t) for t in thresholds]

        fig = plt.figure(figsize=(8,4))
        plt.plot(thresholds, counts, marker="o")
        plt.axvline(threshold, linestyle="--")
        plt.xlabel("Threshold")
        plt.ylabel("Matched Drugs")
        plt.title("Effect of Threshold on Similar Drugs")
        st.pyplot(fig)
        plt.close(fig)

    with tab4:

        st.subheader("Drug Chemical Profile")

        st.image(
            data["structure_image"],
            width=300
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Molecular Weight",
            data["molecular_weight"]
        )

        c2.metric(
            "Formula",
            data["molecular_formula"]
        )

        st.write(
            "**IUPAC Name:**",
            data["iupac_name"]
        )
    
    with tab5:

        st.subheader("Model Output Analysis")

        preds = np.array(data["predictions"])

        st.markdown("### Prediction Distribution")

        fig = plt.figure(figsize=(10, 4))
        plt.hist(preds, bins=30)
        plt.xlabel("Prediction Score")
        plt.ylabel("Frequency")
        plt.title("Distribution of ADR Prediction Scores")
        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        st.markdown("### Key Statistics")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Max Score", round(preds.max(), 4))
        c2.metric("Mean Score", round(preds.mean(), 4))
        c3.metric("Median Score", round(np.median(preds), 4))
        c4.metric("Std Dev", round(preds.std(), 4))

        st.divider()

        st.markdown("### Interpretation Guide")

        st.info("""
        • Higher scores indicate stronger predicted association with adverse drug reactions  
        • Wider distribution suggests broader side effect profile  
        • Narrow distribution suggests selective/low-risk prediction pattern  
        """)

else:
    st.info("Enter a drug and click Predict.")