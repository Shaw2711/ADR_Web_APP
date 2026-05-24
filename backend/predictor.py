import numpy as np
from similarity import compute_similarity, filter_by_threshold
from fingerprint import generate_fingerprint
from drug_info import get_drug_info, get_target_info

class VKRPredictor:

    def __init__(
        self,
        dgi_train,
        fp_train,
        adr_train,
        model,
        drug_names,
        side_effect_names
    ):
        self.dgi_train = dgi_train
        self.fp_train = fp_train
        self.adr_train = adr_train
        self.model = model
        self.drug_names = drug_names
        self.side_effect_names = side_effect_names

    def run(
        self,
        drug_name,
        threshold=0.7
    ):
        info = get_drug_info(drug_name)
        targets = get_target_info(drug_name)
        fp, smiles = generate_fingerprint(
            drug_name
        )

        if fp is None:
            return {
                "error": "Fingerprint not generated"
            }

        fp = np.array(fp, dtype=float)

        sims = compute_similarity(
            fp,
            self.fp_train
        )

        selected_idx = filter_by_threshold(
            sims,
            threshold
        )

        if len(selected_idx) == 0:
            return {
                "error":
                "No similar drugs found"
            }


        similar_drugs = [
            self.drug_names[i]
            for i in selected_idx
        ]


        dgi_input = np.mean(
            self.dgi_train[selected_idx],
            axis=0
        )


        preds = self.model.predict(
            dgi_input,
            fp
        )


        top_idx = np.argsort(
            preds
        )[::-1][:20]

        top_side_effects = [
            self.side_effect_names[i]
            for i in top_idx
        ]

        return {
            "drug": drug_name,
            "smiles": smiles,

            "pubchem_url":
                f"https://pubchem.ncbi.nlm.nih.gov/#query={drug_name}",

            "similar_drugs":
                similar_drugs,

            "similar_scores":
                sims[selected_idx].tolist(),

            "predictions":
                preds.tolist(),

            "top_side_effects":
                top_side_effects,
            
            "true_labels":
                self.adr_train[
                self.drug_names.index(drug_name)
                ].tolist()
                if drug_name in self.drug_names
                else None,
            "structure_image": info.get("structure_image"),
            "molecular_formula": info.get("molecular_formula"),
            "molecular_weight": info.get("molecular_weight"),
            "iupac_name": info.get("iupac_name"),
            "targets": targets,
        }