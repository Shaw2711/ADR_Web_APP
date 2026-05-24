from fastapi import FastAPI
from pydantic import BaseModel

from data_loader import DataLoader
from predictor import VKRPredictor
from model import VKRNMFModel
from fingerprint import generate_fingerprint


app = FastAPI(
    title="ADR Prediction API"
)


loader = DataLoader()

dgi_train, fp_train, adr_train = loader.get_all()

drug_names = loader.fp_df.index.tolist()
side_effect_names = loader.adr_df.columns.tolist()


model = VKRNMFModel()

model.fit(
    dgi_train,
    fp_train,
    adr_train
) 

predictor = VKRPredictor(
    dgi_train=dgi_train,
    fp_train=fp_train,
    adr_train=adr_train,
    model=model,
    drug_names=drug_names,
    side_effect_names=side_effect_names
)



class DrugRequest(BaseModel):
    drug_name: str
    threshold: float = 0.7



@app.get("/")
def home():
    return {
        "message": "ADR Predictor Running"
    }


@app.post("/predict")
def predict(req: DrugRequest):

    result = predictor.run(
        req.drug_name,
        req.threshold
    )

    return result

@app.get("/drug_info/{drug_name}")
def drug_info(drug_name: str):

    fp, smiles = generate_fingerprint(drug_name)

    if fp is None:
        return {
            "error": "Drug not found"
        }

    # reuse predictor logic (optional but powerful)
    result = predictor.run(drug_name, threshold=0.7)

    return {
        "drug": drug_name,
        "smiles": smiles,
        "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/#query={drug_name}",
        "top_side_effects": result["top_side_effects"],
        "similar_drugs": result["similar_drugs"]
    }