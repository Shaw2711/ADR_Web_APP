import pubchempy as pcp
import requests


def get_pubchem_info(drug_name):

    try:
        c = pcp.get_compounds(
            drug_name,
            "name"
        )[0]

        return {
            "formula": c.molecular_formula,
            "weight": c.molecular_weight,
            "smiles": c.canonical_smiles
        }

    except:
        return {}


def get_side_effects(drug_name):

    try:
        url = (
            "https://api.fda.gov/drug/event.json?"
            f"search=patient.drug.medicinalproduct:{drug_name}"
            "&limit=5"
        )

        r = requests.get(url)

        return r.json()

    except:
        return {}


def get_drug_info(drug_name):

    try:
        # PubChem properties
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/MolecularFormula,MolecularWeight,IUPACName/JSON"

        r = requests.get(url)

        if r.status_code != 200:
            return {}

        data = r.json()

        props = data["PropertyTable"]["Properties"][0]

        return {
            "molecular_formula":
                props.get("MolecularFormula"),

            "molecular_weight":
                props.get("MolecularWeight"),

            "iupac_name":
                props.get("IUPACName"),

            "structure_image":
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/PNG"
        }

    except:
        return {}


def get_target_info(drug_name):

    try:
        url = f"https://mychem.info/v1/query?q={drug_name}"

        r = requests.get(url)

        if r.status_code != 200:
            return []

        data = r.json()

        hits = data.get("hits", [])

        targets = []

        for h in hits:

            if "drugbank" in h:

                db = h["drugbank"]

                if "targets" in db:

                    for t in db["targets"]:

                        targets.append({
                            "gene":
                                t.get("gene_name"),

                            "protein":
                                t.get("name"),

                            "organism":
                                t.get("organism")
                        })

        return targets

    except:
        return []