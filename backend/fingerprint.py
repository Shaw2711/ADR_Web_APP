import pubchempy as pcp
from openbabel import openbabel


MANUAL_CIDS = {
    "IVERMECTIN": 6321424,
    "POLYTHIAZIDE": 4829,
    "VERTEPORFIN": 5362422
}


def get_smiles(drug_name):

    try:
        result = pcp.get_compounds(
            drug_name,
            "name"
        )

        if result:
            return result[0].connectivity_smiles

    except:
        pass

    if drug_name.upper() in MANUAL_CIDS:

        cid = MANUAL_CIDS[
            drug_name.upper()
        ]

        result = pcp.get_compounds(
            cid,
            "cid"
        )

        if result:
            return result[0].connectivity_smiles

    return None


def smiles_to_fp2(smiles):

    conv = openbabel.OBConversion()
    conv.SetInFormat("smi")

    mol = openbabel.OBMol()

    if not conv.ReadString(
        mol,
        smiles
    ):
        return None

    fp_type = openbabel.OBFingerprint.FindFingerprint(
        "FP2"
    )

    vec = openbabel.vectorUnsignedInt()

    fp_type.GetFingerprint(
        mol,
        vec
    )

    bits = []

    for word in vec:
        for bit in range(32):
            bits.append(
                1 if (word >> bit) & 1 else 0
            )

    return bits[:1024]


def generate_fingerprint(drug):

    smiles = get_smiles(drug)

    if smiles is None:
        return None, None

    fp = smiles_to_fp2(smiles)

    return fp, smiles