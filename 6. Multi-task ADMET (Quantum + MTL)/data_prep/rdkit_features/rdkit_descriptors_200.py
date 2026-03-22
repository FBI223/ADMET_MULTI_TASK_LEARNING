import pandas as pd
import os
from descriptastorus.descriptors import rdNormalizedDescriptors
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- CONFIG ---
N_CORES = 8
INPUT_FILE = "unique_smiles_to_calculate.csv"
OUTPUT_FILE = "rdkit_200_exact.csv"

# --- GENERATOR (Chemprop-compatible) ---
generator = rdNormalizedDescriptors.RDKit2DNormalized()

# --- NAZWY FEATURE (version-agnostic) ---
test = generator.process("CC")

if hasattr(generator, "columns"):
    FEATURE_NAMES = generator.columns
elif hasattr(generator, "GetDescriptorNames"):
    FEATURE_NAMES = generator.GetDescriptorNames()
else:
    FEATURE_NAMES = [f"f_{i}" for i in range(len(test) - 1)]

# usuń smiles jeśli jest
if FEATURE_NAMES[0] == "smiles":
    FEATURE_NAMES = FEATURE_NAMES[1:]

assert len(FEATURE_NAMES) == 200


def get_features(smiles):
    try:
        feats = generator.process(smiles)

        if feats is None or len(feats) != 201:
            return {"smiles": smiles, "success": False}

        values = feats[1:]

        res = {"smiles": smiles, "success": True}
        for name, val in zip(FEATURE_NAMES, values):
            res[name] = val

        return res

    except Exception:
        return {"smiles": smiles, "success": False}


if __name__ == "__main__":

    print("Features:", len(FEATURE_NAMES))  # musi być 200
    print(FEATURE_NAMES)
    if not os.path.exists(INPUT_FILE):
        print("Brak pliku")
        exit()

    df = pd.read_csv(INPUT_FILE)
    smiles_list = df["smiles"].dropna().tolist()

    results = []

    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        futures = {executor.submit(get_features, sm): sm for sm in smiles_list}

        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    df_res = pd.DataFrame(results)
    df_success = df_res[df_res["success"] == True].drop(columns=["success"])

    print("Kolumny:", len(df_success.columns))  # 201 = smiles + 200

    df_success.to_csv(OUTPUT_FILE, index=False)

    print("DONE:", OUTPUT_FILE)