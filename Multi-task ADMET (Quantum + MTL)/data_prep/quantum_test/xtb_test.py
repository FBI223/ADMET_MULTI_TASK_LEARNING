import os
import subprocess
import re
from rdkit import Chem
from rdkit.Chem import AllChem


# =========================
# SMILES → XYZ
# =========================
def smiles_to_xyz(smiles, fname):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    mol = Chem.AddHs(mol)

    if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
        return False

    AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()

    with open(fname, "w") as f:
        f.write(f"{mol.GetNumAtoms()}\n\n")
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")

    return True


# =========================
# PARSER (FIXED)
# =========================
def parse_xtb(output):
    energy = None
    gap = None
    dipole = None
    electrons = None

    lines = output.splitlines()

    for i, line in enumerate(lines):
        l = line.lower()

        # energy
        if "total energy" in l and "eh" in l:
            energy = float(re.findall(r"-?\d+\.\d+", line)[0])

        # gap
        if "homo-lumo gap" in l:
            gap = float(re.findall(r"-?\d+\.\d+", line)[0])

        # electrons
        if "# electrons" in l:
            electrons = int(re.findall(r"\d+", line)[0])

        # ===== DIPLOE FIX =====
        if "molecular dipole" in l:
            print(">>> FOUND DIPOLE BLOCK")
            print(lines[i])
            print(lines[i + 1] if i + 1 < len(lines) else "")
            print(lines[i + 2] if i + 2 < len(lines) else "")

            if i + 2 < len(lines):
                nums = re.findall(r"-?\d+\.\d+", lines[i + 2])
                if len(nums) >= 3:
                    x, y, z = map(float, nums[:3])
                    dipole = (x ** 2 + y ** 2 + z ** 2) ** 0.5

    # fallback (extra debug)
    if dipole is None:
        print("!!! Dipole NOT FOUND in structured block → trying fallback")
        m = re.search(r"dipole.*?([\d]+\.\d+)\s*debye", output, re.IGNORECASE | re.DOTALL)
        if m:
            dipole = float(m.group(1))

    return {
        "energy": energy,
        "gap": gap,
        "dipole": dipole,
        "electrons": electrons
    }


# =========================
# xTB PIPELINE
# =========================
def run_xtb_pipeline(xyz_file, smi):
    print("\n========================================")
    print(f"SMILES: {smi}")
    print("========================================")

    # --- STEP 1: OPT ---
    subprocess.run(
        ["xtb", xyz_file, "--gfn", "2", "--opt"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # --- STEP 2: SP ---
    result = subprocess.run(
        ["xtb", "xtbopt.xyz", "--gfn", "2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    output = result.stdout

    # ===== DEBUG: pokaż fragment dipole =====
    print("\n--- DEBUG SEARCH 'dipole' ---")
    for line in output.splitlines():
        if "dipole" in line.lower():
            print(line)

    # parsowanie
    data = parse_xtb(output)

    print("\n--- FINAL PARSED ---")
    print(f"ENERGY: {data['energy']}")
    print(f"GAP: {data['gap']}")
    print(f"DIPOLE: {data['dipole']}")
    print(f"ELECTRONS: {data['electrons']}")

    return data


# =========================
# PIPELINE
# =========================
def process(smiles_list):
    results = []

    for i, smi in enumerate(smiles_list):
        xyz = f"mol_{i}.xyz"


        ok = smiles_to_xyz(smi, xyz)
        if not ok:
            print("RDKit fail:", smi)
            continue

        data = run_xtb_pipeline(xyz, smi)
        results.append((smi, data))

    return results


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    smiles_list = [
        "CC1=CC=C(C=C1)C(=O)O",
        "CN1CCC(CC1)C2=CN=CC=C2",
        "CCOC(=O)C1=CC=CC=C1Cl",
        "CCN(CC)CCOC(=O)C1=CC=CC=C1",
        "COC1=CC=C(C=C1)C=O"
    ]

    results = process(smiles_list)

    print("\nFINAL:")
    for smi, data in results:
        print(smi, data)