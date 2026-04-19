import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- dane ---
corr = pd.read_csv("korelacje.csv", index_col=0)
results = pd.read_csv("final_results.csv")

tasks = corr.columns.tolist()

# --- rho ---
rho = {t: corr.loc[t].drop(t).mean() for t in tasks}

# --- delta ---
delta = {row["Task"]: row["MTL_GNN"] - row["STL_GNN"] for _, row in results.iterrows()}

# --- wektory ---
rho_vals = [rho[t] for t in tasks if t in delta]
delta_vals = [delta[t] for t in tasks if t in delta]
labels = [t for t in tasks if t in delta]

# --- plot ---
plt.figure(figsize=(6,6))
plt.scatter(rho_vals, delta_vals)

for i, t in enumerate(labels):
    plt.text(rho_vals[i], delta_vals[i], t, fontsize=8)

plt.axhline(0, linestyle="--")
plt.axvline(0, linestyle="--")

# --- trend ---
z = np.polyfit(rho_vals, delta_vals, 1)
p = np.poly1d(z)
plt.plot(rho_vals, p(rho_vals), linestyle="--")

plt.xlabel("rho")
plt.ylabel("delta")

# --- zapis zamiast show ---
plt.savefig("rho_vs_delta.png", dpi=300, bbox_inches="tight")
plt.close()