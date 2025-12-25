import pandas as pd

# Read files
lossC = 10.0
A = pd.read_csv(f"D:/Pose/SODEF NLP/A5000 exps/from wandb/lossC={lossC}/advglue.csv")
B = pd.read_csv(f"D:/Pose/SODEF NLP/A5000 exps/from wandb/lossC={lossC}/test.csv")
C = pd.read_csv(f"D:/Pose/SODEF NLP/A5000 exps/from wandb/lossC={lossC}/eigval.csv")

# Check step column consistency
if not (A.iloc[:, 0].equals(B.iloc[:, 0]) and A.iloc[:, 0].equals(C.iloc[:, 0])):
    raise ValueError("Step columns do not match")

# Step column
merged = A.iloc[:, [0]].copy()
merged.columns = ["step"]

# Function to extract first column of each 6-column experiment block
def extract_experiment_columns(df):
    exp_cols = {}
    cols = df.columns[1:]  # exclude step column

    for i in range(0, len(cols), 6):
        col_name = cols[i]
        exp_name = col_name.split("/")[0]  # extract experiment name
        exp_cols[exp_name] = df[col_name]

    return exp_cols

# Extract experiment data
A_exps = extract_experiment_columns(A)
B_exps = extract_experiment_columns(B)
C_exps = extract_experiment_columns(C)

# Merge x, y, z per experiment
for exp in A_exps.keys():
    merged[f"{exp}_advglue"] = A_exps[exp]
    merged[f"{exp}_test"] = B_exps[exp]
    merged[f"{exp}_eigval"] = C_exps[exp]

# # Save result
# merged.to_csv("merged.csv", index=False)

# print("Merged CSV saved as merged.csv")


# Identify experiments automatically
experiments = sorted(
    {col.rsplit("_", 1)[0] for col in merged.columns if col.endswith("_advglue")}
)

rows = []

for exp in experiments:
    x_col = f"{exp}_advglue"
    y_col = f"{exp}_test"
    z_col = f"{exp}_eigval"

    # Index of maximum x
    idx = merged[x_col].idxmax()

    rows.append({
        "experiment": exp,
        "step": merged.loc[idx, "step"],
        "advglue": merged.loc[idx, x_col],
        "test": merged.loc[idx, y_col],
        "eigval": merged.loc[idx, z_col],
    })

# Create result DataFrame
summary = pd.DataFrame(rows)

# Save to CSV
summary.to_csv(f"{lossC} max_x_per_experiment.csv", index=False)

print("Saved max_x_per_experiment.csv")