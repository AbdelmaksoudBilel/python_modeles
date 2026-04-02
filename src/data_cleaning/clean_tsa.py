import pandas as pd
import os

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"

def load_datasets():
    files = [f for f in os.listdir(RAW_PATH) if f.endswith(".csv")]
    dfs = []

    for file in files:
        file_path = os.path.join(RAW_PATH, file)

        if file.endswith(".csv"):
            df = pd.read_csv(file_path)

        dfs.append(df)

    return dfs


def clean_qchat(df):

    # Uniformiser yes/no
    df = df.replace({
        "yes": 1,
        "no": 0,
        "Yes": 1,
        "No": 0
    })

    # Supprimer valeurs manquantes
    df = df.dropna()

    # Supprimer doublons
    df = df.drop_duplicates()

    return df


def standardize_dataset(df):

    # Renommer A10 si nécessaire
    if "A10_Autism_Spectrum_Quotient" in df.columns:
        df = df.rename(columns={"A10_Autism_Spectrum_Quotient": "A10"})

    # Harmoniser Age
    if "Age_Mons" in df.columns:
        df["Age_Years"] = df["Age_Mons"] / 12

    # Harmoniser target
    if "ASD_traits" in df.columns:
        df = df.rename(columns={"ASD_traits": "target"})
    if "Class/ASD Traits " in df.columns:
        df = df.rename(columns={"Class/ASD Traits ": "target"})

    print("Class/ASD Traits" in df.columns)
    print(df.columns)
    # Colonnes finales à garder
    needed_cols = [
        "A1","A2","A3","A4","A5",
        "A6","A7","A8","A9","A10",
        "Age_Years","Sex",
        "Jaundice","Family_mem_with_ASD",
        "target"
    ]

    df = df[[col for col in needed_cols if col in df.columns]]
    df["Sex"] = df["Sex"].str.lower().str.strip()

    return df


def main():
    dfs = load_datasets()
    standardized_dfs = [standardize_dataset(df) for df in dfs]
    cleaned_dfs = [clean_qchat(df) for df in standardized_dfs]

    # Concaténation
    final_df = pd.concat(cleaned_dfs, axis=0)

    print(final_df.A1.value_counts())
    # Supprimer doublons globaux
    final_df = final_df.drop_duplicates()

    print(final_df.A1.value_counts())

    os.makedirs(PROCESSED_PATH, exist_ok=True)
    final_df.to_csv(os.path.join(PROCESSED_PATH, "qchat_cleaned.csv"), index=False)

    print("Dataset nettoyé et sauvegardé.")


if __name__ == "__main__":
    main()