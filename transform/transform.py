import pandas as pd
import os

def clean_election_data(year, num_candidates, input_path):
    df = pd.read_csv(input_path)
    df["annee"] = year

    cols_communes = [
        'Code du département', 'Libellé du département', 'Code de la circonscription',
        'Libellé de la circonscription', 'Code de la commune', 'Libellé de la commune',
        'Code du b.vote', 'Inscrits', 'Abstentions', '% Abs/Ins', 'Votants', '% Vot/Ins',
        'Blancs', '% Blancs/Ins', '% Blancs/Vot', 'Nuls', '% Nuls/Ins', '% Nuls/Vot',
        'Exprimés', '% Exp/Ins', '% Exp/Vot'
    ]

    # Adjust columns for years without 'Code de la circonscription' and 'Code du b.vote'
    if year in [1995, 2002, 2007, 2012, 2017]:
        cols_communes = [col for col in cols_communes if col not in ['Code de la circonscription', 'Libellé de la circonscription', 'Code du b.vote']]
        if year == 2017:
            cols_communes[cols_communes.index('Blancs')] = 'Blancs et nuls'
            cols_communes.remove('% Blancs/Ins')
            cols_communes.remove('% Blancs/Vot')

    cols_candidats = ['N°Panneau', 'Sexe', 'Nom', 'Prénom', 'Voix', '% Voix/Ins', '% Voix/Exp']

    data = []

    for i in range(num_candidates):
        if i == 0:
            cols_selection = cols_communes + cols_candidats
        else:
            cols_selection = cols_communes + [f'{col}.{i}' for col in cols_candidats]

        df_candidat = df[cols_selection].copy()
        if i != 0:  # Rename columns for candidates beyond the first
            df_candidat = df_candidat.rename(columns={f'{col}.{i}': col for col in cols_candidats})

        # Create a unique Candidat_ID by combining year and candidate index
        df_candidat['Candidat_ID'] = df_candidat.apply(lambda x: f"{year}_{i}", axis=1)
        data.append(df_candidat)

    df_final = pd.concat(data, ignore_index=True)
    return df_final

# List to hold data from all years
all_years_data = []

election_years = {
    1995: 9,
    2002: 16,
    2007: 12,
    2012: 10,
    2017: 11,
    2022: 12,
}

for year, num_candidates in election_years.items():
    input_path = f"./extract/csv/{year}_tour_1.csv"
    df_year = clean_election_data(year, num_candidates, input_path)
    all_years_data.append(df_year)

# Combine all years into a single DataFrame
df_all_years = pd.concat(all_years_data, ignore_index=True)

# Optionally, save the combined DataFrame to a CSV file
output_path = "./export/all_years_combined.csv"
df_all_years.to_csv(output_path, index=False)

print("Combined data saved to:", output_path)