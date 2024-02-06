import pandas as pd
import numpy as np

all = pd.DataFrame()

def transform_1995():
    pd.read_csv('data/csv/1995_tour_1.csv')
    
    # On garde que les colonnes qui nous intéressent
    df = df.loc[df['Code du département'] == 33]

    