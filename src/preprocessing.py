import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['Unnamed: 0', 'Unnamed: 5', 'Unnamed: 6'], errors='ignore')
    df.dropna(subset=['emotion', 'empathetic_dialogues'], inplace=True)
    return df

def encode_emotions(df):
    df['emotion_label'] = df['emotion'].astype('category').cat.codes
    return df
