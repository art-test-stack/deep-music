import pandas as pd
# wav and mp3 files

def preprocess(csv_file):
    df = pd.read_csv(csv_file)
    df['audio_path'] = df['audio_path'].str.replace('wav', 'mp3')
    df.to_csv('data.csv', index=False)
    return df