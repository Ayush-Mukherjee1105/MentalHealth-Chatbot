import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the log
log_file = "conversation_log.csv"
df = pd.read_csv(log_file)

# Parse timestamps
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date

# === 1. Emotion Frequency Bar Chart ===
def plot_emotion_counts():
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='emotion', order=df['emotion'].value_counts().index, palette='coolwarm')
    plt.title("Emotion Frequency")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("emotion_frequency.png")
    plt.close()

# === 2. Emotion Over Time ===
def plot_emotion_over_time():
    emotion_per_day = df.groupby(['date', 'emotion']).size().unstack(fill_value=0)
    emotion_per_day.plot(kind='line', figsize=(12, 6), marker='o')
    plt.title("Emotion Trends Over Time")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.legend(title="Emotion")
    plt.tight_layout()
    plt.savefig("emotion_trends.png")
    plt.close()

if __name__ == "__main__":
    plot_emotion_counts()
    plot_emotion_over_time()
    print("✅ Plots saved: emotion_frequency.png and emotion_trends.png")
