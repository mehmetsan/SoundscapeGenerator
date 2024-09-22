import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('result.csv')

# Sample code assuming you have a pandas DataFrame df
plt.figure(figsize=(10, 6))

# Scatterplot with 'valence' on x-axis, 'arousal' on y-axis, colored by 'emotion'
sns.scatterplot(
    data=df,
    x='valence',
    y='arousal',
    hue='emotion',
    palette='Set1',  # You can choose a color palette or leave it for default
    s=100,  # You can adjust the size of the points
    alpha=0.7  # Transparency level for better visibility if points overlap
)

plt.title('Valence vs Arousal with Emotion Coloring')
plt.xlabel('Valence')
plt.ylabel('Arousal')
plt.legend(title='Emotion')
plt.grid(True)
plt.savefig('./resources/emotions_painted.png')
plt.show()
