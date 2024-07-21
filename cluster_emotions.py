import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


if not os.path.exists('./result.csv'):
    print('Result dataset is missing, run the prepare_dataset.py script first to generate the dataset')

else:
    dataset = pd.read_csv('./result.csv')

    # Get the valence-arousal coordinates
    coordinates = dataset[['valence', 'arousal']].copy()

    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(coordinates)

    # Initialize the KMeans model
    kmeans = KMeans(n_clusters=12, random_state=42)

    # Fit the model
    kmeans.fit(scaled_df)

    # Get the cluster labels
    labels = kmeans.labels_

    # Add the cluster labels to the DataFrame
    coordinates['cluster'] = labels

    # Plot the clusters
    plt.scatter(coordinates['valence'], coordinates['arousal'], c=coordinates['cluster'], cmap='viridis')
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.title('Emotions Clustering')
    plt.savefig('./resources/emotion_clusters.png')
    plt.show()
    plt.close()
