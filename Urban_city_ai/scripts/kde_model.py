import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde


def build_kde(filepath):
    df = pd.read_csv(filepath)

    # =========================
    # STEP 1: Handle clustering (Safecity issue)
    # =========================
    df_grouped = df.groupby(['latitude', 'longitude']).size().reset_index(name='count')

    # =========================
    # STEP 2: Reduce dominance of repeated coords
    # =========================
    df_grouped['weight'] = np.log1p(df_grouped['count'])

    coords = np.vstack([df_grouped['longitude'], df_grouped['latitude']])
    weights = df_grouped['weight']

    # =========================
    # STEP 3: KDE (balanced bandwidth)
    # =========================
    kde = gaussian_kde(coords, weights=weights, bw_method=0.03)

    return kde, df


def normalize_kde(kde, df):
    # =========================
    # STEP 4: Build reference grid
    # =========================
    lons = np.linspace(df['longitude'].min(), df['longitude'].max(), 200)
    lats = np.linspace(df['latitude'].min(), df['latitude'].max(), 200)

    grid = np.array([
        [kde([lon, lat])[0] for lon in lons]
        for lat in lats
    ])

    # =========================
    # STEP 5: Robust normalization bounds
    # =========================
    # avoids extreme spikes dominating everything
    p_low = np.percentile(grid, 5)
    p_high = np.percentile(grid, 95)

    def normalized(point):
        val = kde(point)[0]

        # clamp extremes
        val = max(p_low, min(val, p_high))

        # scale to 0–1
        return (val - p_low) / (p_high - p_low + 1e-9)

    return normalized


def test_kde(kde_norm):
    points = [
        [77.2, 28.6],
        [77.0, 28.5],
        [77.3, 28.7]
    ]

    print("\nNormalized KDE values:")
    for p in points:
        print(p, "→", kde_norm(p))


if __name__ == "__main__":
    kde, df = build_kde("../data/raw/safecity_full.csv")
    kde_norm = normalize_kde(kde, df)

    print("KDE + normalization ready")

    test_kde(kde_norm)