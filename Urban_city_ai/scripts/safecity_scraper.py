import requests
import numpy as np
import time
import pandas as pd

url = "https://webapp.safecity.in/api/reported-incidents/map-coordinates"

headers = {
    "User-Agent": "Mozilla/5.0"
}

# Delhi bounds
LAT_MIN, LAT_MAX = 28.4, 28.9
LON_MIN, LON_MAX = 76.9, 77.4

STEP = 0.02 

# storage vectors
all_data = []
seen_ids = set()

print("Starting FULL Delhi scraping...\n")

lat_vals = np.arange(LAT_MIN, LAT_MAX, STEP)
lon_vals = np.arange(LON_MIN, LON_MAX, STEP)

total_requests = 0

# Scraping
for lat in lat_vals:
    for lon in lon_vals:

        payload = {
            "lang_id": 1,
            "client_id": 1,
            "city": "New Delhi",
            "map_zoom": 16,

            "map_bound[ne][lat]": lat + STEP,
            "map_bound[ne][lng]": lon + STEP,

            "map_bound[sw][lat]": lat,
            "map_bound[sw][lng]": lon,

            "map_bound[nw][lat]": lat + STEP,
            "map_bound[nw][lng]": lon,

            "map_bound[se][lat]": lat,
            "map_bound[se][lng]": lon + STEP,
        }

        try:
            response = requests.post(url, data=payload, headers=headers)
            data = response.json()

            points = data.get("data", [])

            new_count = 0

            for point in points:
                if point["id"] not in seen_ids:
                    seen_ids.add(point["id"])
                    all_data.append(point)
                    new_count += 1

            print(f"Grid ({lat:.2f}, {lon:.2f}) → {len(points)} fetched | {new_count} new")

        except Exception as e:
            print(f"Error at grid ({lat:.2f}, {lon:.2f}): {e}")

        total_requests += 1

        # checkpoint
        if total_requests % 20 == 0:
            df_temp = pd.DataFrame(all_data)
            df_temp.to_csv("../data/raw/safecity_checkpoint.csv", index=False)
            print(f"Checkpoint saved ({len(df_temp)} rows)\n")

        
        time.sleep(0.5)


df = pd.DataFrame(all_data)

output_path = "../data/raw/safecity_full.csv"
df.to_csv(output_path, index=False)

print(f"Total unique incidents: {len(df)}")
print(f"Saved to: {output_path}")
