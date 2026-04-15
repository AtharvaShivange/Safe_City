import osmnx as ox
import networkx as nx
import numpy as np
from shapely.geometry import LineString
import matplotlib.pyplot as plt

from kde_model import build_kde, normalize_kde


# =========================
# STEP 1: Load Road Graph
# =========================
print("Loading road network...")
G = ox.graph_from_place("New Delhi, India", network_type='drive')


# =========================
# STEP 2: Load KDE
# =========================
print("Building KDE...")
kde, df = build_kde("../data/raw/safecity_full.csv")
kde_norm = normalize_kde(kde, df)


# =========================
# STEP 3: Sampling Function
# =========================
def sample_points(line, num_points=10):
    return [line.interpolate(i / num_points, normalized=True) for i in range(num_points)]


# =========================
# STEP 4: Compute Road Risk (FIXED)
# =========================
def compute_road_risk(line, length):
    points = sample_points(line)

    risks = []
    for p in points:
        lon, lat = p.x, p.y
        risks.append(kde_norm([lon, lat]))

    avg_risk = np.mean(risks)

    # 🔥 exposure-based risk
    return avg_risk * length


# =========================
# STEP 5: Assign Risk + Weight (FIXED)
# =========================
print("Assigning risk to edges...")

lambda_ = 2  # balanced trade-off

for u, v, data in G.edges(data=True):
    if 'geometry' in data:
        line = data['geometry']
    else:
        point_u = (G.nodes[u]['x'], G.nodes[u]['y'])
        point_v = (G.nodes[v]['x'], G.nodes[v]['y'])
        line = LineString([point_u, point_v])

    length = data.get('length', 1)

    risk = compute_road_risk(line, length)

    data['risk'] = risk
    data['weight'] = length + lambda_ * risk

print("Risk assignment complete.")


# =========================
# STEP 6: Diagnostics
# =========================
edge_risks = [data['risk'] for u, v, data in G.edges(data=True)]

print("\n===== RISK DIAGNOSTICS =====")
print("Total edges:", len(edge_risks))

print("\nBasic stats:")
print("Mean risk:", np.mean(edge_risks))
print("Min risk:", np.min(edge_risks))
print("Max risk:", np.max(edge_risks))

print("\nPercentiles:")
print("25%:", np.percentile(edge_risks, 25))
print("50%:", np.percentile(edge_risks, 50))
print("75%:", np.percentile(edge_risks, 75))
print("95%:", np.percentile(edge_risks, 95))


# =========================
# STEP 7: Routing
# =========================
print("Running route computation...")

nodes = list(G.nodes())
source = nodes[0]
target = nodes[50]

short_route = nx.shortest_path(G, source, target, weight='length')
safe_route = nx.shortest_path(G, source, target, weight='weight')


# =========================
# STEP 8: Utility
# =========================
def compute_cost(route, weight_type):
    cost = 0
    for i in range(len(route) - 1):
        u = route[i]
        v = route[i + 1]
        edge_data = G.get_edge_data(u, v)[0]
        cost += edge_data.get(weight_type, 0)
    return cost


# =========================
# STEP 9: Compare Routes
# =========================
short_distance = compute_cost(short_route, 'length')
safe_distance = compute_cost(safe_route, 'length')

short_risk = compute_cost(short_route, 'risk')
safe_risk = compute_cost(safe_route, 'risk')

print("\n===== ROUTE COMPARISON =====")

print("\nShortest Route:")
print("Distance:", short_distance)
print("Risk:", short_risk)

print("\nSafest Route:")
print("Distance:", safe_distance)
print("Risk:", safe_risk)

print("\nRoute Length (nodes):")
print("Shortest:", len(short_route))
print("Safest:", len(safe_route))


# =========================
# STEP 10: Heatmap + Routes
# =========================
print("\nGenerating heatmap...")

lons = np.linspace(df['longitude'].min(), df['longitude'].max(), 300)
lats = np.linspace(df['latitude'].min(), df['latitude'].max(), 300)

X, Y = np.meshgrid(lons, lats)

points = np.vstack([X.ravel(), Y.ravel()])
Z = np.array([kde_norm(p) for p in points.T])
Z = Z.reshape(X.shape)

fig, ax = ox.plot_graph(
    G,
    show=False,
    close=False,
    node_size=0,
    edge_color='black',
    edge_alpha=0.3,
    edge_linewidth=0.5,
    bgcolor='white'
)

heatmap = ax.imshow(
    Z,
    extent=[lons.min(), lons.max(), lats.min(), lats.max()],
    origin='lower',
    cmap='inferno',
    alpha=0.6,
    interpolation='bilinear'
)

plt.colorbar(heatmap, ax=ax, label="Risk Level")

ox.plot_graph_route(
    G, short_route,
    ax=ax,
    show=False,
    close=False,
    route_color='blue',
    route_linewidth=3
)

ox.plot_graph_route(
    G, safe_route,
    ax=ax,
    show=False,
    close=False,
    route_color='red',
    route_linewidth=3
)

orig_node = short_route[0]
dest_node = short_route[-1]

ax.scatter(
    G.nodes[orig_node]['x'], G.nodes[orig_node]['y'],
    c='green', s=80, label='Start', zorder=5
)

ax.scatter(
    G.nodes[dest_node]['x'], G.nodes[dest_node]['y'],
    c='purple', s=80, label='End', zorder=5
)

plt.legend()
plt.title("Risk Heatmap + Shortest (Blue) vs Safest (Red)")
plt.show()