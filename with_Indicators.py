 import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from datetime import datetime
import random
import gc  # Garbage collection for memory management
from sklearn.cluster import DBSCAN # Import DBSCAN for clustering
import itertools
from sklearn.neighbors import NearestNeighbors

# ---------- 1. Load and prepare data ----------
print("Loading StockBean data...")

# Memory-efficient loading with chunking
chunk_size = 500000  # Adjust based on available memory
chunks = []

try:
    for chunk in pd.read_csv("/Users/lee/Project_Folder/Stockz_Project/Data/Input/StockBean_Mar3_1123 copy 2.csv", chunksize=chunk_size):
        # Keep only necessary columns to save memory
        essential_cols = ['Ticker', 'Date', 'Points_0', 'Points_1', 'Points_2', 'Close', 'Volume', 'Adj_Volume']
        chunk = chunk[[col for col in essential_cols if col in chunk.columns]]
        chunks.append(chunk)
    
    stockbean_df = pd.concat(chunks)
    del chunks
    gc.collect()  # Free up memory

except FileNotFoundError:
    print("Error: The data file '/Users/lee/Project_Folder/Stockz_Project/Data/Input/StockBean_Mar3_1123 copy 2.csv' was not found.")
    print("Please ensure the data file is in the correct directory and try again.")
    exit()


# Convert date column to datetime
stockbean_df['Date'] = pd.to_datetime(stockbean_df['Date'])

# Sort by date to ensure proper trajectory sequence
stockbean_df = stockbean_df.sort_values(['Ticker', 'Date'])

# Determine volume column
if 'Adj_Volume' in stockbean_df.columns and stockbean_df['Adj_Volume'].notna().any():
    volume_col = 'Adj_Volume'
else:
    volume_col = 'Volume'

# ---------- 2. Print some basic info about the data ----------
print(f"Data shape: {stockbean_df.shape}")
print(f"Date range: {stockbean_df['Date'].min()} to {stockbean_df['Date'].max()}")
print(f"Number of unique tickers: {stockbean_df['Ticker'].nunique()}")

# ---------- 3. Sample the data if needed (keeping original logic) ----------
sample_size = 1000000 # Reduced for performance with clustering
use_full_dataset = False 

if not use_full_dataset and len(stockbean_df) > sample_size:
    print(f"Sampling {sample_size} points from {len(stockbean_df)} total points for visualization...")
    
    # Using a simplified sampling method for speed
    # We will sample tickers, and then points within those tickers
    unique_tickers_list = stockbean_df['Ticker'].unique()
    num_tickers_to_sample = min(500, len(unique_tickers_list)) # Limit tickers for performance
    sampled_tickers = np.random.choice(unique_tickers_list, num_tickers_to_sample, replace=False)

    sampled_df = stockbean_df[stockbean_df['Ticker'].isin(sampled_tickers)]

    if len(sampled_df) > sample_size:
        # If still too large, sample points from the selected tickers
        stockbean_df = sampled_df.sample(n=sample_size, random_state=42)
    else:
        stockbean_df = sampled_df

    print(f"Sampled data shape: {stockbean_df.shape}")
    print(f"Number of unique tickers in sample: {stockbean_df['Ticker'].nunique()}")

# Ensure data is sorted for trajectory creation
stockbean_df = stockbean_df.sort_values(['Ticker', 'Date'])


# ---------- 4. NEW: Arrow Generation Logic ----------
print("\nGenerating data for directional arrows...")

# --- 1) Build segments with timestamp --------------------------------------
segments = []
for ticker, grp in stockbean_df.groupby('Ticker'):
    grp = grp.sort_values('Date')
    pts = grp[['Points_0','Points_1','Points_2']].values
    dates = grp['Date'].values
    for i in range(len(pts)-1):
        p1, p2 = pts[i], pts[i+1]
        if not np.array_equal(p1,p2):
            segments.append({
                'ticker':   ticker,
                'midpoint': (p1 + p2) / 2,
                'direction': p2 - p1,
                'date':     dates[i+1]       # timestamp of the "head" point
            })

if not segments:
    print("Could not generate any line segments. Check data integrity.")
else:
    segments_df = pd.DataFrame(segments)

    # --- 2) Sample 100 chronologically spaced arrows per ticker ----------------
    arrow_data = {}
    for ticker, segs in segments_df.groupby('ticker'):
        segs = segs.sort_values('date')
        N = len(segs)
        # pick indices 0, N/99, 2N/99, …, N-1
        idx = np.linspace(0, N-1, min(N,100), dtype=int)
        mids = np.vstack(segs['midpoint'].values)[idx]
        dirs = np.vstack(segs['direction'].values)[idx]
        # normalize directions
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms[norms==0] = 1
        arrow_data[ticker] = {
            'mids': mids,
            'dirs': (dirs / norms)
        }

    print(f"Generated chronologically spaced arrows for {len(arrow_data)} tickers.")

# ---------- 5. Create a visualization with trajectories AND arrows ----------
print("Creating ticker trajectory visualization...")

output_dir = "stockbean_visualizations_with_arrows"
os.makedirs(output_dir, exist_ok=True)

unique_tickers = stockbean_df['Ticker'].unique()
print(f"Processing {len(unique_tickers)} unique tickers...")

# --- Better color map: cycle through a big palette -------------------------
all_colors = (
    px.colors.qualitative.Plotly +
    px.colors.qualitative.D3 +
    px.colors.qualitative.Light24
)
color_cycle = itertools.cycle(all_colors)
color_map = {ticker: next(color_cycle) for ticker in unique_tickers}

# Use a single figure for simplicity; batching is complex with pre-calculated arrows
fig = go.Figure()

# --- 3) Plot cones + connecting line with gradient + start/stop markers ----
if arrow_data:
    print(f"Adding chronologically spaced arrows with gradient lines for {len(arrow_data)} tickers...")
    for ticker, data in arrow_data.items():
        mids, dirs = data['mids'], data['dirs']
        # cones
        fig.add_trace(go.Cone(
            x=mids[:,0], y=mids[:,1], z=mids[:,2],
            u=dirs[:,0],  v=dirs[:,1],  w=dirs[:,2],
            sizemode='absolute', sizeref=0.5, anchor='tail',
            showscale=False,
            name=ticker,
            legendgroup=ticker,
            showlegend=False
        ))
        # connecting line (gradient)
        colors = np.linspace(0,1,len(mids))
        fig.add_trace(go.Scatter3d(
            x=mids[:,0], y=mids[:,1], z=mids[:,2],
            mode='lines',
            line=dict(
                width=4,
                colorscale='Viridis',
                color=colors,
                cmin=0, cmax=1
            ),
            showlegend=False,
            legendgroup=ticker
        ))
        # start indicator (green)
        fig.add_trace(go.Scatter3d(
            x=[mids[0,0]], y=[mids[0,1]], z=[mids[0,2]],
            mode='markers',
            marker=dict(size=8, color='green'),
            showlegend=False,
            legendgroup=ticker
        ))
        # end indicator (red)
        fig.add_trace(go.Scatter3d(
            x=[mids[-1,0]], y=[mids[-1,1]], z=[mids[-1,2]],
            mode='markers',
            marker=dict(size=8, color='red'),
            showlegend=False,
            legendgroup=ticker
        ))

# --- 4) Add trajectories and layout as before -----------------------------
for ticker in unique_tickers:
    df_t = stockbean_df[stockbean_df['Ticker']==ticker]
    if len(df_t)>1:
        fig.add_trace(go.Scatter3d(
            x=df_t['Points_0'], y=df_t['Points_1'], z=df_t['Points_2'],
            mode='lines+markers',
            name=ticker,
            legendgroup=ticker,
            line=dict(width=2, color=color_map[ticker]),
            marker=dict(size=2, color=color_map[ticker]),
            opacity=0.6,
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "Date: %{customdata[0]}<br>" +
                "Close: $%{customdata[1]:.2f}<br>" +
                "<extra></extra>"
            ),
            text=df_t['Ticker'],
            customdata=np.stack((
                df_t['Date'].dt.strftime("%Y-%m-%d"),
                df_t['Close'].astype(float)
            ), axis=-1)
        ))


# ---------- 6. Finalize Layout and Save ----------
print("Finalizing plot layout...")

fig.update_layout(
    title="StockBean 3D Embedding with Chrono-Connected Arrows and Start/Stop Indicators",
    scene=dict(
        xaxis_title="Point 0",
        yaxis_title="Point 1",
        zaxis_title="Point 2",
        aspectmode='data' # 'data' aspect ratio is better for seeing clusters
    ),
    height=900,
    width=1200,
    template="plotly_dark",
    legend_title="Tickers"
)

# Save the visualization
output_file = os.path.join(output_dir, "StockBean_trajectories_with_indicators.html")
fig.write_html(output_file, include_plotlyjs='cdn')
print(f"\n✓ Visualization saved successfully!")
print(f"Open this file in your browser: {output_file}")