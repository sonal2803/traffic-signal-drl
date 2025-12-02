import pandas as pd
import numpy as np
import os

def process_ngsim(file_path, grid_cells=50):
    print(f"Reading from: {file_path}")

    if not os.path.exists(file_path):
        print("File not found. Check the path.")
        return

    df = pd.read_csv(file_path)

    # Create Y-position bins (road segments)
    y_min, y_max = df['Local_Y'].min(), df['Local_Y'].max()
    bins = np.linspace(y_min, y_max, grid_cells + 1)
    df['Grid_Segment'] = pd.cut(df['Local_Y'], bins=bins, labels=False, include_lowest=True)

    # Aggregate density per Frame_ID and Grid_Segment
    density_per_frame = df.groupby(['Frame_ID', 'Grid_Segment']).size().unstack(fill_value=0)

    # Save processed data to SEAI project/data/ngsim_density.csv
    output_path = os.path.join(os.path.dirname(__file__), 'data', 'ngsim_density.csv')
    density_per_frame.to_csv(output_path)

    print(f"✅ Density processing complete! Output saved to: {output_path}")

if __name__ == '__main__':
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data_20250318.csv')
    process_ngsim(file_path)
