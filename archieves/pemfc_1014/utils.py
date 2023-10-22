import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def correct_abnormal_values_for_feature(data, feature, segment_size=800, threshold_pct=2, window_size=60):
    corrected_data = data.copy()
    num_segments = len(data) // segment_size

    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size
        segment_mean = data[feature][start_idx:end_idx].mean()

        # Identify abnormal values in the segment
        upper_threshold = segment_mean * (1 + threshold_pct / 100)
        lower_threshold = segment_mean * (1 - threshold_pct / 100)
        abnormal_indices = corrected_data.loc[start_idx:end_idx][(corrected_data[feature] > upper_threshold) |
                                                                 (corrected_data[feature] < lower_threshold)].index

        # Correct the abnormal values using moving average
        for idx in abnormal_indices:
            if idx > window_size:
                moving_avg = corrected_data[feature][idx - window_size:idx].mean()
                corrected_data.at[idx, feature] = moving_avg

    return corrected_data

def data_clean(data, process_data=True):

    if not process_data:
        return data

    corrected_data = data.copy()

    for feature in data.columns:
        corrected_data = correct_abnormal_values_for_feature(corrected_data, feature)

    return corrected_data


def plot_corrected_data(process_data=True):
    # Load the data
    A1 = pd.read_csv('/content/FC2_Ageing_part1.csv', encoding='ISO-8859-1')
    A2 = pd.read_csv('/content/FC2_Ageing_part2.csv', encoding='ISO-8859-1')
    data = pd.concat([A1, A2], ignore_index=True)

    if not process_data:
        return data

    corrected_data = data.copy()

    for feature in data.columns:
        corrected_data = correct_abnormal_values_for_feature(corrected_data, feature)

    # Plotting for the 'Utot (V)' column as an example
    plt.figure(figsize=(15, 6))
    plt.plot(data['Time (h)'], data['Utot (V)'], 'b', label='Original Utot (V)')
    plt.plot(corrected_data['Time (h)'], corrected_data['Utot (V)'], 'r', label='Corrected Utot (V)')
    plt.xlabel('Time(h)')
    plt.ylabel('Stack Voltage(V)')
    plt.title('Utot (V) Correction using Segment-Based Dynamic Threshold (Optimized)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return corrected_data


# Example Usage
# processed_data = plot_corrected_data(process_data=True)