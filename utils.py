import pandas as pd
import matplotlib.pyplot as plt

def data_clean(x: pd.DataFrame) -> pd.DataFrame:
    """
    处理异常数据
    :param x: raw数据，包含异常值
    :return: 清洗好的数据，异常值被替换
    """

    # Data condense for every 9 min
    data_c = [x.iloc[0]]
    last_time = data_c[0]['Time']
    for _, row in x.iterrows():
        if row['Time'] - last_time >= 0.1:
            data_c.append(row)
            last_time = row['Time']

    data_c = pd.DataFrame(data_c)

    # Selecting Features
    variables = data_c[['Time', 'TinH2', 'TinAIR', 'ToutH2', 'TinWAT', 'I', 'PoutAIR', 'HrAIRFC', 'ToutAIR']]

    # Moving average filter
    windowSize = 10
    output = variables.copy()
    for column in variables.columns:
        output[column] = variables[column].rolling(window=windowSize).mean()

    return output

# Load data
A1 = pd.read_csv('D:/E/Sheffield/Spring/ITP/data/FC2_Ageing_part1.csv')
A2 = pd.read_csv('D:/E/Sheffield/Spring/ITP/data/FC2_Ageing_part2.csv')
data = pd.concat([A1, A2], ignore_index=True)

cleaned_data = data_clean(data)

# Plotting
plt.figure()
plt.plot(data['Time'], data['Utot'], 'b', label='original data')
plt.plot(cleaned_data['Time'], cleaned_data['Utot'], 'r', label='cleaned data')
plt.xlabel('Time(h)')
plt.ylabel('Stack Voltage(V)')
plt.title('quasi-Dynamic Dataset after Cleaning')
plt.legend()
plt.show()

# Plotting after Moving Average Filter (MAF)
plt.figure()
plt.plot(cleaned_data['Time'], cleaned_data['I'], 'b', label='original data')
plt.plot(cleaned_data['Time'], cleaned_data['I'], 'r', linewidth=1.5, label='MAF data')
plt.xlabel('Time(h)')
plt.ylabel('Stack Voltage(V)')
plt.title('Quasi-Dynamic MAF after Cleaning')
plt.legend()
plt.show()
