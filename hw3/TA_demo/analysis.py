import pandas as pd
import numpy as np
import os

# read the csv files
df1 = pd.read_csv('/home/yuehpo/coding/DeepMIR_2023fall/hw3/TA_demo/t0.8_p0.95_pop1k7.csv')
df2 = pd.read_csv('/home/yuehpo/coding/DeepMIR_2023fall/hw3/TA_demo/t1.2_p0.95_pop1k7.csv')
df3 = pd.read_csv('/home/yuehpo/coding/DeepMIR_2023fall/hw3/TA_demo/t1.2_p0.9_pop1k7.csv')

# group by piece name (50, 100, 400) or (50, 100, 390)
df1['piece_name'] = df1['piece_name'].apply(lambda x: x.split('_')[0])
df2['piece_name'] = df2['piece_name'].apply(lambda x: x.split('_')[0])
df3['piece_name'] = df3['piece_name'].apply(lambda x: x.split('_')[0])
print(f"df1: {df1}, df2: {df2}, df3: {df3}")


df1m = df1.groupby('piece_name').mean().reset_index()
df2m = df2.groupby('piece_name').mean().reset_index()
df3m = df3.groupby('piece_name').mean().reset_index()
df1s = df1.groupby('piece_name').std().reset_index()
df2s = df2.groupby('piece_name').std().reset_index()
df3s = df3.groupby('piece_name').std().reset_index()

# 3 digits
df1m = df1m.round(3)
df2m = df2m.round(3)
df3m = df3m.round(3)
df1s = df1s.round(3)
df2s = df2s.round(3)
df3s = df3s.round(3)

# make 'piece_name' float and sort 
df1m['piece_name'] = df1m['piece_name'].astype(float)
df1m = df1m.sort_values(by=['piece_name'])
df2m['piece_name'] = df2m['piece_name'].astype(float)
df2m = df2m.sort_values(by=['piece_name'])
df3m['piece_name'] = df3m['piece_name'].astype(float)
df3m = df3m.sort_values(by=['piece_name'])
df1s['piece_name'] = df1s['piece_name'].astype(float)
df1s = df1s.sort_values(by=['piece_name'])
df2s['piece_name'] = df2s['piece_name'].astype(float)
df2s = df2s.sort_values(by=['piece_name'])
df3s['piece_name'] = df3s['piece_name'].astype(float)
df3s = df3s.sort_values(by=['piece_name'])

print(f"t0.8_p0.95m: {df1m}", f"t1.2_p0.95m: {df2m}", f"t1.2_p0.9m: {df3m}", sep='\n')
print()
print(f"t0.8_p0.95s: {df1s}", f"t1.2_p0.95s: {df2s}", f"t1.2_p0.9s: {df3s}", sep='\n')

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
file_paths = [
    '/home/yuehpo/coding/DeepMIR_2023fall/hw3/TA_demo/gt_pop1k7.csv',        # Replace with your actual file path
    '/home/yuehpo/coding/DeepMIR_2023fall/hw3/TA_demo/cp_30_pop1k7.csv',  # Replace with your actual file path
    '/home/yuehpo/coding/DeepMIR_2023fall/hw3/TA_demo/cp_60_pop1k7.csv',    # Replace with your actual file path
    '/home/yuehpo/coding/DeepMIR_2023fall/hw3/TA_demo/cp_high_pop1k7.csv',    # Replace with your actual file path
]

# file_paths = [
#     '/home/yuehpo/coding/DeepMIR_2023fall/hw3/TA_demo/gt_pop1k7.csv',        # Replace with your actual file path
#     '/home/yuehpo/coding/DeepMIR_2023fall/hw3/TA_demo/t1.2_p0.95_pop1k7.csv',  # Replace with your actual file path
#     '/home/yuehpo/coding/DeepMIR_2023fall/hw3/TA_demo/t1.2_p0.9_pop1k7.csv',    # Replace with your actual file path
#     '/home/yuehpo/coding/DeepMIR_2023fall/hw3/TA_demo/t0.8_p0.95_pop1k7.csv',    # Replace with your actual file path
# ]

file_paths = [
    '/home/yuehpo/coding/DeepMIR_2023fall/hw3/TA_demo/gt_pop1k7.csv',        # Replace with your actual file path
    '/home/yuehpo/coding/DeepMIR_2023fall/hw3/TA_demo/cp_30_pop1k7.csv',  # Replace with your actual file path
    '/home/yuehpo/coding/DeepMIR_2023fall/hw3/TA_demo/t1.2_p0.95_pop1k7.csv',  # Replace with your actual file path
    # '/home/yuehpo/coding/DeepMIR_2023fall/hw3/TA_demo/t1.2_p0.9_pop1k7.csv',    # Replace with your actual file path
    
]

dataframes = [pd.read_csv(file_path) for file_path in file_paths]
# Columns you want to compare
columns_to_compare = ['H1', 'H4', 'GS'] # Replace with your actual column names
colors = ['blue', 'green', 'red'] # Replace with your actual colors
# Creating a figure for the subplots
plt.figure(figsize=(15, 5))

# Looping through each column to create a subplot
for i, column in enumerate(columns_to_compare, 1):
    # Extracting the relevant column from each dataframe
    data_to_plot = [df[column] for df in dataframes]

    # Creating a subplot for each column
    plt.subplot(1, 3, i)
    
    # Creating box plots with specified colors
    box = plt.boxplot(data_to_plot, patch_artist=True)
    
    # Applying colors
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Adding titles and labels
    plt.title(f'Box Plot for {column}')
    plt.xlabel('Dataframes')
    plt.ylabel('Values')

    # Customizing x-axis labels to represent each dataframe
    # plt.xticks([1, 2, 3, 4], ['Ground Truth', 'CP30', 'CP60', 'CP High'])
    # plt.xticks([1, 2, 3, 4], ['Ground Truth', 't1.2_p0.95', 't1.2_p0.9', 't0.8_p0.95'])
    plt.xticks([1, 2, 3], ['Ground Truth', 'CP', 'remi-xl t1.2_p0.95'])
    # plt.xticks(rotation=45)
    

# Adjust layout to prevent overlap
plt.tight_layout()

# Showing the plot
plt.savefig('boxplot.png')