# load test
import os
import pandas as pd

data = pd.read_csv("r11946024.csv")
print(data['id'][0])
for id in data['id']:
    print(id)