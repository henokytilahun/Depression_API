import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Student_Depression_Dataset.csv')

#replace id values with 0
df['id'] = np.arange(len(df))
