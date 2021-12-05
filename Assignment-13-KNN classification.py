
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

df=pd.read_csv("glass.csv")
df

X= df.iloc[:,:9]
Y= df.iloc[:,9]

# # Standardizing the data

def get_standardized_data(data):
    df_norm = (data-data.min())/(data.max()-data.min())
    return(df_norm)

X=get_standardized_data(X)
X

df["Type"].value_counts()

#Since the class 5 has just 4 data points hence we keep the n_splits as 3




