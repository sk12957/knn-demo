import numpy.random
import numpy as np
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt


@st.cache
def get_data():
    return pd.read_csv("RidingMowers.csv")


@st.cache(allow_output_mutation=True)
def get_model():
    return pickle.load(open("knn_mower.model", "rb"))


knn_loaded = get_model()

lot_size = st.sidebar.slider("Lot size", 0, 150, 75)
income = st.sidebar.slider("Income", 10, 100, 55)

y_pred = knn_loaded.predict([[income, lot_size]])

if y_pred == 1:
    st.title("Bought")
else:
    st.title("Not Bought")
st.metric("N-Neighbours", 3)
df = get_data()

fig, ax = plt.subplots()
df = pd.get_dummies(df)

df_bought = df.loc[df["Response_Bought"] == 1]
df_not = df.loc[df["Response_Not Bought"] == 1]
ax.scatter(df_bought.Lot_Size, df_bought.Income, c="#27A7D8")
ax.scatter(df_not.Lot_Size, df_not.Income, c="#D85827")
if not y_pred:
    ax.scatter(lot_size, income, s=150, edgecolors="#D85827", c="#5FA15E")
else:
    ax.scatter(lot_size, income, s=150, edgecolors="#27A7D8", c="#5FA15E")
ax.annotate(",".join(str(x) for x in [lot_size, income]), (lot_size, income))
plt.xlabel("Lot size")
plt.ylabel("Income")
st.pyplot(fig)
