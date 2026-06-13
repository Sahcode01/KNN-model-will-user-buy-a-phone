import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

st.title("Will User Buy Phone?")
st.write("This is to check if a particular user will buy phone or not")

oldUsers = np.array([
    [1, 0],
    [4, 0],
    [2, 1],
    [5, 1],
    [4, 1],
    [3, 0],
    [2, 0],
    [1, 0],

])

outcome = np.array([0,1,1,1,1,1,0,0])

model = KNeighborsClassifier(n_neighbors=3)
model.fit(oldUsers,outcome)

st.sidebar.header("New User Data")
phoneAge = st.sidebar.slider("How old is the phone?", min_value=1,max_value=6,value=2,step=1)
screenCondition = st.sidebar.selectbox("Is their phone screen cracked?", options= ["No","Yes"])

if screenCondition == "Yes":
    screenConditionNum = 1
else:
    screenConditionNum = 0

currentUserDf = pd.DataFrame({
    "How Old is the phone":[phoneAge],
    "Is the screen cracked?":[screenCondition]
})

st.dataframe(currentUserDf)

if st.button("Run KNN Algorithm"):
    theNewUser = [[phoneAge, screenConditionNum]]
    prediction = model.predict(theNewUser)[0]

    probabilities = model.predict_proba(theNewUser)[0]

    if prediction == 1:
        st.success("This user will likely buy phone")
        st.write(f"Confidence: AI is **{probabilities[1]*100:.0f}**% sure")
    else:
        st.error("This user will likely not buy phone")
        st.write(f"Confidence: AI is **{probabilities[0]*100:.0f}**% sure")

