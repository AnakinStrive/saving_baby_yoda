import streamlit as st
import time
import numpy as np
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from math import sqrt
import base64

st.title('Finding Baby Yoda!')
'As one of the groups of Darkside, we got a report from our spies that says Baby Yoda is in the galaxy and our spies shared possible coordinates. Our master young Anakin Skywalker wanted from these steps;'

'Plotting the clusters with K-Means'
'Find the coordinates of the planet where it is hidden!'
'Giving the coordinates to our master so he will use the force and give us possible coordinates on the planet'

'After getting the new coordinates, our master wants these steps;'
'Use PCA (top 2components) to project the Dark Force'
'Use KMeans'
'Give him the coordinates of Baby Yoda then he can convince him to be in Darkside!'

LOGO_IMAGE = r"C:\Users\gabri\Documents\Github\savingbabyyoda\datasets\The_Mandalorian.jpg"
st.markdown(
    """
    <style>
    .container {
        display: flex;
    }
    .logo-text {
        font-weight:700 !important;
        font-size:50px !important;
        color: #f9a01b !important;
        padding-top: 75px !important;
    }
    .logo-img {
        float:right;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" src="data:image/jpg;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
    </div>
    """,
    unsafe_allow_html=True
)
df=pd.read_csv(r"C:\Users\gabri\Documents\Github\savingbabyyoda\datasets\galaxies.csv")
df.head()

'Now we can plot the universe with scatter plots!!'
small_universe, ax = plt.subplots()
#fig1 = plt.figure()
ax.scatter(df["X"],df["Y"])
st.pyplot(small_universe)

X=df[["X","Y"]]
#Creating K Means object
k_means = KMeans(init = "k-means++", n_clusters = 3, n_init = 8)
#Fitting
k_means.fit(X)

#Labels
k_means_cluster_centers = k_means.cluster_centers_
labels = k_means.labels_
result = np.where(k_means_cluster_centers == np.amax(k_means_cluster_centers,))

result=int(result[1])
st.text('Plotting upper cluster')
outer_rim, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,8))
ax1.set_title('K Means')
ax1.scatter(df["X"],df["Y"],c=labels,cmap='autumn')
ax2.set_title("Original")
ax2.scatter(df["X"],df["Y"])
ax1.grid()
ax2.grid()
st.pyplot(outer_rim)

df["Labels"]=labels


'Plotting upper cluster'
target_galaxy, ax = plt.subplots()
ax.scatter(df.loc[df["Labels"]==result]["X"],df.loc[df["Labels"]==result]["Y"])
st.pyplot(target_galaxy)

#Finding planet location where Baby Yoda is.
x_baby=df.loc[df["Labels"]==result]["X"].max()
y_baby=df.loc[df["X"]==x_baby,"Y"]

st.text('Marking the location of the planet')

target_planet, ax = plt.subplots()
ax.scatter(df.loc[df["Labels"]==1]["X"],df.loc[df["Labels"]==1]["Y"])
ax.scatter(x_baby,y_baby,s=90, c='red', marker='X')
st.pyplot(target_planet)

'Now it is time to giving the location to our master Anakin Skywalker'

st.text(x_baby)
st.text(y_baby.values[0])

'Our master gave us a location dataset which name is "planet.csv"'
df_new=pd.read_csv(r"C:\Users\gabri\Documents\Github\savingbabyyoda\datasets\planet.csv")
df_new.head()

'Time to apply PCA for creating only 2 components'
pca = PCA(n_components=2)

X_new=df_new[["X","Y","Z","Temp","climate"]].values

components = pca.fit_transform(X_new)

'Finding the gravity center with K Means'

model = KMeans(n_clusters=1)
model.fit(components)
gravity = model.predict(components)

gravity_center = model.cluster_centers_
gravity_center_x = gravity_center[:,0]
gravity_center_y = gravity_center[:,1]

# finds the distance between a point and all the others
def find_distance(point, points):
    distances = []
    for i, p in enumerate(points):
        distances.append(sqrt((point[:,0] - points[i,0])**2 + (point[:,1] - points[i,1])**2))
    
    return distances

dist = find_distance(gravity_center, components)

# find the closest point given a list of points and a list of distances
def closest_point(points, distances):

    val, idx = min((val, idx) for (idx, val) in enumerate(distances))

    return points[idx]

baby_coords = closest_point(components, dist)
'Getting the principal components coordinates to plot the map and the gravity center coordinates to plot on to the map'

# Getting principal components coordinates
xs = components[:, 0]
ys = components[:, 1]

# Getting the gravity center coordinates
gravity_center = model.cluster_centers_
gravity_center_x = gravity_center[:,0]
gravity_center_y = gravity_center[:,1]

'Plotting location of the Baby Yoda with coordinates'

final_location, ax = plt.subplots()
#fig = plt.figure(figsize=(20,10))
ax.scatter(x=xs, y=ys)
ax.scatter(x=gravity_center_x, y=gravity_center_y, marker='X', s=200)
ax.scatter(baby_coords[0], baby_coords[1], c='red', marker='+', s=200)
st.pyplot(final_location)
print(f'The coordinates of baby Yoda are: {baby_coords}')
print(f'the the distance between them \nand the gravity center {gravity_center}\nis {min(dist)}')

'Now our master and baby will meet!'

LOGO_IMAGE1 = r"C:\Users\gabri\Documents\Github\savingbabyyoda\datasets\The-Mandalorian-Baby.jpg"
st.markdown(
    """
    <style>
    .container {
        display: flex;
    }
    .logo-text {
        font-weight:700 !important;
        font-size:50px !important;
        color: #f9a01b !important;
        padding-top: 75px !important;
    }
    .logo-img {
        float:right;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" src="data:image/jpg;base64,{base64.b64encode(open(LOGO_IMAGE1, "rb").read()).decode()}">
    </div>
    """,
    unsafe_allow_html=True
)