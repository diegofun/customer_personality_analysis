import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import outliers
from yellowbrick.cluster import KElbowVisualizer

#############################################
#               DATA

#   to show the whole dataframe, no summary
# pd.set_option('display.max_columns', None, 'display.max_rows', None)

#   read file
data = pd.read_csv("marketing_campaign2.csv")

#   null values
print("\n\n         NULL VALUES BY COLUMN")
print(data.isnull().sum())

#   show null values in matrix
# msno.matrix(data)

#   dropping rows with null data, because there are only a few
data = data.dropna()
print("\n\n         NULLS AFTER DELETE ROWS")
print(data.isnull().sum())

#   duplicated rows
print("\n\n         DUPLICATED ROWS")
print(data.duplicated().sum())

# Dt_Customer to date-time
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='mixed', dayfirst=True)
print("\n\nNewest customer's enrolment date:", max(data['Dt_Customer']))
print("Oldest customer's enrolment date:", min(data['Dt_Customer']))

# Age from Year_Birth (up to 2015 since data collected was at that age)
data['Age'] = 2015 - data['Year_Birth']

# Unifying spending
data['Spent'] = data['MntWines'] \
                + data['MntFruits'] \
                + data['MntMeatProducts'] \
                + data['MntFishProducts'] \
                + data['MntSweetProducts'] \
                + data['MntGoldProds']

# Living_With (binary data) from Marital_Status
data['Living_With'] = data['Marital_Status'].replace(
    {'Married': 'Partner', 'Together': 'Partner', 'Absurd': 'Alone', 'Widow': 'Alone', 'YOLO': 'Alone',
     'Divorced': 'Alone', 'Single': 'Alone'})

# Children from kids and teens
data['Children'] = data['Kidhome'] + data['Teenhome']

# Family_Size from children and living_with
data['Family_Size'] = data['Living_With'].replace({'Alone': 1, 'Partner':2}) + data['Children']

# Is_Parent from children
data['Is_Parent'] = np.where(data.Children > 0, 1, 0)

# Drop not needed columns
to_drop = ['Marital_Status', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue', 'Year_Birth', 'ID']
data = data.drop(to_drop, axis=1)

print("\n\n")
print(data.describe())

print("\n\n")
print(data.describe(include=object).T)


################################################
#           PLOTTING
#   Spent, Income, Age pairplot with Children as hue
#sns.pairplot(data, vars=['Spent', 'Income', 'Age'], hue='Children')

#   Spent, Income, Age pairplot with Education as hue
#sns.pairplot(data, vars=['Spent', 'Income', 'Age'], hue='Education')

#   Summary of educational level
#plt.figure()
#data['Education'].value_counts().plot.pie()

#   For a demographic analysis, it must be correlated the children, the age, and education variables.
#   To do this, the Age variable should be cast as ordinal data.
#   Not needed for the spent analysis.


#############################################
#           outliers
print('\n\n')
numerical = ['Income', 'Recency', 'Age', 'Spent']
out = outliers.detect_outliers(numerical, data)


#########################################################
#           CATEGORICAL OUTLIERS
# fraction frequency/total data
print('\n           CATEGORICAL OUTLIERS')
categorical = [var for var in data.columns if data[var].dtype == 'O']
for var in categorical:
    print(data[var].value_counts() / float(len(data)))


#####################################################
#           Encoding categorical ordinal features
data['Education'] = data['Education'].replace({'Basic': 0, '2n Cycle': 1, 'Graduation': 2, 'Master': 3, 'PhD': 4})
data['Living_With'] = data['Living_With'].map({'Alone': 0, 'Partner': 1})


#####################################################
#           SCALING NUMERICAL FEATURES
#           and dropping outliers on deals accepted and promotions
data_old = data.copy()
data_unscaled = data.copy()
for i in range(len(out)):
    data = data.drop(out[i].index)
    data_unscaled = data_unscaled.drop(out[i].index)
cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response']
data = data.drop(cols_del, axis=1)
scaler = StandardScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
data_unscaled.reset_index(drop=True, inplace=True)


#####################################################
#       FEATURES CORRELATION
#   corr_mat = data.corr()
# plt.figure(figsize=(20,20))
# sns.heatmap(corr_mat, annot=True, cmap='mako', center=0)


######################################################
#           DIMENSIONALITY REDUCTION PCA
#   what is lambda i?
p = PCA(n_components=3)
p.fit(data)
W = pd.DataFrame(p.components_.T, index=data.columns, columns=['W1', 'W2', 'W3'])
print('\n\n')
print('         PCA ANALYSIS')
print(f"{p.explained_variance_}     Variance ")
print(f"{p.explained_variance_ratio_}       Explained variability ")
print(f"{p.explained_variance_ratio_.cumsum()}      Cumulative sum")
# sns.barplot(x=list(range(1, 4)), y=p.explained_variance_, palette='GnBu_r')
# plt.xlabel('i')
# plt.ylabel('Lambda i')
data_PCA = pd.DataFrame(p.transform(data), columns=(['col1', 'col2', 'col3']))
data_PCA.describe()
# x = data_PCA['col1']
# y = data_PCA['col2']
# z = data_PCA['col3']
# fig = plt.figure(figsize=(13,8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c='darkred', marker='o')
# ax.set_title('A 3D Projection of Data In the Reduced Dimension')
# plt.show()


#######################################
#           ELBOW FOR THE NUMBER OF CLUSTER
# Elbow_M = KElbowVisualizer(KMeans(), k=10)
# Elbow_M.fit(data_PCA)
# Elbow_M.show()


#######################################
#           K-MEANS (WITH ELBOW SUGGESTED CLUSTERS)
#       I'm not sure if adding the kmeans data to previous dataframes makes sens
data_PCA['Clusters'] = AgglomerativeClustering(n_clusters=4).fit_predict(data_PCA)
data['Clusters'] = data_PCA['Clusters']
data_unscaled['Clusters'] = data_PCA['Clusters']
# plt.figure()
# ax = plt.subplot(111, projection='3d', label='bla')
# ax.scatter(x, y, z, s=40, c=data_PCA['Clusters'], marker='o', cmap='Set1_r')
# ax.set_title('Clusters')
# plt.show()
print('\n\n')
print('         KMEANS WITH 4 ELBOW SUGGESTED CLUSTERS')
print(data_PCA.Clusters.value_counts())
# plt.figure()
# sns.swarmplot(x=data['Clusters'], y=data['Spent'], color="#CBEDDD", alpha=0.7)
# sns.boxenplot(x=data_unscaled['Clusters'], y=data_unscaled['Spent'])


#####################################
#           TOTAL CAMPAIGN ACCEPTED
data_unscaled['Total_Promos'] = data_unscaled['AcceptedCmp1'] +\
                                data_unscaled['AcceptedCmp2'] +\
                                data_unscaled['AcceptedCmp3'] +\
                                data_unscaled['AcceptedCmp4'] +\
                                data_unscaled['AcceptedCmp5']
# plt.figure()
# pl = sns.countplot(x=data_unscaled['Total_Promos'], hue=data_unscaled['Clusters'])
# pl.set_title('Count Of Promotion Accepted')
# pl.set_xlabel('Number Of Total Accepted Promotions')
# plt.legend(loc='upper right')
# plt.show()


#################################################
#           PURCHASES CASES
# plt.figure()
# pl = sns.boxenplot(y=data_unscaled.NumDealsPurchases, x=data_unscaled['Clusters'])
# pl.set_title('Number of Deals Purchased')

# plt.figure()
# pl = sns.boxenplot(y=data_unscaled.NumWebPurchases, x=data_unscaled['Clusters'])
# pl.set_title('Number of Web Purchased')

# plt.figure()
# pl = sns.boxenplot(y=data_unscaled.NumCatalogPurchases, x=data_unscaled['Clusters'])
# pl.set_title('Number of Catalog Purchased')

# plt.figure()
# pl = sns.boxenplot(y=data_unscaled.NumStorePurchases, x=data_unscaled['Clusters'])
# pl.set_title('Number of Store Purchased')


##################################################
#           GROUPS DESCRIPTION
Personal = ['Kidhome', 'Teenhome', 'Age', 'Children', 'Family_Size', 'Is_Parent', 'Education', 'Living_With']

for i in Personal:
    sns.jointplot(
        x=data_unscaled[i],
        y=data_unscaled['Spent'],
        hue=data_unscaled['Clusters'],
        kind='kde',
        palette='rocket')

#   (orden de gastos de mayor a menor) Cluster X: Características

#   +++  Cluster 1: tiene hijos, 1 niño o 1 adolescente, algunos con 2.
#                   educacion del 1 al 4. 30 a 70 años.

#   +    Cluster 3: mayoría 1 niño, luego 1 adolescentes, muchos no
#                   tienen algunos tienen 2. educacion del 0 al 4. 15 a 70 años

#   ++   Cluster 0: tiene hijos, la mayoría 2, 1 niño y 1 adolescente,
#                   o dos de los mismos, hasta 3. educacion del 1 al 4. 23 a 70 años

#   ++++ Cluster 2: 0 hijos. educacion del 1 al 4. 18 a 70 años.

