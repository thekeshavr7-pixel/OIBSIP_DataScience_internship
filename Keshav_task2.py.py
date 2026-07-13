

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


data = pd.read_csv('ifood_df.csv')

print("Shape:", data.shape)
print("\nColumns:", data.columns.tolist())
print("\nFirst 5 rows:")
print(data.head())



print("\nMissing Values:")
print(data.isnull().sum())


data.dropna(subset=['Income'], inplace=True)
print("\nShape after dropping missing Income rows:", data.shape)


data.drop_duplicates(inplace=True)


data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], dayfirst=True)


data['Age'] = 2024 - data['Year_Birth']


spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
              'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
data['MntTotal'] = data[spend_cols].sum(axis=1)


data['In_relationship'] = data['Marital_Status'].isin(
    ['Married', 'Together']).astype(int)


campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
                 'AcceptedCmp4', 'AcceptedCmp5']
data['AcceptedCmpOverall'] = data[campaign_cols].sum(axis=1)

print("\nCleaned Data Sample:")
print(data[['Age', 'Income', 'MntTotal', 'In_relationship',
            'AcceptedCmpOverall']].head())


print("\n===== DESCRIPTIVE STATISTICS =====")
print(data[['Age', 'Income', 'MntTotal', 'Recency',
            'Kidhome', 'Teenhome']].describe().round(2))

print("\nAverage Income        :", round(data['Income'].mean(), 2))
print("Average Age           :", round(data['Age'].mean(), 2))
print("Average Total Spent   :", round(data['MntTotal'].mean(), 2))
print("Average Recency (days):", round(data['Recency'].mean(), 2))

print("\nTotal Spending per Category:")
print(data[spend_cols].sum().sort_values(ascending=False))

print("\nEducation Distribution:")
print(data['Education'].value_counts())

print("\nMarital Status Distribution:")
print(data['Marital_Status'].value_counts())


data[spend_cols].sum().sort_values(ascending=False).plot(
    kind='bar', color='steelblue', edgecolor='black')
plt.title('Total Spending by Product Category')
plt.ylabel('Total Amount Spent')
plt.xlabel('Product Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


sns.histplot(data['Income'], kde=True, color='purple', bins=30)
plt.title('Income Distribution of Customers')
plt.xlabel('Income')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


sns.histplot(data['Age'], kde=True, color='orange', bins=30)
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


sns.scatterplot(x='Income', y='MntTotal', data=data, alpha=0.6, color='green')
plt.title('Income vs Total Spending')
plt.xlabel('Income')
plt.ylabel('Total Amount Spent')
plt.tight_layout()
plt.show()


data['Marital_Status'].value_counts().plot(
    kind='bar', color='coral', edgecolor='black')
plt.title('Marital Status Distribution')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


data['Education'].value_counts().plot(
    kind='bar', color='mediumpurple', edgecolor='black')
plt.title('Education Level Distribution')
plt.xlabel('Education')
plt.ylabel('Count')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


channel_cols = ['NumWebPurchases', 'NumCatalogPurchases',
                'NumStorePurchases', 'NumDealsPurchases']
data[channel_cols].sum().plot(
    kind='bar', color='teal', edgecolor='black')
plt.title('Total Purchases by Channel')
plt.ylabel('Total Purchases')
plt.xlabel('Channel')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


data[campaign_cols].sum().plot(
    kind='bar', color='gold', edgecolor='black')
plt.title('Campaign Acceptance Count')
plt.ylabel('Number of Customers Who Accepted')
plt.xlabel('Campaign')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

heatmap_cols = spend_cols + ['Income', 'Age', 'Recency',
                              'NumWebPurchases', 'NumStorePurchases',
                              'NumCatalogPurchases', 'MntTotal']
plt.figure(figsize=(13, 9))
sns.heatmap(data[heatmap_cols].corr(), annot=True,
            cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()


features = data[['Income', 'MntTotal', 'In_relationship']]

scaler = StandardScaler()
scaled = scaler.fit_transform(features)


inertia = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled)
    inertia.append(km.inertia_)

plt.plot(K_range, inertia, marker='o', color='blue', linewidth=2)
plt.title('Elbow Method - Finding Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(K_range)
plt.grid(True)
plt.tight_layout()
plt.show()


print("\nSilhouette Scores:")
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(scaled)
    score = silhouette_score(scaled, labels)
    print(f"  K={k}  ->  Silhouette Score: {score:.4f}")


kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled)

print("\nCustomer Count per Cluster:")
print(data['Cluster'].value_counts().sort_index())

print("\nCluster Size (%):")
print((data['Cluster'].value_counts(normalize=True)
       * 100).round(1).sort_index())




plt.figure(figsize=(9, 6))
sns.scatterplot(x='Income', y='MntTotal', hue='Cluster',
                data=data, palette='Set1', alpha=0.7)
plt.title('Customer Segments: Income vs Total Spending')
plt.xlabel('Income')
plt.ylabel('Total Amount Spent')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()


data.groupby('Cluster')['Income'].mean().plot(
    kind='bar', color='teal', edgecolor='black')
plt.title('Average Income per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Income')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


data.groupby('Cluster')['MntTotal'].mean().plot(
    kind='bar', color='salmon', edgecolor='black')
plt.title('Average Total Spending per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Total Spending')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


data.groupby('Cluster')['In_relationship'].mean().plot(
    kind='bar', color='gold', edgecolor='black')
plt.title('Proportion In Relationship per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Proportion (0=Single, 1=In Relationship)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


data.groupby('Cluster')['MntWines'].mean().plot(
    kind='bar', color='darkred', edgecolor='black')
plt.title('Average Wine Spending per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Avg Spending on Wines')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


summary_cols = ['Income', 'MntTotal', 'Age', 'Recency',
                'In_relationship', 'MntWines', 'MntMeatProducts',
                'MntFruits', 'Kidhome', 'Teenhome']

cluster_summary = data.groupby('Cluster')[summary_cols].mean().round(2)
print("\n===== CLUSTER SUMMARY =====")
print(cluster_summary.to_string())


print("""
===== MARKETING RECOMMENDATIONS =====

Cluster 0 - High Value, In Relationship:
  -> High income, high spenders, married/together
  -> Promote premium wines and quality food products
  -> Use family-oriented advertising images

Cluster 1 - Low Value, Single:
  -> Low income, low spenders, single
  -> Offer discount coupons and loyalty reward programs
  -> Focus on budget-friendly product bundles

Cluster 2 - High Value, Single:
  -> High income, high spenders, single
  -> Promote wines, fruits and premium products
  -> Use social/party/travel themed ad campaigns

Cluster 3 - Low Value, In Relationship (LARGEST GROUP):
  -> Low income, low spenders, married/together
  -> Offer family bundle deals and seasonal discounts
  -> Most important group to convert into higher spenders
""")
