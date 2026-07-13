import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("AB_NYC_2019.csv")

print("===== FIRST 5 ROWS =====")
print(df.head())


print("\n===== DATASET INFO =====")
print(df.info())


print("\nDataset Shape:", df.shape)


print("\n===== MISSING VALUES =====")
print(df.isnull().sum())


duplicates = df.duplicated().sum()

print("\n===== DUPLICATE RECORDS =====")
print("Total Duplicates:", duplicates)


df = df.drop_duplicates()


if "reviews_per_month" in df.columns:
    df["reviews_per_month"] = df["reviews_per_month"].fillna(
        df["reviews_per_month"].median()
    )

if "last_review" in df.columns:
    df["last_review"] = df["last_review"].fillna("No Review")

if "host_name" in df.columns:
    df["host_name"] = df["host_name"].fillna("Unknown")

if "name" in df.columns:
    df["name"] = df["name"].fillna("Unnamed Listing")



df.columns = df.columns.str.strip().str.lower()

print("\n===== CLEANED COLUMN NAMES =====")
print(df.columns)


print("\n===== DESCRIPTIVE STATISTICS =====")
print(df.describe())

plt.figure(figsize=(8,5))

plt.boxplot(df["price"])

plt.title("Outlier Detection - Price")

plt.ylabel("Price")

plt.show()


plt.figure(figsize=(8,5))

plt.boxplot(df["minimum_nights"])

plt.title("Outlier Detection - Minimum Nights")

plt.ylabel("Minimum Nights")

plt.show()

df.to_csv("cleaned_AB_NYC_2019.csv", index=False)

print("\nCleaned dataset saved successfully!")
df.drop_duplicates()
