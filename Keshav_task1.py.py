import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("retail_sales.csv")

print("FIRST 5 RECORDS")
print(df.head())

print("\nDATASET INFORMATION")
print(df.info())


print("\nMISSING VALUES")
print(df.isnull().sum())

df = df.drop_duplicates()


print("\n DESCRIPTIVE STATISTICS")
print(df.describe())


print("\nAverage Sales:", df["Total Amount"].mean())


print("Median Sales:", df["Total Amount"].median())


print("Mode Sales:")
print(df["Total Amount"].mode())


print("Standard Deviation:", df["Total Amount"].std())


df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)


category_sales = df.groupby("Product Category")["Total Amount"].sum()

print("\n SALES BY PRODUCT CATEGORY")
print(category_sales)

plt.figure(figsize=(8,5))
category_sales.plot(kind="bar")
plt.title("Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()


gender_sales = df.groupby("Gender")["Total Amount"].sum()

print("\n SALES BY GENDER")
print(gender_sales)

plt.figure(figsize=(6,4))
gender_sales.plot(kind="bar")
plt.title("Sales by Gender")
plt.xlabel("Gender")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

monthly_sales = df.groupby(df["Date"].dt.month)["Total Amount"].sum()

print("\nMONTHLY SALES")
print(monthly_sales)

plt.figure(figsize=(8,5))
monthly_sales.plot(kind="line", marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()
plt.figure(figsize=(8,5))
plt.hist(df["Age"], bins=10)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

numeric_cols = df.select_dtypes(include=["int64","float64"])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_cols.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()


top_customers = df.groupby("Customer ID")["Total Amount"].sum().sort_values(ascending=False).head(10)

print("\nTOP 10 CUSTOMERS")
print(top_customers)


print("\nINSIGHTS")
print("1. Identify the product category with highest sales.")
print("2. Observe monthly sales trends.")
print("3. Compare male and female purchasing patterns.")
print("4. Analyze customer age distribution.")
print("5. Find top spending customers.")

print("\nRECOMMENDATIONS")
print("1. Increase inventory for top-selling categories.")
print("2. Run promotions during low-sales months.")
print("3. Focus marketing on high-value customer groups.")
print("4. Reward loyal customers with offers.")

print("\nPROJECT COMPLETED SUCCESSFULLY!")
