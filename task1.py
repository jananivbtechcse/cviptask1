import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

#reading hackathon_working _data
df = pd.read_csv("/kaggle/input/store-transaction-data/Hackathon_Working_Data.csv")
df.head(10)
#describe the data
print(df.describe())
#counting the null counts
null_counts =df.isnull().sum()
print(null_counts)
# Fill null values with a specific value (e.g., 0)
df_filled = df.fillna(0)

# Save the DataFrame with filled values to a new CSV file
df_filled.to_csv('filled_file.csv', index=False)# Visualize transaction distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['BILL_AMT'], bins=30, kde=True)
plt.title('Distribution of Bill Amount')
plt.xlabel('Bill Amount')
plt.ylabel('Frequency')
plt.show()
#transaction based on month
labels = ['M1', 'M2', 'M3']
label_counts = df['MONTH'].value_counts()

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(label_counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Classes')
plt.show()

# Visualize customer spending behavior
plt.figure(figsize=(10, 6))
sns.scatterplot(x='BILL_AMT', y='QTY', data=df)
plt.title('Customer Spending Behavior')
plt.xlabel('Bill Amount')
plt.ylabel('Quantity')
plt.show()

#Creating Dummies For Month Column For Getting Numeric Data For Seaborn Bar Plot

month_dummies =pd.get_dummies(df['MONTH'])
month_dummies.head()

# Explore categorical variables (e.g., product categories)
plt.figure(figsize=(100, 50))  # Increase the size as needed
sns.countplot(x='GRP', data=df, palette='viridis')  # You can also specify a color palette
plt.title('Transaction Count by Product Group', fontsize=18)
plt.xlabel('Product Group', fontsize=100)
plt.ylabel('Transaction Count', fontsize=100)
plt.xticks(rotation=45, ha='right', fontsize=50)  # Adjust x-axis tick labels
plt.yticks(fontsize=50)  # Adjust y-axis tick labels
plt.show()
