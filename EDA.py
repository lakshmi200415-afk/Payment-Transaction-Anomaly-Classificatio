import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')

print(df.head())
print(df['Class'].value_counts())

# Graph 1 - Normal vs Fraud
df['Class'].value_counts().plot(kind='bar', color=['green','red'])
plt.title('Normal(0) vs Fraud(1) - Class')
plt.xlabel('Class')
plt.ylabel('Count')
plt.savefig('graph1.png')
plt.show()

# Graph 2 - Amount vs Class
plt.figure()
plt.hist(df[df['Class']==0]['amount'], bins=30, alpha=0.5, label='Normal')
plt.hist(df[df['Class']==1]['amount'], bins=30, alpha=0.5, label='Fraud')
plt.title('Amount vs Anomaly')
plt.xlabel('Amount')
plt.legend()
plt.savefig('graph2.png')
plt.show()

print("EDA DONE - Graphs saved as graph1.png, graph2.png")