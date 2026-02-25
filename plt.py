import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt   

df =  pd.read_csv('titanic_clean.csv')
sns.set_style('whitegrid')  

plt.figure(figsize=(10,6))
sns.barplot(x='Pclass', y='Survived', data=df, palette='viridis')
plt.title('Survival Rate by Passenger Class on the Titanic', fontsize=16)
plt.xlabel('Passenger Class', fontsize=14)
plt.ylabel('Survival Rate', fontsize=14)
plt.ylim(0, 1)
plt.show()




g = sns.catplot(
    data=df, 
    x='Pclass', 
    y='Fare', 
    hue='Survived', 
    col='Embarked', 
    kind='box', 
    showfliers=False,
    palette='Set1',
    height=5, 
    aspect=0.8
)
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle('Fare Distribution by Pclass and Survival across Embarked Ports', fontsize=16)
g.set_axis_labels("Passenger Class", "Fare")
g.add_legend(title="Survived")
plt.savefig('figure2_multivariate.png')
plt.show()
    