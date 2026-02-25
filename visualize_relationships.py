import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_multivariate_plot():
    # Load dataset
    df = pd.read_csv('titanic_clean.csv')

    # Clean string columns (remove trailing whitespace)
    # The user noted trailing spaces in Sex, likely also present in Embarked
    if 'Sex' in df.columns and df['Sex'].dtype == 'object':
        df['Sex'] = df['Sex'].str.strip()
    if 'Embarked' in df.columns and df['Embarked'].dtype == 'object':
        df['Embarked'] = df['Embarked'].str.strip()

    # Drop rows with missing values in the relevant columns for cleaner plotting
    df_clean = df.dropna(subset=['Embarked', 'Pclass', 'Fare', 'Survived'])

    # Set the style
    sns.set_theme(style="ticks")

    # Create the visualization
    # Goal: Compare Embarked, Pclass, Fare, and Survived
    # X: Pclass
    # Y: Fare
    # Hue: Survived
    # Column: Embarked
    
    g = sns.catplot(
        data=df_clean,
        x="Pclass", 
        y="Fare", 
        hue="Survived", 
        col="Embarked",
        kind="box",  # Box plot shows the distribution of Fare well
        height=6, 
        aspect=0.8,
        palette="muted"
    )

    # Adjust title and labels
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Relationship between Fare, Pclass, Embarked and Survival', fontsize=16)
    g.set_axis_labels("Passenger Class", "Fare")
    g._legend.set_title("Survived\n(0: No, 1: Yes)")

    # Save the plot
    output_file = 'multivariate_plot.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    create_multivariate_plot()
