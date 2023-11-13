import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# mpl.rcParams[axes.spines.right] = False
# mpl.rcParams[axes.spines.top] = False
# mpl.rcParams[text.usetex] = True
# plt.rcParams.update({text.latex.preamble: r"\usepackage{amsfonts}"})
mpl.rcParams["figure.dpi"] = 300


def visualize_year_distribution(movies, style="darkgrid"):
    plt.figure(figsize=(8, 4))

    sns.set_style(style)
    sns.histplot(movies.movie_release_date, bins=100, color="blue", kde=True)

    plt.title("Distribution of Movies Across Years")
    plt.xlabel("Release Year")
    plt.ylabel("Number of Movies")
    plt.show()


def visualize_missing_values(movies, style="darkgrid"):
    missing_values = movies.isnull().sum()
    missing_percentage = (movies.isnull().sum() / len(movies)) * 100
    missing_values_df = pd.DataFrame(
            {"Missing Values": missing_values, "Percentage": missing_percentage}
        ).sort_values(by="Percentage", ascending=False)
    missing_values_df = missing_values_df[missing_values_df.Percentage > 0]
    print(missing_values_df)
    
    missing_values_df.plot(kind='barh', figsize=(8, 6), color='blue', legend=False)
    
    plt.title('Percentage of Missing Values in Each Column')
    plt.ylabel('Columns')
    plt.xlabel('Percentage Missing')
    plt.show()
    
def visualize_gender_distribution(movies, style='darkgrid'):
    roles = ['actor_gender', 'director_gender', 'producer_gender', 'writer_gender']
    fig, axes = plt.subplots(1, 4, figsize=(8, 5))
    fig.suptitle('Gender Distribution in Different Roles', y=0.8)

    for i, role in enumerate(roles):
        gender_counts = movies[role].value_counts()
        axes[i].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
        axes[i].set_title(role.replace('_', ' ').title())

    plt.tight_layout()
    plt.show()