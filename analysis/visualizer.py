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

    missing_values_df.plot(kind="barh", figsize=(8, 6), color="blue", legend=False)

    plt.title("Percentage of Missing Values in Each Column")
    plt.ylabel("Columns")
    plt.xlabel("Percentage Missing")
    plt.show()


def visualize_gender_distribution(movies, style="darkgrid"):
    roles = ["actor_gender", "director_gender", "producer_gender", "writer_gender"]
    fig, axes = plt.subplots(1, 4, figsize=(8, 5))
    fig.suptitle("Gender Distribution in Different Roles", y=0.8)

    for i, role in enumerate(roles):
        gender_counts = movies[role].value_counts()
        axes[i].pie(
            gender_counts, labels=gender_counts.index, autopct="%1.1f%%", startangle=140
        )
        axes[i].set_title(role.replace("_", " ").title())

    plt.tight_layout()
    plt.show()


def visualize_gender_proportion_repartition(movies, style="darkgrid"):
    # Filtering and extract proportions of male/female characters per movie
    movies = movies.loc[movies["actor_gender"].isin(["F", "M"])].copy(deep=True)
    male_female_counts = (
        movies.groupby(["wikiID", "actor_gender"])["character_name"]
        .nunique()
        .unstack(fill_value=0)
    )
    male_female_counts = male_female_counts.join(
        movies.groupby("wikiID")["character_name"].nunique().rename("total_char")
    ).assign(
        percents_of_female=lambda x: x["F"] / x["total_char"] * 100,
        percents_of_male=lambda x: x["M"] / x["total_char"] * 100,
    )

    male_female_counts.rename(
        columns={"F": "female_char", "M": "male_char"}, inplace=True
    )
    male_female_counts = male_female_counts.merge(
        movies[["year", "wikiID"]].drop_duplicates(), on="wikiID", how="left"
    )

    male_female_counts.dropna(
        inplace=True, subset=["percents_of_female", "percents_of_male"]
    )

    # Plot results into two distinct plot
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    sns.set_style(style)

    sns.histplot(
        male_female_counts["percents_of_female"],
        bins=50,
        label="F",
        alpha=1,
        color="pink",
        ax=ax[0],
        kde=True,
    )
    ax[0].legend(loc="upper right", fontsize=14)

    sns.histplot(
        male_female_counts["percents_of_male"],
        bins=50,
        label="M",
        alpha=0.5,
        color="blue",
        ax=ax[1],
        kde=True,
    )
    ax[1].legend(loc="upper right", fontsize=14)

    fig.supxlabel("Percentage of characters per movie", fontsize=16)
    fig.suptitle(
        "Histograms of the percentage of Female and Male characters per movie",
        fontsize=20,
    )

    plt.tight_layout()
    plt.show()
