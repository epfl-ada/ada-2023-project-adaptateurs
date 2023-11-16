import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["figure.dpi"] = 300


def visualize_year_distribution(movies, style="darkgrid"):
    """
    Visualize the distribution of movies across years.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing movie data.
    style (str): Style of the plot. Default is "darkgrid".

    Returns:
    None
    """

    plt.figure(figsize=(8, 4))

    unique_movies = movies.drop_duplicates(subset="movie_title")
    mean = unique_movies.groupby("year")["wikiID"].count().mean()

    sns.set_style(style)
    sns.histplot(unique_movies.movie_release_date, bins=100, color="blue", kde=True)

    plt.title("Distribution of Movies Across Years")
    plt.xlabel("Release Year")
    plt.ylabel("Number of Movies")
    plt.show()

    print("Total number of movies: ", len(unique_movies))
    print(f"Mean number of movies per year: {round(mean)}")


def visualize_missing_values(movies, style="darkgrid"):
    """
    Visualize the percentage of missing values in each column of a pandas DataFrame.

    Parameters:
    movies (pandas.DataFrame): The DataFrame to analyze.
    style (str): The style of the plot. Default is "darkgrid".

    Returns:
    None
    """

    missing_values = movies.isnull().sum()
    missing_percentage = (movies.isnull().sum() / len(movies)) * 100
    missing_values_df = pd.DataFrame(
        {"Missing Values": missing_values, "Percentage": missing_percentage}
    ).sort_values(by="Percentage", ascending=False)
    missing_values_df = missing_values_df[missing_values_df.Percentage > 0]

    sns.set_style(style)
    missing_values_df.Percentage.plot(
        kind="barh", figsize=(8, 6), color="blue", legend=False, alpha=0.5
    )

    plt.title("Percentage of Missing Values in Each Column")
    plt.ylabel("Columns")
    plt.xlabel("Percentage Missing")
    plt.show()


def visualize_gender_distribution(movies, style="darkgrid"):
    """
    Visualize the gender distribution in different roles of a given dataset of movies.

    Parameters:
    movies (pandas.DataFrame): The dataset of movies to analyze.
    style (str): The style of the plot. Default is "darkgrid".

    Returns:
    None
    """

    roles = ["actor_gender", "director_gender", "producer_gender"]
    fig, axes = plt.subplots(1, len(roles), figsize=(8, 5))
    fig.suptitle("Gender Distribution in Different Roles", y=0.8)

    for i, role in enumerate(roles):
        gender_counts = movies[role].value_counts()
        axes[i].pie(
            gender_counts, labels=gender_counts.index, autopct="%1.1f%%", startangle=140
        )
        axes[i].set_title(role.replace("_", " ").title())

    plt.tight_layout()
    plt.show()


def visualize_actors_distribution(movies, style="darkgrid"):
    """
    Visualize the distribution of the number of actors in movies.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing the movies data.
    style (str): Style of the plot (default is "darkgrid").

    Returns:
    None
    """

    number_of_actors = movies.groupby("wikiID")["actor_name"].agg("count")

    plt.hist(number_of_actors, bins=50, log=True)
    plt.xlabel("Number of actors")
    plt.ylabel("Number of movies")
    plt.title("Distribution of actors")
    plt.show()


def visualize_gender_proportion_repartition(movies, style="darkgrid"):
    """
    Visualizes the proportion of male and female characters in movies.

    Parameters:
    movies (pandas.DataFrame): A DataFrame containing information about movies.
    style (str): The style of the plot. Default is "darkgrid".

    Returns:
    pandas.DataFrame: A DataFrame containing the proportion of male and female characters in each movie.
    """

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

    return male_female_counts


def visualize_regression(movies_gender_prop, style="darkgrid"):
    """
    Visualize the correlation between the percentage of women actors in movies and their average rating.

    Args:
        movies_gender_prop (pandas.DataFrame): A DataFrame containing the percentage of women actors and the average rating of movies.
        style (str, optional): The style of the plot. Defaults to "darkgrid".
    """

    plt.figure(figsize=(10, 5))

    sns.set_style(style)
    sns.regplot(
        x=movies_gender_prop["percents_of_female"],
        y=movies_gender_prop["vote_average"],
        scatter_kws={"s": 2},
        line_kws={"color": "red"},
    )

    plt.ylabel("Average Rating", fontsize=14)
    plt.xlabel("Percentage of women actors", fontsize=14)

    plt.title("Correlation between Rating and proportion of Women", fontsize=20)

    plt.show()


def visualize_age_distribution_by_gender(movies, style="darkgrid"):
    """
    Visualize the distribution of actor ages by gender using a boxplot.

    Args:
    movies (pandas.DataFrame): A DataFrame containing movie data with columns 'actor_gender' and 'actor_age_at_movie_release'.

    Returns:
    None
    """

    plt.figure(figsize=(15, 6))
    sns.set_style(style)
    sns.boxplot(
        x="actor_gender", y="actor_age_at_movie_release", data=movies, palette="flare"
    )
    plt.title("Distribution of Actor Ages by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Age")
    plt.show()


def visualize_age_evolution(movies, style="darkgrid"):
    """
    Visualize the evolution of actors' age over the years.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing information about movies and actors.
    style (str): Style of the plot. Default is "darkgrid".

    Returns:
    None
    """

    plt.figure(figsize=(10, 6))
    sns.set_style(style)
    sns.lineplot(
        data=movies,
        x="year",
        y="actor_age_at_movie_release",
        hue="actor_gender",
        errorbar=("ci", 95),
        palette=["red", "blue"],
    )
    plt.title("Evolution of actors' age over the years")
    plt.xlabel("Year")
    plt.ylabel("Age")
    plt.show()


def visualize_feminity_score_distribution(movies, style="darkgrid"):
    """
    Visualize the distribution of the femininity score across movies.

    Args:
    movies (pandas.DataFrame): A DataFrame containing movie data with a 'femininity_score' column.

    Returns:
    None
    """

    plt.figure(figsize=(8, 4))
    sns.set_style(style)
    sns.histplot(movies.feminity_score, bins=40, color="skyblue", log=True)
    plt.title("Distribution of Femininity Score Across Movies")
    plt.xlabel("Femininity Score")
    plt.ylabel("Number of Movies")
    plt.show()
