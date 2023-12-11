import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import mpld3


mpl.rcParams["figure.dpi"] = 300
color_M = 'darkblue' # ~rgb(0,0,139)
color_F = 'crimson'  # ~rgb(182,28,66)
color_G = 'dimgrey'

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
    sns.histplot(unique_movies.movie_release_date, bins=100, color=color_G, kde=True)

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
        kind="barh", figsize=(8, 6), color=color_G, legend=False, alpha=0.5
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
            gender_counts,
            labels=gender_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=[color_M, color_F],
        )

        axes[i].set_title(role.replace("_", " ").title(), size=10)

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

    plt.hist(number_of_actors, bins=50, log=True, color=color_G, alpha=0.5)
    plt.xlabel("Number of actors")
    plt.ylabel("Number of movies")
    plt.title("Distribution of actors")
    plt.show()


# not used anymore -> see visualize_gender_prop
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
        color=color_G,
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


def visualize_gender_prop(movies, style="darkgrid"):
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

    _df = pd.DataFrame(
        {
            "Female actresses": male_female_counts["percents_of_female"],
            "Male actors": male_female_counts["percents_of_male"],
        }
    )

    # Plot results into two distinct plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 8), sharey=True)

    sns.set_style(style)

    sns.histplot(
        data=_df,
        ax=ax,
        stat="count",
        multiple="stack",
        bins=20,
        alpha=1,
        palette=[color_F, color_M],
        element="bars",
        legend=True,
    )

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
        color=color_G,
        scatter_kws={"s": 2, "alpha": 0.3},
        line_kws={"color": color_F},
    )

    plt.ylabel("Average Rating", fontsize=14)
    plt.xlabel("Percentage of women actresses", fontsize=14)

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
        x="actor_gender",
        y="actor_age_at_movie_release",
        data=movies,
        palette=[color_F, color_M],
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
        palette=[color_F, color_M]
    )
    plt.title("Evolution of actors' age over the years")
    plt.xlabel("Year")
    plt.ylabel("Age")
    plt.show()


def visualize_age_evolution_HTML(movies, output_html='html_plots/age_evolution_plot.html'):
    """
    Visualize the average evolution of actors' and actresses' age over the years using Plotly and export as HTML.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing information about movies, actors, and their ages.
    output_html (str): The name of the output HTML file.

    Returns:
    None
    """

    # Calculate the median age of actors and actresses for each year
    median_ages = movies.groupby(['year', 'actor_gender'])['actor_age_at_movie_release'].median().reset_index()

    color_discrete_map = {'F': color_F, 'M': color_M}

    # Create the plot using Plotly
    fig = px.line(
        median_ages,
        x="year",
        y="actor_age_at_movie_release",
        color="actor_gender",
        title="Evolution of the median age of Actors and Actresses over the Years",
        labels={"actor_age_at_movie_release": "Median Age", "actor_gender": "Actor Gender", "year": "Year"},
        color_discrete_map=color_discrete_map
    )

    fig.update_yaxes(range=[0, 50])
    
    # Display the plot
    fig.show()

    # Export the plot to an HTML file
    fig.write_html(output_html)


def visualize_bechdel_distribution(movies_with_bechdel, style="darkgrid"):
    """
    Visualize the distribution of Bechdel test scores across movies.
    Args: movies_with_bechdel (pandas.DataFrame): A DataFrame containing movie data with a 'bechdel_rating' column.
    Returns: None
    """
    df_unique_movies = movies_with_bechdel.drop_duplicates(
        subset=["movie_title", "year"]
    )

    # Group by year and Bechdel grade, then count the number of movies
    grouped_bechdel = (
        df_unique_movies.groupby(["year", "bechdel_rating"])
        .size()
        .unstack(fill_value=0)
    )

    # Determine the complete range of years
    all_years = range(grouped_bechdel.index.min(), grouped_bechdel.index.max() + 1)

    # Create a DataFrame with all years and merge
    all_years_df = pd.DataFrame(index=all_years)

    grouped = all_years_df.merge(
        grouped_bechdel, left_index=True, right_index=True, how="left"
    ).fillna(0)

    # Define a red-to-green colormap
    cmap = plt.get_cmap("RdYlGn")
    colors = cmap(np.linspace(0, 1, len(grouped.columns)))

    # Create a stacked bar chart
    grouped.plot(kind="bar", stacked=True, figsize=(15, 7), color=colors)

    # Filter x-axis ticks to display one year every five years
    tick_labels = [year if year % 5 == 0 else "" for year in all_years]
    plt.xticks(range(len(grouped)), tick_labels, rotation=0)

    # Adding labels and title
    plt.xlabel("Year")
    plt.ylabel("Number of Movies")
    plt.title("Number of Movies by Bechdel test score per Year")
    plt.legend(title="Bechdel score")

    # Show the plot
    plt.show()
    return df_unique_movies


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
    sns.histplot(movies.feminity_score, bins=18, color=color_G, alpha=0.5, log=True)
    plt.title("Distribution of Femininity Score Across Movies")
    plt.xlim(0,0.18)
    plt.xlabel("Femininity Score")
    plt.ylabel("Number of Movies")
    plt.show()


def visualize_masculinity_score_distribution(movies, style="darkgrid"):
    """
    Visualize the distribution of the femininity score across movies.

    Args:
    movies (pandas.DataFrame): A DataFrame containing movie data with a 'femininity_score' column.

    Returns:
    None
    """

    plt.figure(figsize=(8, 4))
    sns.set_style(style)
    sns.histplot(movies.masculinity_score, bins=30, color=color_G, alpha=0.5, log=True)
    plt.title("Distribution of Masculinity Score Across Movies")
    plt.xlim(0,0.18)
    plt.xlabel("Masculinity Score")
    plt.ylabel("Number of Movies")
    plt.show()
    
def visualize_wordcloud_roles(actor_with_role):
    role_women = actor_with_role.loc[actor_with_role['actor_gender']=='F'].copy(deep=True)
    role_men = actor_with_role.loc[actor_with_role['actor_gender']=='M'].copy(deep=True)

    women_counts = role_women['role'].value_counts().head(5)
    men_counts = role_men['role'].value_counts().head(5)

    # Create WordClouds for women and men
    women_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Oranges').generate_from_frequencies(women_counts)
    men_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Oranges').generate_from_frequencies(men_counts)

    # Plotting the WordClouds
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(women_wordcloud, interpolation='bilinear')
    plt.title('Women Main Characters', color='black', fontsize=16)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(men_wordcloud, interpolation='bilinear')
    plt.title('Men Main Characters', color='black', fontsize=16)
    plt.axis('off')

    plt.show()
    return


    

def visualize_director_gender_proportion(movies, style="darkgrid", year_range=5):
    """
    Visualize the average proportion of male and female directors over a specified range of years.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing information about movies and directors.
    style (str): Style of the plot. Default is "darkgrid".
    year_range (int): The range of years to average over. Default is 5.

    Returns:
    None
    """

    min_year = movies['year'].min()
    max_year = movies['year'].max()
    year_mod = (max_year - min_year) % year_range
    if year_mod != 0:
        movies = movies[movies['year'] > min_year + year_mod]

    # Group by year range
    movies['year_range'] = np.floor((movies['year'] - min_year) / year_range) * year_range + min_year
    gender_counts = movies.groupby(['year_range', 'director_gender']).size().unstack(fill_value=0)

    plt.figure(figsize=(12, 6))
    sns.set_style(style)

    # Adjust the order of plotting to put female on top
    gender_counts = gender_counts[['M', 'F']]

    # Plot stacked bar plot for counts
    gender_counts.plot(kind='bar', stacked=True, color={"F": color_F, "M": color_M}, ax=plt.gca())

    plt.title(f"Count of Movies with Male and Female Directors Over {year_range}-Year Periods")
    plt.xlabel(f"Year Ranges Starting from {min_year}")
    plt.ylabel("Count of Movies")
    plt.legend(title='Director Gender', labels=['Male', 'Female'])
    plt.show()
    return


def visualize_producer_gender_proportion(movies, style="darkgrid", year_range=5):
    """
    Visualize the average proportion of male and female directors over a specified range of years.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing information about movies and directors.
    style (str): Style of the plot. Default is "darkgrid".
    year_range (int): The range of years to average over. Default is 5.

    Returns:
    None
    """

    # Adjust DataFrame to ensure the number of years is divisible by year_range
    min_year = movies['year'].min()
    max_year = movies['year'].max()
    year_mod = (max_year - min_year) % year_range
    if year_mod != 0:
        movies = movies[movies['year'] > min_year + year_mod]

    # Group by year range
    movies['year_range'] = np.floor((movies['year'] - min_year) / year_range) * year_range + min_year
    gender_counts = movies.groupby(['year_range', 'producer_gender']).size().unstack(fill_value=0)

    plt.figure(figsize=(12, 6))
    sns.set_style(style)

    # Adjust the order of plotting to put female on top
    gender_counts = gender_counts[['M', 'F']]

    # Plot stacked bar plot for counts
    gender_counts.plot(kind='bar', stacked=True, color={"F": color_F, "M": color_M}, ax=plt.gca())

    plt.title(f"Count of Movies with Male and Female Producer Over {year_range}-Year Periods")
    plt.xlabel(f"Year Ranges Starting from {min_year}")
    plt.ylabel("Count of Movies")
    plt.legend(title='Producer Gender', labels=['Male', 'Female'])
    plt.show()
    return


def visualize_type_of_role_credited(movies_import, gender='B'):
    """
    Visualize the distribution of the type of role credited (Credited by name, job title...) across movies.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing information about movies and actors.
    style (str): Style of the plot. Default is "darkgrid".
    gender (str): M, F or B (both)
    Returns:
    None
    """

    fig, ax = plt.subplots(figsize=(9, 6))

    if gender == 'B':
        movies = movies_import.copy()
        plt.title('Distribution of the type of role credited across movies for both gender')
    else:
        movies = movies_import[movies_import['actor_gender']==gender]
        if gender == 'M':
            plt.title('Distribution of the type of role credited across movies for actors')
        elif gender == 'F':
            plt.title('Distribution of the type of role credited across movies for actresses')
    # Group by decade
    movies['decade'] = (movies['movie_release_date'].dt.year // 10) * 10
    df_cummu = movies.groupby(movies.decade)['role_cat'].value_counts(normalize=True).reset_index()

    df = df_cummu.set_index('decade', inplace=True)
    _ = df_cummu.pivot(columns='role_cat', values='proportion').plot.area(ax=ax)

    _ = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()
    return
    

def visualize_proportion_gender_credited(movies):
    """
    Visualize the distribution uncredited/credited roles by gender.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing information about movies and actors.
    style (str): Style of the plot. Default is "darkgrid".
    Returns:
    None
    """

    F_credited = movies[(movies['credited']==True) & (movies['actor_gender']=='F')]['actor_name'].count()
    F_uncredited = movies[(movies['credited']==False) & (movies['actor_gender']=='F')]['actor_name'].count()
    M_credited = movies[(movies['credited']==True) & (movies['actor_gender']=='M')]['actor_name'].count()
    M_uncredited = movies[(movies['credited']==False) & (movies['actor_gender']=='M')]['actor_name'].count()

    fig, ax = plt.subplots()
    ax.pie([F_credited,F_uncredited,M_credited,M_uncredited], labels=["F_credited","F_uncredited","M_credited","M_uncredited"])
    plt.show()
    return
    
def visualize_proportion_specific_gender_credited(movies, gender = 'F'):
    """
    Visualize the distribution uncredited/credited roles by gender.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing information about movies and actors.
    style (str): Style of the plot. Default is "darkgrid".
    Returns:
    None
    """

    X_credited = movies[(movies['credited']==True) & (movies['actor_gender']==gender)]['actor_name'].count()
    X_uncredited = movies[(movies['credited']==False) & (movies['actor_gender']==gender)]['actor_name'].count()
    percentage_uncredited = round(X_uncredited/(X_credited+X_uncredited)*100,2)
    fig, ax = plt.subplots()
    gender_job = 'actors' if gender=='M' else 'actresses'
    plt.title(f'{percentage_uncredited} of actors are uncredited'.format(percentage_uncredited=percentage_uncredited))
    ax.pie([X_credited,X_uncredited], labels=[f"{gender} credited",f"{gender} uncredited"])
    plt.show()
    return
    
def visualize_un_credited_stacked(movies_import, gender='B'):
    """
    Visualize the distribution of the type of role credited (Credited by name, job title...) across movies.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing information about movies and actors.
    style (str): Style of the plot. Default is "darkgrid".
    gender (str): M, F or B (both)
    Returns:
    None
    """

    fig, ax = plt.subplots(figsize=(9, 6))

    if gender == 'B':
        movies = movies_import.copy()
        plt.title('Distribution of whether a role was credited or not')
    else:
        movies = movies_import[movies_import['actor_gender']==gender]
        if gender == 'M':
            plt.title('Distribution of whether a role was credited or not for an actor')
        elif gender == 'F':
            plt.title('Distribution of whether a role was credited or not for actresses')
    # Group by decade
    movies['decade'] = (movies['movie_release_date'].dt.year // 10) * 10
    df_cummu = movies.groupby(movies.decade)['credited'].value_counts(normalize=True).reset_index()

    df = df_cummu.set_index('decade', inplace=True)
    _ = df_cummu.pivot(columns='credited', values='proportion').plot.area(ax=ax)

    _ = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()
    return

def visualize_wordcloud_job_roles(movies):
    role_women = movies.loc[movies['actor_gender']=='F'].copy(deep=True)
    role_women = role_women[role_women['role_cat']=='JOB']
    role_women = role_women.drop(role_women[role_women['role'].str.contains('Self')].index)
    role_women = role_women.drop(role_women[role_women['role'].str.contains('Narrator')].index)
    role_women = role_women.drop(role_women[role_women['role'].str.contains('voice')].index)
    role_men = movies.loc[movies['actor_gender']=='M'].copy(deep=True)
    role_men = role_men[role_men['role_cat']=='JOB']
    role_men = role_men.drop(role_men[role_men['role'].str.contains('Self')].index)
    role_men = role_men.drop(role_men[role_men['role'].str.contains('Narrator')].index)
    role_men = role_men.drop(role_men[role_men['role'].str.contains('voice')].index)


    women_counts = role_women['role'].value_counts().head(10)
    men_counts = role_men['role'].value_counts().head(10)
    # Create WordClouds for women and men
    women_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Oranges').generate_from_frequencies(women_counts)
    men_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Oranges').generate_from_frequencies(men_counts)

    # Plotting the WordClouds
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(women_wordcloud, interpolation='bilinear')
    plt.title('Women Jobs as a Characters', color='black', fontsize=16)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(men_wordcloud, interpolation='bilinear')
    plt.title('Men Jobs as a Characters', color='black', fontsize=16)
    plt.axis('off')

    plt.show()
    return

def visualize_wordcloud_r2j_roles(movies):
    role_women = movies.loc[movies['actor_gender']=='F'].copy(deep=True)
    role_women = role_women[role_women['role_cat']=='ROLE_TO_JOB']
    role_women = role_women.drop(role_women[role_women['role'].str.contains('Self')].index)
    role_women = role_women.drop(role_women[role_women['role'].str.contains('Narrator')].index)
    role_women = role_women.drop(role_women[role_women['role'].str.contains('voice')].index)
    role_women['role'] = role_women['role'].apply(lambda x: x.split("'s")[-1])
    role_men = movies.loc[movies['actor_gender']=='M'].copy(deep=True)
    role_men = role_men[role_men['role_cat']=='ROLE_TO_JOB']
    role_men = role_men.drop(role_men[role_men['role'].str.contains('Self')].index)
    role_men = role_men.drop(role_men[role_men['role'].str.contains('Narrator')].index)
    role_men = role_men.drop(role_men[role_men['role'].str.contains('voice')].index)
    role_men['role'] = role_men['role'].apply(lambda x: x.split("'s")[-1])


    women_counts = role_women['role'].value_counts().head(10)
    men_counts = role_men['role'].value_counts().head(10)
    # Create WordClouds for women and men
    women_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Oranges').generate_from_frequencies(women_counts)
    men_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Oranges').generate_from_frequencies(men_counts)

    # Plotting the WordClouds
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(women_wordcloud, interpolation='bilinear')
    plt.title('Women Supporting Jobs as Characters', color='black', fontsize=16)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(men_wordcloud, interpolation='bilinear')
    plt.title('Men Supporting Jobs as Characters', color='black', fontsize=16)
    plt.axis('off')

    plt.show()
    return