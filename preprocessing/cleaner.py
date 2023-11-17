import pandas as pd
import numpy as np
import gender_guesser.detector as gender
import json
from ast import literal_eval
import wiki_request


def clean_movie_df(movie_df):
    """
    Clean the movie dataframe by converting the release date to a real datetime, dropping movies with no release date,
    extracting the year from the release date, and converting the fbids for genres, languages, and countries to lists.

    :param movie_df: pandas DataFrame containing movie information
    :return: cleaned pandas DataFrame
    """

    # Convert release date to real datetime
    movie_df["movie_release_date"] = pd.to_datetime(
        movie_df["movie_release_date"], errors="coerce"
    )

    # Drop movies with no release date
    movie_df.dropna(subset=["movie_release_date"], inplace=True)
    movie_df["year"] = movie_df["movie_release_date"].dt.year.astype(int)

    def get_values_from_dict(dict_: str) -> list:
        return list(json.loads(dict_).values())

    movie_df["fbid_genres"] = movie_df["fbid_genres"].apply(get_values_from_dict)
    movie_df["fbid_languages"] = movie_df["fbid_languages"].apply(get_values_from_dict)
    movie_df["fbid_countries"] = movie_df["fbid_countries"].apply(get_values_from_dict)
    return movie_df


def clean_bechdel_df(bechdel_df):
    """
    Cleans the given Bechdel dataset by dropping the 'id' column and renaming the 'title' and 'rating' columns.

    Args:
    - bechdel_df: pandas DataFrame containing the Bechdel dataset

    Returns:
    - pandas DataFrame: cleaned Bechdel dataset
    """
    bechdel_df = bechdel_df.drop(columns=["id"])
    bechdel_df = bechdel_df.rename(
        columns={"title": "movie_title", "rating": "bechdel_rating"}
    )
    return bechdel_df


def clean_credit_df(credit_df, meta_df, scraping=False):
    """
    Clean the credit dataframe by extracting the director, producer and writer names and genders.

    Parameters:
    credit_df (pandas.DataFrame): The credit dataframe to clean.
    meta_df (pandas.DataFrame): The metadata dataframe containing the imdb_id.

    Returns:
    pandas.DataFrame: The cleaned credit dataframe.
    """

    credit_df["imdbid"] = meta_df["imdb_id"]
    credit_df.dropna(subset=["imdbid"], inplace=True)
    credit_df.imdbid = credit_df.imdbid.apply(lambda x: x[2:])
    credit_df.drop(["cast"], axis=1, inplace=True)
    credit_df["crew"].fillna("[]", inplace=True)
    credit_df["crew"] = credit_df["crew"].apply(literal_eval)

    def get_job(namedict, target_job, target):
        try:
            name = [x[target] for x in namedict if x["job"] == target_job][0]
        except:
            name = np.nan
        return name

    credit_df["director"] = credit_df["crew"].apply(
        lambda x: get_job(x, "Director", "name")
    )
    credit_df["director_gender"] = credit_df["crew"].apply(
        lambda x: get_job(x, "Director", "gender")
    )
    credit_df["producer"] = credit_df["crew"].apply(
        lambda x: get_job(x, "Producer", "name")
    )
    credit_df["producer_gender"] = credit_df["crew"].apply(
        lambda x: get_job(x, "Producer", "gender")
    )
    credit_df["writer"] = credit_df["crew"].apply(
        lambda x: get_job(x, "Writer", "name")
    )
    credit_df["writer_gender"] = credit_df["crew"].apply(
        lambda x: get_job(x, "Writer", "gender")
    )

    print("Before using wikipedia and genderguesser:")
    print(
        f"Percentage of movies with a director's name that could not be gendered: {round((len(credit_df[credit_df['director_gender'] == 0]) + credit_df['director_gender'].isna().sum()) / len(credit_df) * 100, 2)}%"
    )
    print(
        f"Percentage of movies with a producer's name that could not be gendered: {round((len(credit_df[credit_df['producer_gender'] == 0]) + credit_df['producer_gender'].isna().sum()) / len(credit_df) * 100, 2)}%"
    )
    print(
        f"Percentage of movies with a writer's name that could not be gendered:   {round((len(credit_df[credit_df['writer_gender'] == 0]) + credit_df['writer_gender'].isna().sum()) / len(credit_df) * 100, 2)}%"
    )

    if scraping:
        credit_df.loc[
            credit_df["director_gender"] == 0.0, "director_gender"
        ] = credit_df.loc[
            credit_df["director_gender"] == 0.0, ["director", "imdbid"]
        ].apply(
            lambda x: wiki_request.get_gender_id(
                wiki_request.get_gender_imdb(x["imdbid"], "director", x["director"])
            ),
            axis=1,
        )

        credit_df.loc[
            credit_df["producer_gender"] == 0.0, "producer_gender"
        ] = credit_df.loc[
            credit_df["producer_gender"] == 0.0, ["producer", "imdbid"]
        ].apply(
            lambda x: wiki_request.get_gender_id(
                wiki_request.get_gender_imdb(x["imdbid"], "producer", x["producer"])
            ),
            axis=1,
        )
        credit_df.loc[
            credit_df["writer_gender"] == 0.0, "writer_gender"
        ] = credit_df.loc[
            credit_df["writer_gender"] == 0.0, ["writer", "imdbid"]
        ].apply(
            lambda x: wiki_request.get_gender_id(
                wiki_request.get_gender_imdb(x["imdbid"], "writer", x["writer"])
            ),
            axis=1,
        )

        print("\nAfter using wikipedia:")
        print(
            f"Percentage of movies with a director's name that could not be gendered: {round(credit_df['director_gender'].isna().sum() / len(credit_df) * 100, 2)}%"
        )
        print(
            f"Percentage of movies with a producer's name that could not be gendered: {round(credit_df['producer_gender'].isna().sum() / len(credit_df) * 100, 2)}%"
        )
        print(
            f"Percentage of movies with a writer's name that could not be gendered:   {round(credit_df['writer_gender'].isna().sum() / len(credit_df) * 100, 2)}%"
        )

    d = gender.Detector()

    def getgender(name):
        res = d.get_gender(name)
        if res == "male" or res == "mostly_male":
            return "M"
        elif res == "female" or res == "mostly_female":
            return "F"
        else:
            return np.nan

    credit_df.loc[
        credit_df["director_gender"] == 0.0, "director_gender"
    ] = credit_df.loc[credit_df["director_gender"] == 0.0, "director"].apply(
        lambda x: getgender(x.split(" ")[0])
    )
    credit_df.loc[
        credit_df["producer_gender"] == 0.0, "producer_gender"
    ] = credit_df.loc[credit_df["producer_gender"] == 0.0, "producer"].apply(
        lambda x: getgender(x.split(" ")[0])
    )
    credit_df.loc[credit_df["writer_gender"] == 0.0, "writer_gender"] = credit_df.loc[
        credit_df["writer_gender"] == 0.0, "writer"
    ].apply(lambda x: getgender(x.split(" ")[0]))

    credit_df.loc[credit_df["director_gender"] == 2.0, "director_gender"] = "M"
    credit_df.loc[credit_df["producer_gender"] == 2.0, "producer_gender"] = "M"
    credit_df.loc[credit_df["writer_gender"] == 2.0, "writer_gender"] = "M"
    credit_df.loc[credit_df["director_gender"] == 1.0, "director_gender"] = "F"
    credit_df.loc[credit_df["producer_gender"] == 1.0, "producer_gender"] = "F"
    credit_df.loc[credit_df["writer_gender"] == 1.0, "writer_gender"] = "F"

    print("\nAfter using genderguesser:")
    print(
        f"Percentage of movies with a director's name that could not be gendered: {round(credit_df['director_gender'].isna().sum() / len(credit_df) * 100, 2)}%"
    )
    print(
        f"Percentage of movies with a producer's name that could not be gendered: {round(credit_df['producer_gender'].isna().sum() / len(credit_df) * 100, 2)}%"
    )
    print(
        f"Percentage of movies with a writer's name that could not be gendered:   {round(credit_df['writer_gender'].isna().sum() / len(credit_df) * 100, 2)}%"
    )

    credit_df.drop(["crew"], axis=1, inplace=True)
    credit_df.drop(["id"], axis=1, inplace=True)

    return credit_df


def clean_metadata_df(meta_df):
    meta_df["release_date"] = pd.to_datetime(meta_df["release_date"], errors="coerce")
    meta_df["year"] = meta_df["release_date"].dt.year
    meta_df = meta_df.rename(
        columns={
            "imdb_id": "imdbid",
            "title": "movie_title",
            "release_date": "movie_release_date",
        }
    )
    meta_df = meta_df[
        ["budget", "popularity", "vote_average", "imdbid", "movie_title", "id", "year"]
    ]
    meta_df["imdbid"] = meta_df["imdbid"].str.replace("tt", "")
    return meta_df


def clean_movies_ranges(movies):
    """
    Clean the movies dataframe by removing rows with invalid actor age and keeping only movies released between 1912 and 2012.

    Args:
    movies (pandas.DataFrame): The dataframe containing the movies data.

    Returns:
    pandas.DataFrame: The cleaned dataframe.
    """

    movies["actor_age_at_movie_release"] = movies["actor_age_at_movie_release"].apply(
        lambda age: np.nan if age < 0 else age
    )
    movies = movies.query("year >= 1912 & year <= 2012")
    return movies


def clean_remove_outlier(df, method="z-score", threshold=0, name=""):
    """
    Remove outliers from a pandas DataFrame column using either z-score or quantile method.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the column to clean.
    method (str): The method to use for outlier detection. Either "z-score" or "quantile". Default is "z-score".
    threshold (float): The threshold value to use for outlier detection. Default is 0.
    name (str): The name of the column to clean. Default is "".

    Returns:
    pandas.DataFrame: The cleaned DataFrame.
    """

    from scipy import stats

    if method == "z-score":
        z_scores = stats.zscore(df[name])
        z_score_threshold = threshold
        df = df[abs(z_scores) < z_score_threshold]

    elif method == "quantile":
        Q1 = df[name].quantile(0.25)
        Q3 = df[name].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[name] >= lower_bound) & (df[name] <= upper_bound)]

    return df


def clean_missing_values_by_dropping(df, to_drop=[]):
    """
    This function takes a pandas dataframe and a list of columns to drop and returns a new dataframe with the specified columns dropped.

    Parameters:
    df (pandas.DataFrame): The dataframe to clean.
    to_drop (list): A list of column names to drop from the dataframe.

    Returns:
    pandas.DataFrame: A new dataframe with the specified columns dropped.
    """
    return df.drop(columns=to_drop, axis=1)
