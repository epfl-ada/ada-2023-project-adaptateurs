import pandas as pd
import numpy as np
import gender_guesser.detector as gender
import json
from ast import literal_eval


def clean_movie_df(movie_df):
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
    bechdel_df = bechdel_df.drop(columns=["id"])
    bechdel_df = bechdel_df.rename(
        columns={"title": "movie_title", "rating": "bechdel_rating"}
    )
    return bechdel_df


def clean_credit_df(credit_df, meta_df):
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

    d = gender.Detector()

    def getgender(name):
        res = d.get_gender(name)
        if res == "male" or res == "mostly_male":
            return "M"
        elif res == "female" or res == "mostly_female":
            return "F"
        else:
            return np.nan

    print("Before using genderguesser:")
    print(
        f"Percentage of movies with a director's name that could not be gendered: {round((len(credit_df[credit_df['director_gender'] == 0]) + credit_df['director_gender'].isna().sum()) / len(credit_df) * 100, 2)}%"
    )
    print(
        f"Percentage of movies with a producer's name that could not be gendered: {round((len(credit_df[credit_df['producer_gender'] == 0]) + credit_df['producer_gender'].isna().sum()) / len(credit_df) * 100, 2)}%"
    )
    print(
        f"Percentage of movies with a writer's name that could not be gendered:   {round((len(credit_df[credit_df['writer_gender'] == 0]) + credit_df['writer_gender'].isna().sum()) / len(credit_df) * 100, 2)}%"
    )

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
    movies["actor_age_at_movie_release"] = movies["actor_age_at_movie_release"].apply(
        lambda age: np.nan if age < 0 else age
    )
    movies = movies.query("year >= 1912 & year <= 2012")
    return movies


def clean_remove_outlier(df, method="z-score", threshold=0, name=""):
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
    return df.drop(columns=to_drop, axis=1)
