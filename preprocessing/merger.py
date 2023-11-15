import pandas as pd


def merge_with_char(movie_df, char_df):
    """
    Fusionne deux dataframes sur les colonnes "wikiID" et "freebaseID".

    Args:
    movie_df (pandas.DataFrame): Dataframe contenant les informations des films.
    char_df (pandas.DataFrame): Dataframe contenant les informations des personnages.

    Returns:
    pandas.DataFrame: Dataframe contenant les informations fusionnées des films et des personnages.
    """

    return pd.merge(
        movie_df,
        char_df.drop(columns="movie_release_date"),
        on=["wikiID", "freebaseID"],
        how="inner",
    )


def merge_with_summaries(merged_df, summaries_df):
    """
    Merge two dataframes on the 'wikiID' column using an inner join.

    Parameters:
    merged_df (pandas.DataFrame): The first dataframe to merge.
    summaries_df (pandas.DataFrame): The second dataframe to merge.

    Returns:
    pandas.DataFrame: The merged dataframe.
    """

    return pd.merge(merged_df, summaries_df, on=["wikiID"], how="inner")


def merge_with_bechdel(merged_df, bechdel_df):
    """
    Merge two dataframes on the columns "movie_title" and "year" using an inner join.

    Parameters:
    merged_df (pandas.DataFrame): The first dataframe to merge.
    bechdel_df (pandas.DataFrame): The second dataframe to merge.

    Returns:
    pandas.DataFrame: The merged dataframe.
    """
    return pd.merge(merged_df, bechdel_df, on=["movie_title", "year"], how="inner")


def merge_with_credits(merged_df, credits_df):
    """
    Merge two dataframes based on the 'imdbid' column using an inner join.

    Args:
    merged_df (pandas.DataFrame): The first dataframe to merge.
    credits_df (pandas.DataFrame): The second dataframe to merge.

    Returns:
    pandas.DataFrame: The merged dataframe.
    """
    return pd.merge(merged_df, credits_df, on=["imdbid"], how="inner")


def merge_with_metadata(merged_df, meta_df):
    """
    Merge two dataframes on the columns "movie_title" and "year".

    Parameters:
    merged_df (pandas.DataFrame): The first dataframe to merge.
    meta_df (pandas.DataFrame): The second dataframe to merge.

    Returns:
    pandas.DataFrame: The merged dataframe.
    """
    return pd.merge(merged_df, meta_df, on=["movie_title", "year"], how="inner")
