import pandas as pd


def merge_with_char(movie_df, char_df):
    return pd.merge(
        movie_df,
        char_df.drop(columns="movie_release_date"),
        on=["wikiID", "freebaseID"],
        how="inner",
    )


def merge_with_summaries(merged_df, summaries_df):
    return pd.merge(merged_df, summaries_df, on=["wikiID"], how="inner")


def merge_with_bechdel(merged_df, bechdel_df):
    return pd.merge(merged_df, bechdel_df, on=["movie_title", "year"], how="inner")


def merge_with_credits(merged_df, credits_df):
    return pd.merge(merged_df, credits_df, on=["imdbid"], how="inner")


def merge_with_metadata(merged_df, meta_df):
    return pd.merge(merged_df, meta_df, on=["movie_title", "year"], how="inner")
