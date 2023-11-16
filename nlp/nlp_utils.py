import re
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

stop_words = stopwords.words("english")

female_keywords = []
with open("./nlp/female_keywords.txt", "r") as file:
    female_keywords = [line.strip() for line in file]

male_keywords = []
with open("./nlp/male_keywords.txt", "r") as file:
    male_keywords = [line.strip() for line in file]


def clean_text(text: str):
    """
    This function takes a movie summary (str) as input and performs several cleaning operations on it, including:
    - converting all text to lowercase
    - removing any text within square brackets (e.g. citations)
    - removing all punctuation
    - removing any words containing numbers
    - removing any stop words (common words like "the", "and", etc.)

    Args:
    - text (str): the summary to be cleaned

    Returns:
    - str: the cleaned text
    """

    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\w*\d\w*", "", text)

    return " ".join(word for word in text.split() if word not in stop_words)


def nlp_tokenize_summaries(df):
    """
    Tokenizes the summary column of a pandas dataframe and adds a new column with the tokens and another column with the length of the summary.

    Args:
    - df: pandas dataframe with a "summary" column

    Returns:
    - df: pandas dataframe with a new "tokens" column and a new "len" column
    """

    df["tokens"] = df["summary"].apply(word_tokenize)
    df["len"] = df["summary"].apply(lambda x: len(x))
    return df


def nlp_compute_feminity_score(nlp_summaries):
    """
    Compute the femininity score of a given set of summaries using a pre-defined list of female keywords.

    Args:
    - nlp_summaries: a pandas DataFrame containing the summaries to analyze

    Returns:
    - nlp_summaries: the input DataFrame with an additional column 'femininity_score' containing the computed score
    """

    vect = CountVectorizer(vocabulary=female_keywords)
    X = vect.fit_transform(nlp_summaries["summary"])
    word_counts = pd.DataFrame(X.toarray(), columns=vect.get_feature_names_out())
    nlp_summaries = pd.concat([nlp_summaries, word_counts], axis=1)
    nlp_summaries["feminity_score"] = (
        1 / nlp_summaries["len"] * nlp_summaries[female_keywords].sum(axis=1)
    )
    return nlp_summaries


def nlp_compute_masculinity_score(nlp_summaries):
    """
    Compute the femininity score of a given set of summaries using a pre-defined list of female keywords.

    Args:
    - nlp_summaries: a pandas DataFrame containing the summaries to analyze

    Returns:
    - nlp_summaries: the input DataFrame with an additional column 'femininity_score' containing the computed score
    """

    vect = CountVectorizer(vocabulary=male_keywords)
    X = vect.fit_transform(nlp_summaries["summary"])
    word_counts = pd.DataFrame(X.toarray(), columns=vect.get_feature_names_out())
    nlp_summaries = pd.concat([nlp_summaries, word_counts], axis=1)
    nlp_summaries["masculinity_score"] = (
        1 / nlp_summaries["len"] * nlp_summaries[male_keywords].sum(axis=1)
    )
    return nlp_summaries
