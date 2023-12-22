from scipy import stats

def ttest_gender_correlation(movies):
    """
    Perform simplified t-test analysis to see if the correlation between the number of female actors and the gender of the producer, director, or both is significant.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing information about movies, directors, producers, and actors.

    Returns:
    dict: A dictionary of p-values for the t-tests
    """

    # Calculate the count of actors for each movie
    actor_counts = movies.groupby('wikiID').size()
    female_actor_counts = movies[movies['actor_gender'] == 'F'].groupby('wikiID').size()

    # Calculate the proportion of female actors
    female_actor_proportion = (female_actor_counts / actor_counts).rename('female_actor_proportion')

    # Create binary columns for the presence of a female director and a female producer
    movies['has_female_director'] = movies['director_gender'] == 'F'
    movies['has_female_producer'] = movies['producer_gender'] == 'F'
    
    # Merge the proportion of female actors into the movies DataFrame
    movies = movies.merge(female_actor_proportion, on='wikiID', how='left')
    
    # Drop duplicates since there can be multiple actors per movie
    movies.drop_duplicates(subset='wikiID', inplace=True)

    # Define the groups for comparison
    no_female_director_producer = movies[(~movies['has_female_director'] & ~movies['has_female_producer'])]['female_actor_proportion']
    female_producer_only = movies[(~movies['has_female_director'] & movies['has_female_producer'])]['female_actor_proportion']
    female_director_only = movies[(movies['has_female_director'] & ~movies['has_female_producer'])]['female_actor_proportion']
    both_female_director_and_producer = movies[(movies['has_female_director'] & movies['has_female_producer'])]['female_actor_proportion']

    # Perform t-tests
    results = {
        "No Female Director/Producer vs Female Producer Only": stats.ttest_ind(no_female_director_producer.dropna(), female_producer_only.dropna(), equal_var=False)[1],
        "No Female Director/Producer vs Female Director Only": stats.ttest_ind(no_female_director_producer.dropna(), female_director_only.dropna(), equal_var=False)[1],
        "No Female Director/Producer vs Both Female Director and Producer": stats.ttest_ind(no_female_director_producer.dropna(), both_female_director_and_producer.dropna(), equal_var=False)[1]
    }

    return results

