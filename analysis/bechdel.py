import pandas as pd


def get_number_of_actors(data):
    actors_per_film = data.groupby(0).size()
    data["Total_Actors_Per_Film"] = data[0].map(actors_per_film)

    male_actors_per_film = data[data[5] == "M"].groupby(0).size()
    female_actors_per_film = data[data[5] == "F"].groupby(0).size()

    data["Male_Actors_Per_Film"] = data[0].map(male_actors_per_film)
    data["Female_Actors_Per_Film"] = data[0].map(female_actors_per_film)

    data["Male_Actors_Per_Film"] = data["Male_Actors_Per_Film"].fillna(0).astype(int)
    data["Female_Actors_Per_Film"] = (
        data["Female_Actors_Per_Film"].fillna(0).astype(int)
    )

    # Display the first few rows of the updated dataset
    data = data[
        [1, "Total_Actors_Per_Film", "Male_Actors_Per_Film", "Female_Actors_Per_Film"]
    ]
    return data
