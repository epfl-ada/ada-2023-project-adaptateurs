# Women's Representation Analysis in the Film Industry

<div>
  <div><b>Team</b>: Adaptateurs</div>
  <div><b>Supervisor</b>: <a href="https://people.epfl.ch/silin.gao?lang=en"> Silin Gao (PhD) </a> </div>
</div>

<span align="center">

<br>

**Lena Vogel,  François Dumoncel,  Aymeric Bacuet,  Kenji Tetard,  Naël Dillenbourg**

*École Polytechnique Fédérale de Lausanne*

<br> 

</span>


## How to run our Notebooks  
<details>
<summary> Tutorial </summary> 

Install necessary package using 

```console
$ pip install -r requirements.txt
```

Decompress data just after cloning this repo
1. CMU original dataset 
```console
$ cd data/ && tar -xvzf *.tar.gz
```

1. TMDB 
```console
$ cd external && unzip movies.zip && mv "Movies Dataset" Movies
```

Or simply decompress archive from file system. You can also directly use the pre-processed pickle file in `data/Processed`
</details>



## Reposository structure

```
├─── data
│    ├─── external
│           └─── movies 
│    ├─── Processed
│    ├─── MovieSummaries
├─── analysis 
│      └─── visualizer.py
├─── preprocessing
       ├─── loader.py
       ├─── cleaner.py
       └─── merger.py
├─── P2.ipynb
├─── requirements.txt
└─── README.md
```


## Table of Contents 📕

<p>
  <a href="#abstract-"> 1. Abstract</a> 
  <br>
  <a href="#research-questions-">2. Research Questions</a> 
  <br>
  <a href="#datasets-">3. Additional Datasets</a> 
  <br>
  <a href="#methods-">4. Methods</a> 
  <br>
  <a href="#proposed-timeline-">5. Timeline</a> 
  <br>
  <a href="#team-organization-">6. Team Organization</a>
</p>





## Abstract 📌

<span align="justify">

This analysis aims to discover the portrayal of women in the film industry by analyzing a rich dataset encompassing aspects from actor profiles to production details. We aim to uncover gender disparities by examining the roles and ages of actresses at the time of a movie's release, and the influence of female directors and producers on a film's success and reception. In addition to that, we will use another dataset: the movies that passed the Bechdel test or not. This test analyzes whether a movie passes the 3 following conditions: 
a) The movie has to have at least two women in it, b) who talk to each other, c) about something other than a man.

While it gives very basic insights on the representativity of women we want to inspect how the result of the Bechdel Test can influence the success of a movie. We will examine the impact of female directors and producers on a film's popularity and box-office performance. Additionally, we will utilize Natural Language Processing technique to assess gender stereotypes within film summaries.

</span>

## Research Questions 🔍
Our research is structured around several key questions:

- **Trends Over Time:** How have trends in passing the Bechdel and Mako-Mori tests evolved over the past decades?

- **Commercial Success and Critical Acclaim:** Is there a notable correlation between a movie passing these tests and its commercial success or critical acclaim?

- **Genre-Specific Representation:** What genres are more likely to pass or fail these tests, and what does this imply about the representation within specific genres?

- **Cultural and International Perspectives:** How does women's representation in films from different countries or cultures measure up against these tests?

- **Narrative Roles and Character Archetypes:** What common narrative roles or character archetypes for women are observed in films that pass these tests compared to those that don't?

- **Gender Diversity in Film Production:** Are films that pass the Bechdel and Mako-Mori tests also likely to exhibit more gender diversity among their writers, directors, and producers?

- **Influence of Female Characters:** Does the presence of leading or strong supporting female characters correlate with passing these tests?

- **Independent Films vs. Major Studio Productions:** How do independent films compare with major studio productions in terms of passing these tests?

- **Public Discourse and Influence:** To what extent do films that pass these tests influence public discourse on gender representation?

- **Challenging Gender Stereotypes:** Are movies that pass these tests more inclined to feature women in non-traditional roles or challenge gender stereotypes?

- **Impact of Film Ratings:** How does a film’s age rating affect its likelihood of passing these tests?

- **Role of Film Festivals and Awards:** What is the role of film festivals and awards in promoting films that pass these tests?

- **Audience Perceptions and Demands:** How have audience perceptions and demands influenced the number of films passing these tests over time?

- **Predictive Analysis:** Can data from these tests predict future trends in women's cinematic representation?

- **Real-world Impact:** What real-world impact do films that pass these tests have on issues of gender equality and female empowerment?

- How have the trends in passing the Bechdel and test changed over the past few decades?
- Is there a correlation between a movie passing this test and its commercial success or critical acclaim?
- What genres are more likely to pass or fail these tests, and what does this suggest about genre-specific representation?
- How does the representation of women in films from different countries or cultures fare when applied to these tests?
- Are films that pass the Bechdel test also likely to have more gender diversity among their writers, directors, and producers?
- How do independent films compare to major studio productions in terms of passing these tests?
- How does the age rating of a film (G, PG, PG-13, R) impact its likelihood of passing these tests?
- Can the data from the Bechdel test be used to predict future trends in women's cinematic representation?
- What impact, if any, do films that pass these tests have on the real-world issues of gender equality and female empowerment?

## Datasets 📊

In this analysis, we will use three different datasets

1. The original and provided CMU dataset (http://www.cs.cmu.edu/~ark/personas/)
2. The TMDB Dataset:
    - How we got it: directly downloaded from [The Movies Dataset on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) (TMDB)
    - Contains metadata on over 45,000 movies. 26 million ratings (in %) from over 270'000 users.
    - Will be used to define a metric for the success of a movie.
3. The Bechdel Dataset:
    - How we got it: Obtained via bechdeltest.com API
    - Contains the Bechdel rating score (0,1,2 or 3) for more than 10'000 movies
    - See `preprocessing/loader.py/load_bechdel_dataset()`

## Methodology 💡
Our analysis will employ a multifaceted approach:

- **Statistical Analysis:** We'll conduct thorough statistical analysis to uncover trends and correlations.

<span align="justify">

**Original Dataset**: Initial cleaning has been conducted.

**Bechdel Dataset**: The Bechdel dataset boasts the advantage of minimal missing values, which simplifies the cleaning process significantly.

**TMDB Dataset**: Our focus with the TMDB dataset is to extract data on the crew members behind the scenes, particularly their gender. We mitigate the impact of missing values by employing the genderguesser package, which allows us to infer genders from first names and reduce the volume of incomplete data.

</span>

### Analysis of Women's place
Analyzing the representation of women in movies through various datasets, including cast members genders, movie success, budget, and Bechdel test results, provides a comprehensive understanding of gender representation in the film industry: in cast and crew members as well as in the movies stories.

By correlating cast members genders with movie success and budgets, we can identify trends and biases in the industry. For instance, examining whether films with a higher proportion of female cast members are as likely to benefit from big or small budgets or have significant box office entries.

Indeed, we consider that women's representation in media such as movies is crutial for advances in feminist struggles, because being able to rely on fictional characters is essential because it allows us to explore and understand complex human experiences and emotions in a safe and imaginative context. 
In order to analyze how well women are represented in movies we used the Bechdel test. It was originally invented in 1985 in a comic strip by Alison Bechdel and tests whether a movie passes the 3 following conditions: 
a) The movie has to have at least two women in it, b) who talk to each other, c) about something other than a man. This results in a score between 0 and 3, depending on the number of tests the movie passes.
It is considered a basic standard of women's representation in movies and also includes some women's typical stereotypes present in stories such as the Smurfette principle, which designs the fact that the woman is the exception and exists only in reference to the men.

In conclusion, by conducting a thorough analysis of women's representation in movies using these datasets we hope to provide valuable insights into the passed and future progress in achieving gender equality in the film industry.

- **Natural Language Processing:** NLP techniques will be applied to film summaries to detect and analyze gender stereotypes.

- **Comparative Analysis:** We'll compare genres, cultures, and production types (independent vs. major studios) in terms of gender representation.

- **Temporal Analysis:** Examining how trends have evolved over time, we'll use this data to potentially forecast future trends.

- **Impact Assessment:** We'll assess the real-world impact of films that pass the Bechdel and Mako-Mori tests, particularly in terms of influencing public discourse and contributing to gender equality movements.

## Proposed Timeline 📆

```C
├── Week 8  - Preprocessing
│  
├── Week 9  - Analysis and pipeline
│  
├── Week 10 - Homework II
│  
├── Week 11 - Data Story and fine-grained analysis through Bechdel test and cie
│  
├── Week 12 - Data Story and NLP on plot summaries 
│    
├── Week 13 - Start the website (Github Pages)
│  
├── Week 14 - Finalization
```

## Team Organization ⚙️


| Name            | Email                                | Task                     |
|-----------------|--------------------------------------|--------------------------|
| F. Dumoncel     | `francois.dumoncel-kessler@epfl.ch`  | Preprocessing and README |
| K. Tetard       | `kenji.tetard@epfl.ch`               |  -  |
| L. Vogel        | `lena.vogel@epfl.ch`                 | README ...  |
| N. Dillenbourg  | `nael.dillenbourg@epfl.ch`           | -   |
| A. Bacuet       | `aymeric.bacuet@epfl.ch`             | Exploratory and data Analysis   |
