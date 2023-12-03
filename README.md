# Analysis of Gender Representation in Global Cinema: The Evolution of Women's Roles and Audience Perceptions

<div>
  <div><b>Team</b>: Adaptateurs</div>
  <div><b>Supervisor</b>: <a href="https://people.epfl.ch/silin.gao?lang=en"> Silin Gao (PhD) </a> </div>
</div>

<span align="center">

<br>

**Léna Vogel,  François Dumoncel,  Aymeric Bacuet,  Kenji Tetard,  Naël Dillenbourg**

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

## Repository structure

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
├─── nlp  
       ├─── nlp_utils.py
       └─── female_keywords.txt
├─── P2.ipynb
├─── requirements.txt
├─── Makefile
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

Imagine a world where the stories we watch on the big screen truly reflect the diversity and complexity of the society we live in. This data analysis project embarks on a journey to explore the evolving landscape of women's representation in cinema, an arena that mirrors cultural shifts and societal attitudes. Our objective is to dissect the film industry's portrayal of women across the world, utilizing a comprehensive dataset that spans from actor profiles to intricate production details.

We want to unravel the layers of gender disparities by delving into the numbers, roles, and ages of actresses at the time of a movie's release. Furthermore, we're curious to understand the influence wielded by female directors and producers on a film's popularity and box-office success. An intriguing aspect of our study involves the Bechdel test - a litmus test for assessing female presence in movies. While seemingly simple, this test opens a window into the deeper narrative of female representativity in cinema.

Our quest is not just about numbers and statistics; it's a pursuit to comprehend how these elements shape the success and acceptance of films. We're also venturing into the realm of Natural Language Processing to dissect gender stereotypes within film summaries, offering a unique perspective on the narrative portrayal of women.

It's a quest to illuminate the untold stories and unseen perspectives in the world of cinema. By unraveling these patterns, we aspire to contribute valuable insights to filmmakers, viewers, and society at large, fostering a more inclusive and equitable cinematic landscape.

</span>

## Research Questions 🔍

Our journey through the cinematic universe is guided by the following pivotal questions:

- How do various film industries, such as Hollywood, Bollywood, and European cinema, portray women in their films?
- How does the representation of women, both in front of the camera and behind the scenes, shape a film's reception and resonance with its audience?
- Can the Bechdel test serve as a meaningful barometer for evaluating a film's portrayal of women, beyond its surface-level criteria?
- In what ways do film summaries, as condensed narratives, reflect gender biases or empowerment, influenced by the gender dynamics within the film's creation?
- Which genres in cinema are leading the charge in representing women as complex, multi-dimensional characters?

## Datasets 📊

In this analysis, we will use three different datasets

- The original and provided [CMU](http://www.cs.cmu.edu/~ark/personas/) dataset.
- The [TMDB Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset): this dataset offers metadata for over 45,000 movies, with a particular focus on user ratings. Our primary interest lies in getting the populariy, the number of votes each movie receives and its average score rating to define a success metric. This dataset also includes valuable details about a movie's crew, such as the gender of the director and producers. These data points are crucial for conducting analyses on women's roles behind the camera.
- The [Bechdel Dataset](https://bechdeltest.com) that contains the Bechdel rating score for more than 10'000 movies. This dataset will be used to conduct a analysis on movies that passes (or not) the Bechdel test.
- [Maybe Oscar ?]

## Methods 💡

### Preprocessing on **data**

<span align="justify"> 

**Original Dataset**: Initial cleaning has been conducted.

**Bechdel Dataset**: The Bechdel dataset boasts the advantage of minimal missing values, which simplifies the cleaning process significantly.

**TMDB Dataset**: Our focus with the TMDB dataset is to extract data on the crew members behind the scenes, particularly their gender. We mitigate the impact of missing values by employing the genderguesser package, which allows us to infer genders from first names and reduce the volume of incomplete data.

</span>

### Analysis of Women's place in movies
We consider that women's representation in media such as movies is crutial for advances in feminist struggles, because being able to relate on fictional characters is essential because it allows us to explore and understand complex human experiences and emotions in a safe and imaginative context. 
Analyzing the representation of women in movies through various datasets, including cast members genders, movie success, budget, and Bechdel test results, provides a comprehensive understanding of gender representation in the film industry; in cast and crew members as well as in the movies stories.

By correlating cast members genders with the movie success and budget, we can identify trends and biases in the industry. For instance, examining whether films with a higher proportion of female cast members are as likely to benefit from big budgets or result in significant box office entries.
In this analysis we will perform all kinds of tests to be able to notice the most significant correlations that could appear between the parameters of a movie and the virtual feminist score it achieves. The latter will be obtained through several tests, the main one being the Bechdel test. 

**Bechdel test:** The Bechdel test was originally invented in 1985 in a comic strip by Alison Bechdel and tests whether a movie passes the 3 following conditions: 
a) The movie has to have at least two women in it, b) who talk to each other, c) about something other than a man. This results in a score between 0 and 3, depending on the number of tests the movie passes.
It is considered to be a basic standard for minimal women's representation in movies and also includes some women's theorized stereotypes such as the Smurfette principle, which designs the fact that in many stories, the woman is the exception and exists only in reference to the men. Even though this test is very limited, it offers great insight on the role and representaion of the women in any movie.

To summarize, by conducting a thorough analysis of women's representation in movies using these datasets we hope to provide valuable insights into the passed and future progress in achieving gender equality in the film industry.

### Techniques
- **Natural Language Processing:** NLP techniques will be applied to film summaries to detect and analyze gender stereotypes.

- **Comparative Analysis:** We'll compare genres, cultures, and production types (independent vs. major studios) in terms of gender representation.

- **Temporal Analysis:** Examining how trends have evolved over time, we'll use this data to potentially forecast future trends.

- **Impact Assessment:** We'll assess the real-world impact of films that pass the Bechdel, particularly in terms of influencing public discourse and contributing to gender equality movements.

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
| F. Dumoncel     | `francois.dumoncel-kessler@epfl.ch`  | Preprocessing, Analysis, Clean the repo, README |
| K. Tetard       | `kenji.tetard@epfl.ch`               |  NLP |
| L. Vogel        | `lena.vogel@epfl.ch`                 | README, cleaning, team organization, Bechdel test |
| N. Dillenbourg  | `nael.dillenbourg@epfl.ch`           | Preprocessing, Wikidata scraping   |
| A. Bacuet       | `aymeric.bacuet@epfl.ch`             | Exploratory data Analysis   |
