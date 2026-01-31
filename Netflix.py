import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('mymoviedb.csv', lineterminator='\n')

# Initial comment:
# - DataFrame has 9827 rows and 9 columns
# - No NaNs or duplicates
# - 'Release_Date' should be converted to datetime, extract year only
# - Drop 'Overview', 'Original_Language', 'Poster_Url' (not useful for analysis)
# - 'Popularity' has noticeable outliers
# - 'Vote_Average' should be categorized
# - 'Genre' column contains comma-separated values with whitespace

# Convert Release_Date to datetime and extract year
df['Release_Date'] = pd.to_datetime(df['Release_Date'])
df['Release_Date'] = df['Release_Date'].dt.year

# Drop unnecessary columns
cols = ['Overview', 'Original_Language', 'Poster_Url']
df.drop(cols, axis=1, inplace=True)

# Function to categorize continuous column into bins
def categorize_col(df, col, labels):
    edges = [
        df[col].describe()['min'],
        df[col].describe()['25%'],
        df[col].describe()['50%'],
        df[col].describe()['75%'],
        df[col].describe()['max']
    ]
    df[col] = pd.cut(df[col], edges, labels=labels, duplicates='drop')
    return df

# Categorize Vote_Average
labels = ['not_popular', 'below_average', 'average', 'popular']
categorize_col(df, 'Vote_Average', labels)
df.dropna(inplace = True)

# Clean Genre column: split and explode comma-separated genres
df['Genre'] = df['Genre'].str.split(',')
df = df.explode('Genre').reset_index(drop=True)
df['Genre'] = df['Genre'].str.strip().astype('category')

# Set plot style
sns.set_style('whitegrid')

# Q1: What is the most frequent genre?
print(df['Genre'].describe())
sns.catplot(
    y='Genre',
    data=df,
    kind='count',
    order=df['Genre'].value_counts().index,
    color='#4287f5'
)
plt.title('Genre column distribution')
plt.show()

# Q2: Which has the highest votes in Vote_Average?
sns.catplot(
    y='Vote_Average',
    data=df,
    kind='count',
    order=df['Vote_Average'].value_counts().index,
    color='#4287f5'
)
plt.title('Votes distribution')
plt.show()

# Q3: Which movie got the highest popularity? What’s its genre?
print("Most popular movie:")
print(df[df['Popularity'] == df['Popularity'].max()])

# Q4: What movie got the lowest popularity? What’s its genre?
print("Least popular movie:")
print(df[df['Popularity'] == df['Popularity'].min()])

# Q5: Which year has the most filmed movies?
df['Release_Date'].hist()
plt.title('Release date column distribution')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.show()

# -----------------------------
# Summary:
# Q1: Drama is the most frequent genre, appearing more than 14% of the time.
# Q2: 25.5% of the dataset falls in the 'popular' vote category.
#     Drama again leads as the most liked genre.
# Q3: 'Spider-Man: No Way Home' is the most popular movie (Action, Adventure, Sci-Fi).
# Q4: 'The United States, Thread' is the least popular (Music, Drama, War, Sci-Fi, History).
# Q5: Year 2020 had the highest number of movie releases.

