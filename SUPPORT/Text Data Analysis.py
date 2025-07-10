import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import emoji
from collections import Counter
import plotly.graph_objs as go
import plotly.offline as pyo
import os
import warnings
from warnings import filterwarnings
filterwarnings('ignore')
import plotly.express as px
import string

comments = pd.read_csv(r'D:\DATA ANALYST PROJECT\PROJECT 1\Data/UScomments.csv', on_bad_lines='skip')
print(comments.head())

## lets find out missing values in your data
print(comments.isnull().sum())

## drop missing values as we have very few & lets update dataframe as well..
print(comments.dropna(inplace=True))

print(comments.isnull().sum())

print(comments.head(6))

text = TextBlob("Logan Paul it's yo big day ‼️‼️‼️").sentiment.polarity

### its a neutral sentence !
print(text)
print(comments.shape)

sample_df = comments.iloc[0:1000]
print(sample_df.shape)



polarity = []

for comment in comments['comment_text']:
    try:
        polarity.append(TextBlob(comment).sentiment.polarity)
    except:
        polarity.append(0)
        
print(len(polarity))

comments['polarity'] = comments['comment_text'].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)

def get_sentiment(p):
    if p > 0:
        return 'Positive'
    elif p < 0:
        return 'Negative'
    else:
        return 'Neutral'

comments['sentiment'] = comments['polarity'].apply(get_sentiment)

sns.countplot(data=comments, x='sentiment', order=['Positive', 'Neutral', 'Negative'])
plt.title("Sentiment Distribution of YouTube Comments")
plt.show()

top_pos = comments.sort_values(by='polarity', ascending=False).head(5)
top_neg = comments.sort_values(by='polarity').head(5)

print("Top Positive Comments:\n", top_pos[['comment_text', 'polarity']])
print("\nTop Negative Comments:\n", top_neg[['comment_text', 'polarity']])

#WordCloud

filter1 = comments['polarity']==1
comments_positive = comments[filter1]
filter2 = comments['polarity']==-1
comments_negative = comments[filter2]
print(comments_positive.head(5))

#stopwords = set(STOPWORDS)
print(comments['comment_text'])
print(type(comments['comment_text']))

### for wordcloud , we need to frame our 'comment_text' feature into string ..
total_comments_positive = ' '.join(comments_positive['comment_text'])

#Positive comments
wordcloud = WordCloud(stopwords=set(STOPWORDS)).generate(total_comments_positive)

plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#Negative comments
total_comments_negative = ' '.join(comments_negative['comment_text'])
wordcloud1 = WordCloud(stopwords=set(STOPWORDS)).generate(total_comments_negative)

plt.imshow(wordcloud1)
plt.axis('off')
plt.show()


#Emoji Analysis

print(comments['comment_text'].head(6))

comment = 'trending 😉'
[char for char in comment if char in emoji.EMOJI_DATA]

emoji_list = []

for char in comment:
    if char in emoji.EMOJI_DATA:
        emoji_list.append(char)
        
print(emoji_list)

all_emojis_list = []

for comment in comments['comment_text'].dropna(): ## in case u have missing values , call dropna()
    for char in comment:
        if char in emoji.EMOJI_DATA:
            all_emojis_list.append(char)
            
print(all_emojis_list[0:10])

#counter
print(Counter(all_emojis_list).most_common(10))
print(Counter(all_emojis_list).most_common(10)[0])
print(Counter(all_emojis_list).most_common(10)[0][0])
print(Counter(all_emojis_list).most_common(10)[1][0])
print(Counter(all_emojis_list).most_common(10)[2][0])
emojis = [Counter(all_emojis_list).most_common(10)[i][0] for i in range(10)]
print(Counter(all_emojis_list).most_common(10)[0][1])
print(Counter(all_emojis_list).most_common(10)[1][1])
print(Counter(all_emojis_list).most_common(10)[2][1])
freqs = [Counter(all_emojis_list).most_common(10)[i][1] for i in range(10)]
print(freqs)

trace = go.Bar(x=emojis , y=freqs, marker=dict(color='lightgreen'))
#iplot([trace])
pyo.plot([trace], filename='emoji_bar_chart.html')
## Conclusions : Majority of the customers are happy as most of them are using emojis like: funny , love , heart , outstanding..

#Collect Entire Data on Youtube

files= os.listdir(r'D:\DATA ANALYST PROJECT\PROJECT 1\Data\additional_data')
print(files)

## extracting csv files only from above list ..
files_csv = [file for file in files if '.csv' in file]
print(files_csv)


full_df = pd.DataFrame()
path = r'D:\DATA ANALYST PROJECT\PROJECT 1\Data\additional_data'


for file in files_csv:
    current_df = pd.read_csv(path+'/'+file, encoding='iso-8859-1', on_bad_lines='skip')
    
    full_df = pd.concat([full_df , current_df] , ignore_index=True)
print(full_df.shape)



duplicates = full_df[full_df.duplicated()]
print("Number of duplicate rows:", duplicates.shape[0])

full_df.drop_duplicates(inplace=True)
print("After removing duplicates:", full_df.shape)

full_df[0:1000].to_csv(r'D:\DATA ANALYST PROJECT\PROJECT 1\Data/youtube_sample.csv' , index=False)
full_df[0:1000].to_json(r'D:\DATA ANALYST PROJECT\PROJECT 1\Data/youtube_sample.json', index=False)

#Storing Data into DataBase

#create engine allows us to connect to database
from sqlalchemy import create_engine

engine = create_engine('sqlite:///D:\DATA ANALYST PROJECT\PROJECT 1\Data/youtube_sample.sqlite')
full_df[0:1000].to_sql('Users' , con=engine , if_exists='append')

#Which Category has Maximum Likes

print(full_df.head(5))
print(full_df['category_id'].unique())

json_df = pd.read_json(r'D:\DATA ANALYST PROJECT\PROJECT 1\Data\additional_data\US_category_id.json')
print(json_df)
print(json_df['items'][0])
print(json_df['items'][1])

cat_dict = {}

for item in json_df['items'].values:
    ## cat_dict[key] = value (Syntax to insert key:value in dictionary)
    cat_dict[int(item['id'])] = item['snippet']['title']
print(cat_dict)

full_df['category_name'] = full_df['category_id'].map(cat_dict)
print(full_df.head(4))

plt.figure(figsize=(12,8))
sns.boxplot(x='category_name' , y='likes' , data=full_df)
plt.xticks(rotation='vertical')
plt.show()


#Finding out Wheather audiance are engaged or not


full_df['like_rate'] = (full_df['likes']/full_df['views'])*100
full_df['dislike_rate'] = (full_df['dislikes']/full_df['views'])*100
full_df['comment_count_rate'] = (full_df['comment_count']/full_df['views'])*100
print(full_df.columns)

plt.figure(figsize=(8,6))
sns.boxplot(x='category_name' , y='like_rate' , data=full_df)
plt.xticks(rotation='vertical')
plt.show()

#analysing relationship between views & likes

sns.regplot(x='views' , y='likes' , data = full_df)
plt.show()

print(full_df.columns)

print(full_df[['views', 'likes', 'dislikes']].corr()) ### finding co-relation values between ['views', 'likes', 'dislikes'])

sns.heatmap(full_df[['views', 'likes', 'dislikes']].corr() , annot=True)
plt.show()

#Which channel have Largest number of trending videos?

print(full_df.head(6))
print(full_df['channel_title'].value_counts())

#frequency table using groupby approach : 

cdf = full_df.groupby(['channel_title']).size().sort_values(ascending=False).reset_index()
cdf = cdf.rename(columns={0:'total_videos'})
print(cdf)

px.bar(data_frame=cdf[0:20] , x='channel_title' , y='total_videos')
plt.show()

### Does Punctuations in title and tags have any relation with views, likes, dislikes comments?

print(full_df['title'][0])
print(string.punctuation)
len([char for char in full_df['title'][0] if char in string.punctuation])

def punc_count(text):
    return len([char for char in text if char in string.punctuation])

sample = full_df[0:10000]
sample['count_punc'] = sample['title'].apply(punc_count)

print(sample['count_punc'])

plt.figure(figsize=(8,6))
sns.boxplot(x='count_punc' , y='views' , data=sample)
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x='count_punc' , y='likes' , data=sample)
plt.show()


#Recommendations Based on Emoji Usage

# Find top emojis
top_emojis = Counter(all_emojis_list).most_common(5)
print("🔍 Top Emojis Used in Comments:")
for emoji_char, freq in top_emojis:
    print(f"✅ Use emoji: {emoji_char} - {freq} times")
    
#Punctuation Impact on Views
#Group by count_punc and calculate average views:

sample['punct_bin'] = pd.cut(sample['count_punc'], bins=[0, 2, 5, 10, 100], labels=['Low', 'Medium', 'High', 'Very High'])
punc_view_summary = sample.groupby('punct_bin')['views'].mean().reset_index()

print("\n📌 Average Views vs. Punctuation Level:")
print(punc_view_summary)

#Recommendation Logic:

optimal_punct_level = punc_view_summary.sort_values(by='views', ascending=False).iloc[0]['punct_bin']
print(f"\n✅ Recommendation: Use a '{optimal_punct_level}' amount of punctuation in your video titles for better engagement.")

#Category-Wise Engagement
#like_rate or views to find high-performing categories:

category_engagement = full_df.groupby('category_name')['like_rate'].mean().sort_values(ascending=False).reset_index()
top_categories = category_engagement.head(3)

print("\n📢 Categories with Highest Like Rates:")
print(top_categories)

print("\n✅ Recommendation: Post videos in the following top-performing categories:")
for i, row in top_categories.iterrows():
    print(f"- {row['category_name']} (avg like rate: {row['like_rate']:.2f}%)")
    
#Final Recommendation Summary Code

print("\n🎯 FINAL RECOMMENDATIONS FOR CONTENT CREATORS 🎥")
print("1️⃣ Use emojis like ❤️, 😂, 😍 in video titles and thumbnails to attract engagement.")
print(f"2️⃣ Limit punctuation to '{optimal_punct_level}' level for better click-through and views.")
print("3️⃣ Focus on posting in high-performing categories such as:")
for i, row in top_categories.iterrows():
    print(f"   • {row['category_name']} (Like Rate: {row['like_rate']:.2f}%)")
print("4️⃣ Engage your audience emotionally — positive sentiment content performs best.")

# Punctuation Level vs. Average Views

plt.figure(figsize=(8, 5))
sns.barplot(data=punc_view_summary, x='punct_bin', y='views', palette='viridis')
plt.title('📌 Average Views by Punctuation Level in Title')
plt.xlabel('Punctuation Level')
plt.ylabel('Average Views')
plt.show()

#Top Categories by Like Rate
#Highlight which categories engage viewers the most
fig = px.bar(top_categories, x='category_name', y='like_rate',
             title='🏆 Top 3 Categories by Like Rate',
             labels={'category_name': 'Category', 'like_rate': 'Average Like Rate (%)'},
             color='like_rate', color_continuous_scale='Blues')
fig.show()










