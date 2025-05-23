import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import accuracy_score
from google.colab import files

base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
pages = 10
page_size = 100

reviews = []

# for i in range(1, pages + 1):
for i in range(1, pages + 1):

    print(f"Scraping page {i}")

    # Create URL to collect links from paginated data
    url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"

    # Collect HTML data from this page
    response = requests.get(url)

    # Parse content
    content = response.content
    parsed_content = BeautifulSoup(content, 'html.parser')
    for para in parsed_content.find_all("div", {"class": "text_content"}):
        reviews.append(para.get_text())

    print(f"   ---> {len(reviews)} total reviews")

reviews

df = pd.DataFrame(np.array(reviews), columns=['Reviews'] )

df.to_csv('df.csv')
files.download('df.csv')

df

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def sentiment_analyser(reviews):
  tokens = tokenizer.encode(reviews, return_tensors='pt')
  result = model(tokens)
  return int(torch.argmax(result.logits))+1

df['Sentiment'] = df['Reviews'].apply(lambda x: sentiment_analyser (x[:512]))

accuracy = accuracy_score(df['Sentiment'], df['Reviews'].apply(lambda x: sentiment_analyser(x[:512])))

accuracy

df.shape

df.isnull().sum()

plt.scatter(df.Sentiment[:200], range(1,201), label='Sentiment', color = 'c')
plt.xlabel('Review Number')
plt.ylabel('Sentiment Score')
plt.legend()
plt.grid(True)
df.plot()
plt.title('Sentiment Analysis of British Airways Reviews')
plt.show

df1 = df.Sentiment == 1
df2 = df.Sentiment == 2
df3 = df.Sentiment == 3
df4 = df.Sentiment == 4
df5 = df.Sentiment == 5
Sent = [df1.sum(), df2.sum(), df3.sum(), df4.sum(), df5.sum()]
my_labels = ["Score=1", "Score=2", "Score=3", "Score=4", "Score=5"]

plt.pie(Sent, labels = my_labels, startangle=90, autopct='%1.1f%%')

