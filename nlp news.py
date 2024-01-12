import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from datetime import datetime
from scipy.sparse import hstack

patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-\n]+"
stopwords_ru = stopwords.words("russian")

#функция для очистки текстовых сообщений от лишних элементов
def remove_ell(doc):
    doc = re.sub(patterns, ' ', str(doc))
    return doc

#подготавливаем excel файл с двумя столбцами:
#1 - столбец с текстом новостей
#2 - столбец с заголовками новостей
df = pd.read_excel('путь к excel файлу', sheet_name = 'название листа в excel файле')

#создаем два новых столбца во фрейме с очищенным текстом новостей и очищенными заголовками
df['Текст очищенный'] = df['Текст'].apply(remove_ell)
df['Заголовок очищенный'] = df['Заголовок'].apply(remove_ell)

#векторизуем с помощью tfidf столбец с очищенными текстами новостей
tfidf = TfidfVectorizer(stop_words = stopwords_ru)
count_words_tfidf_text = tfidf.fit_transform(df.iloc[:,2])
text_columns = tfidf.get_feature_names_out()

#векторизуем с помощью tfidf столбец с очищенными заголовками новостей
tfidf = TfidfVectorizer(stop_words = stopwords_ru)
count_words_tfidf_title = tfidf.fit_transform(df.iloc[:,3])
title_columns = tfidf.get_feature_names_out()

#две получившиеся матрицы объединяем в одну
df_resulting = hstack([count_words_tfidf_text, count_words_tfidf_title])

#кластеризуем с помощью DBSCAN
#наилучший результат дает метрика - косинусное сходство
#epsilon нужно будет варьировать в зависимости от количества сообщений, но eps = 0.5
#почти в любом случае дает удобоваримый результат
#min_samples оставляем 1
result = DBSCAN(eps = 0.5, min_samples=1, metric = 'cosine').fit_predict(df_resulting)

#упорядочиваем кластеры по убыванию в зависимости от их размера
d = pd.DataFrame(result)
d['Текст'] = df['Текст']
d['Заголовок'] = df['Заголовок']
d_ser = d[0].value_counts()
d_ser = d_ser.reset_index()
d_result = d.merge(d_ser, left_on = 0, right_on = 'index', how = 'left')
d_result = d_result.sort_values(['0_y', 'key_0'], ascending = False)

with pd.ExcelWriter('excel файл с кластерами') as writer:
    d_result.to_excel(writer)