# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 23:27:08 2020

@author: Trinayan Borah
"""

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle
import numpy as np

books=pd.read_csv('Books_preprocessed.csv')
books=books.reset_index().drop_duplicates(subset=['book_title'], keep='first')
ratings=pd.read_csv('Ratings_preprocessed-Copy1.csv')
ratings['no_of_ratings']=ratings['isbn'].map(ratings.groupby('isbn')['book_rating'].count())
users=pd.read_csv('Users_preprocessed.csv')
counts1 = ratings['user_id'].value_counts()
ratings = ratings[ratings['user_id'].isin(counts1[counts1 >= 200].index)]
counts = ratings['book_rating'].value_counts()
ratings = ratings[ratings['book_rating'].isin(counts[counts >= 50].index)]
combine_book_rating = pd.merge(ratings, books, on='isbn')

columns = ['year_of_publication', 'publisher', 'book_author', 'image_url_s', 'image_url_m', 'image_url_l','Unnamed: 0_x','Unnamed: 0_y']
combine_book_rating = combine_book_rating.drop(columns, axis=1)

combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['book_title'])

book_ratingCount = (combine_book_rating.
     groupby(by = ['book_title'])['book_rating'].
     count().
     reset_index().
     rename(columns = {'book_rating': 'totalRatingCount'})
     [['book_title', 'totalRatingCount']]
    )

rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'book_title', right_on = 'book_title', how = 'left')

popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

booK_pivot=rating_popular_book.pivot_table(columns='user_id',index='book_title',values='book_rating')

booK_pivot.fillna(0,inplace=True)


book_sparse=csr_matrix(booK_pivot)

#booK_pivot.to_excel("book_pivot.xlsx")

ne_model=NearestNeighbors(algorithm='brute')

ne_model.fit(book_sparse)

#Top 10 rated books
ratings_count=pd.DataFrame(ratings.groupby(['isbn'])['book_rating'].sum())
#ratings_count
top10=ratings_count.sort_values('book_rating',ascending=False).head(10)
top=top10.merge(books,left_index=True,right_on='isbn')
#print(top['isbn'])

pickle.dump(ne_model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
re_list=[]
img_list=[]
author_list=[]
year_list=[]
publisher_list=[]
def recommend_book(book_name):
    re_list.clear()
    img_list.clear()
    author_list.clear()
    year_list.clear()
    publisher_list.clear()
    top_size=0
    x=np.where(booK_pivot.index==book_name)
    print(x[0].size)
    if(x[0].size==0):
        data_size=x[0].size
    else:
        data_size=x[0].size
        book_id=np.where(booK_pivot.index==book_name)[0][0]
        distances,suggestions=model.kneighbors(booK_pivot.iloc[book_id,:].values.reshape(1,-1),n_neighbors=6)
        for i in range(0,len(suggestions[0])):
            re_list.append(booK_pivot.index[suggestions[0][i]])
            img_list.append(books.loc[books['book_title']==booK_pivot.index[suggestions[0][i]]]['image_url_l'].values[0])
            author_list.append(books.loc[books['book_title']==booK_pivot.index[suggestions[0][i]]]['book_author'].values[0])
            year_list.append(books.loc[books['book_title']==booK_pivot.index[suggestions[0][i]]]['year_of_publication'].values[0])
            publisher_list.append(books.loc[books['book_title']==booK_pivot.index[suggestions[0][i]]]['publisher'].values[0])       
    return re_list,img_list,author_list,year_list,publisher_list,top_size,data_size


Top_10_books_title=[]
Top_10_books_img=[]
Top_10_books_author=[]
Top_10_books_year=[]
Top_10_books_publisher=[]
def getTop10():
    Top_10_books_title.clear()
    Top_10_books_img.clear()
    Top_10_books_author.clear() 
    Top_10_books_year.clear()
    Top_10_books_publisher.clear()
    Top_10_books_title.append((top['book_title'].tolist()))
    Top_10_books_img.append(top['image_url_l'].tolist())
    Top_10_books_author.append(top['book_author'].tolist())
    Top_10_books_year.append(top['year_of_publication'].tolist())
    Top_10_books_publisher.append(top['publisher'].tolist())
    top_size=len(Top_10_books_title)
    return Top_10_books_title,Top_10_books_img,Top_10_books_author,Top_10_books_year,Top_10_books_publisher,top_size;
print('Done')



