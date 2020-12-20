# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 23:43:24 2020

@author: Trinayan Borah
"""

from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from knn_model import *

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

#booK_pivot=pd.read_csv('book_pivot.csv')

#book_sparse=csr_matrix(knn_model.booK_pivot)

@app.route('/')
def home():
    Top_10_books_title,Top_10_books_img,Top_10_books_author,Top_10_books_year,Top_10_books_publisher,t_size=getTop10()
    print(Top_10_books_title)
    return render_template("index.html",top10=Top_10_books_title,imglink=Top_10_books_img,author=Top_10_books_author,year=Top_10_books_year,pub=Top_10_books_publisher,top_size=t_size,data_size= 0)
    #print(top_10_book_list)
    #return render_template("index.html")

@app.route('/recommend',methods=['POST','GET'])
def predict():
    features=[str(x) for x in request.form.values()]
    #final=[np.array(features)]
    print(len(features))
    book_list,img_list,author_list,year_list,publisher_list,t_size,data_size=recommend_book(features[0])
    #print(book_list)
    print(data_size)
    if(data_size==0):
        return render_template('index.html',top_size=t_size,data_size=data_size)
    else:
        return render_template('index.html',pred=book_list,imglink=img_list,author=author_list,year=year_list,pub=publisher_list,top_size=t_size,data_size= len(features))

@app.route('/top10')
def top10():
    Top_10_books_title,Top_10_books_img,Top_10_books_author,Top_10_books_year,Top_10_books_publisher=getTop10()
    print(Top_10_books_title)
    return render_template("index.html",top10=Top_10_books_title,imglink=Top_10_books_img,author=Top_10_books_author,year=Top_10_books_year,pub=Top_10_books_publisher)

if __name__ == '__main__':
    app.run(debug=True)