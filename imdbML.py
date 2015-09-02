import pandas as pd
import MySQLdb
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class MPAADataFrame:
    
    def __init__(self, username, password):
        """
        Pass in username and password
        """
        self.df = parseMPAARatings(username, password)
        self.pluckReason()
        self.pluckRating()
        self.createRatingValue()
        
    def pluckReason(self):

        """
        Plucks the reason for a rating from the 
        pandas dataframe 'info' key and returns
        """
        def safeSub(pat, x):
            try:
                return re.sub(pat, '', x)
            except:
                print x
            return x
    
        pat = re.compile('^(Rated)\s*(\sG|PG\s|PG-13\s|PG- 13|R\s|NC-17\s)\s*(for)*\s*')
        vmatch = np.vectorize(lambda x: safeSub(pat,x))
        self.reason = np.array([r for r in vmatch(self.df['info'].values)])

        
    def pluckRating(self):

        """
        Plucks the G, PG, PG-13, R rating from the 
        pandas dataframe 'info' key and returns it
        """

        def safeSearch(pat, x):
            try:
                return re.search(pat, x).group()
            except:
                print x
            return x
        
        pat = re.compile('(\sG|PG\s|PG-13\s|PG- 13|R\s|NC-17\s)')
        vmatch = np.vectorize(lambda x: safeSearch(pat,x))
        self.rating = np.array([r.replace(' ', '') for r in vmatch(self.df['info'].values)])

    def createRatingValue(self):
        """ 
        Maps MPAA rating to an integer
        """
        rating_key = {'G': 0, 'PG': 1, 'PG-13': 2, 'R': 3, 'NC-17': 4}
        self.rating_value = [rating_key[key] for key in self.rating]


    def testNaiveBayes(self):
        """
        Breaks the reason into Counted, tokenized, stemmed words
        """
        self.clf = Pipeline([('vect', CountVectorizer(encoding='ISO-8859-1')),
                             ('tfidf', TfidfTransformer()),
                             ('classifier', MultinomialNB()),
                             ])
        
        self.n_train = int(round(len(self.reason) * .67))
        self.clf.fit(self.reason[:self.n_train], self.rating_value[:self.n_train])
    

def parseMPAARatings(username, password, dbase="imdb.db"):
    
    """
    Parses the SQL IMDb and puts the title and MPAA rating and
    justification into a returned pandas dataframe
    """

    # connect to the database
    db = MySQLdb.connect(host="localhost", user=username,
                         passwd=password, db=dbase)

    # use pandas to get rows i want
    df = pd.read_sql('SELECT movie_info.id, title, info FROM movie_info \
    LEFT OUTER JOIN title ON movie_id = title.id WHERE info_type_id=97;',
                     con=db)    
    print 'loaded dataframe from MySQL. records:', len(df)
    db.close()

    return df

def addRatings(df, username, password, dbase='imdb.db'):

    """
    Calls pluckRating and adds results back to the movie_info table 
    """
    rating = pluckRating(df)

    db = MySQLdb.connect(host="localhost", user=username,
                         passwd=password, db=dbase)
    
    c = db.cursor()

    for i in df['id'].values:
        query = """UPDATE movie_info SET rating = '{:s}' WHERE id = {:d};""".format(rating[i-1], i)
        c.execute(query)
        
    db.commit()
    db.close()
        
        
