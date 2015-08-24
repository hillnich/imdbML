import pandas as pd
import MySQLdb
import re
import numpy as np

def parseMPAARatings(password, dbase="imdb.db"):
    
    """
    Parses the SQL IMDb and puts the title and MPAA rating and
    justification into a returned pandas dataframe
    """

    # connect to the database
    db = MySQLdb.connect(host="localhost", user="nhill",
                         passwd=password, db=dbase)

    # use pandas to get rows i want
    df = pd.read_sql('SELECT movie_info.id, title, info FROM movie_info \
    LEFT OUTER JOIN title ON movie_id = title.id WHERE info_type_id=97;',
                     con=db)    
    print 'loaded dataframe from MySQL. records:', len(df)
    db.close()

    return df

def pluckRating(df):

    """
    Plucks the G, PG, PG-13, R rating from the 
    pandas dataframe 'info' key and adds it as another entry
    in the dataframe
    """

    def safeSearch(pat, x):
        try:
            return re.search(pat, x).group()
        except:
            print x
            return x
    
    pat = re.compile('(\sG|PG\s|PG-13\s|PG- 13|R\s|NC-17\s)')
    vmatch = np.vectorize(lambda x: safeSearch(pat,x))
    rating = np.array([r.replace(' ', '') for r in vmatch(df['info'].values)])

    return rating

def addRatings(df, password, dbase='imdb.db'):

    """
    Calls pluckRating and adds results back to the movie_info table 
    """
    rating = pluckRating(df)

    db = MySQLdb.connect(host="localhost", user="nhill",
                         passwd=password, db=dbase)
    
    c = db.cursor()

    for i in df['id'].values:
        query = """UPDATE movie_info SET rating = '{:s}' WHERE id = {:d};""".format(rating[i-1], i)
        c.execute(query)
        
    db.commit()
    db.close()
        
        
