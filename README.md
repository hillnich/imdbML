### Constructing IMDb MySQL database
Start by downloading into IMDb list data that you want included in your SQL database.  These are made available as plain text data files from IMDb and are made available on several ftp sites found here: http://www.imdb.com/interfaces

If you then place those in the `data` directory and are running a MySQL databaseserver, you can execute the following to build your IMDb MySQL database representing all the downloaded lists in `data` with:
```
python imdbpy2sql.py -d data/ --mysql-force-myisam -u mysql://user:password@localhost/imdb.db
```

For more information on the various options available via `imdbpy2sql.py` (eg: how to use a different type of database), check out the full imdbpy project at https://github.com/alberanid/imdbpy and http://imdbpy.sourceforge.net/support.html

Note: The SQL queries in imdbML.py assume a MySQL database.

### Grabbing Spam Dataset
To run `naiveBayes_spam.py`, grab all the `20021010*bz2` files from http://spamassassin.apache.org/publiccorpus/

Untar them in the `spam_data` directory, and then Naive Bayes classifer can be run from the command line:
```
python naiveBayes_spam.py
```
Then, make sure you've installed the Python stemming module:
```
pip install stemming
```
