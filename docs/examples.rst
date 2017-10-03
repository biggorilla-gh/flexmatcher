========
Examples
========

Imagine that we have two datasets on movies provided in pandas dataframe
format. Let's say that we are interested in three attributes, namely 'movie_name',
'movie_year' and 'movie_rating'. The name of columns in the two datasets may
differ form the names that we just listed. Let's say that we look into the first
two dataset and specify how each column maps to one of the three attributes 
that is of interest to us.::

    vals1 = [['year', 'Movie', 'imdb_rating'],
            ['2001', 'Lord of the Rings', '8.8'],
            ['2010', 'Inception', '8.7'],
            ['1999', 'The Matrix', '8.7']]
    header = vals1.pop(0)
    data1 = pd.DataFrame(vals1, columns=header)
    # creating the second dataset
    vals2 = [['title', 'produced', 'popularity'],
            ['The Godfather', '1972', '9.2'],
            ['Silver Linings Playbook', '2012', '7.8'],
            ['The Big Short', '2015', '7.8']]
    header = vals2.pop(0)
    data2 = pd.DataFrame(vals2, columns=header)
    # specifying the mappings for the first and second datasets
    data1_mapping = {'year': 'movie_year',
                     'imdb_rating': 'movie_rating',
                     'Movie': 'movie_name'}
    data2_mapping = {'popularity': 'movie_rating',
                     'produced': 'movie_year',
                     'title': 'movie_name'}

Now, let's assume that we are given a thirs dataset.::

    # creating the third dataset
    vals3 = [['rt', 'id', 'yr'],
            ['8.5', 'The Pianist', '2002'],
            ['7.7', 'The Social Network', '2010']]
    header = vals3.pop(0)
    data3 = pd.DataFrame(vals3, columns=header)

We can use flexmatcher to find how the columns in the new dataset
are related to the attributes of our interest. To do so, we need to
create an instance of FlexMatcher, make a list of available datasets
and their mappings to the desired attributes, and train the FlexMatcher.::

    schema_list = [data1, data2]
    mapping_list = [data1_mapping, data2_mapping]
    fm = flexmatcher.FlexMatcher(schema_list, mapping_list, sample_size=100)
    fm.train()

Then, we can use the trained FlexMatcher to predict the mappings for the 
third dataset as follows.::

    predicted_mapping = fm.make_prediction(data3)
    
The result is a dictionary that maps every column to an attribute. For instance,::

    >>> print(predicted_mapping['rc'])
    movie_rating

