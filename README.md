# autobots
##### a small set of `scikit-learn` sklearn.Transformers (har har) to make constructing feature sklearn.Pipelines out of heterogenous pd.DataFrames 'easier' (less FeatureUnions out of nested Pipelines, and accessible at a higher hierarchy for GridSearch and such)
##### tldr; objs you can plug into a sklearn.pipeline.Pipeline that can take a whole DataFrame and only transform certain things at each step

## Usage:
```from autobots import *
from sklearn.pipeline import Pipeline
```
```



#vectorize text in this column


#dummies

#dummies

```

## define transformations, instantiate objects ##
```
#calculate a new column out of a df.column w/ a custom function
build_newcolumn = ColumnMapper(func=MyFunction,column='myColumn',name='newColumn',drop=False)

# date_to_utc defined above
utc = ColumnMapper(func=date_to_utc,column='date',name='utc',drop=False)

#merge w/weather from weatherdf by DATE
merge_weather = DfMerger(weather,on=['utc','date'],how='left',copy=True,v #dummy trap locations

#dummy the address column
dummy_address = DummyEncoder(column='address') #dummy species

#dummy the species column
dummy_species = DummyEncoder(column='species')

#drop stuff we hate
drop_us = 'lat long loc utc date 2007 2008 2009 2010 2011 2012 2013 2014' drop_columns = ColumnSelector(columns=drop_us,drop=True)

```
pipe = []
pipe.appemnd

```

