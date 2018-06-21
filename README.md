# autobots
a small set of `scikit-learn` sklearn.Transformers (har har) to make constructing feature `sklearn.Pipeline`s out of heterogenous DataFrames 'easier' (less FeatureUnions out of nested Pipelines, and accessible at a higher hierarchy for GridSearch and such)

##### tldr; objs you can plug into a sklearn.pipeline.Pipeline that can take a whole DataFrame and only transform certain things at each step

## Usage:
```
from autobots import *

import pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
```

### define transormations
```
my_map_function = (lambda x: x)  #lambda over a column

my_apply_function = (lambda row: row['col3'] + row['col4']) #lambda over rows

def my_custom_function (x):
  #do some stuff
  return output


```

## instantiate objects
```
# transform col1 into colA
colA = ColumnMapper(func=my_map_function, column='col1', name='colA', drop=True)

# create colB from col2
colB = ColumnMapper(func=my_custom_function, column='col2', name='colB', drop=False)

# create colC from col3 and col4
colB = ColumnApplier(func=my_apply_function, name='colB', axis=1)

# merge w/values from df2 by key
merge_df1_df2 = DfMerger(df2,on=['id','date'], how='left', copy=True, validate='m:1')

# pd.get_dummies that can fit_transform X_train and transform X_test based on X_train's dummy columns
dummy_col5 = DummyEncoder(column='col5')

# drop columns
drop_list = 'other column names we dont want'.split()
drop_columns = ColumnSelector(columns=drop_list,drop=True)

```

## easy pipeline!
```
# build a pipe

pipe = []
pipe.append(('colA',colA))
pipe.append(('colB',colB))
pipe.append(('colC',colC))
pipe.append(('merge',merge_df1_df2))
pipe.append(('dummy_col5',dummy_col5))
pipe.append(('drop_columns',drop_columns))

preprocess_X = Pipeline(pipe)
```
 
## build features
```
from sklearn.model_selection import train_test_split

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y)

Xtrain = preprocess_X.fit_transform(Xtrain)
Xtest = preprocess_X.transform(Xtest)
```

## work on unseen data
```
newdf = pd.read_csv('./from_kaggle.csv')

X = preprocess_X.transform(newdf)

yhat - model.predict(X)
```
