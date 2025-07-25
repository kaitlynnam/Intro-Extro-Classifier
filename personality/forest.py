from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

print(train.head())

print(train.isnull().sum())

X_train = train[['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']]
y_train = train['Personality']

X_test = test

y_test = submission['Personality']


numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
categorial_cols = ['Stage_fear', 'Drained_after_socializing',]


preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='mean'))
    ]), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorial_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('forest', RandomForestClassifier(max_depth=7, min_samples_leaf=1, min_samples_split=5, n_estimators=10))
])

pipeline.fit(X_train, y_train)


accuracy = pipeline.score(X_test, y_test)

print(f"Accuracy: {accuracy:.4f}")


param_grid = {
    'forest__n_estimators': [10, 100, 500, 1000],
    'forest__max_depth': [3, 5, 7, 10],
    'forest__min_samples_split': [2, 5, 10],
    'forest__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2) #n_job=-1 allows the model to use all of the computer's cores, verbose=2 means it will give real time logs
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best accuracy:", grid_search.best_score_)