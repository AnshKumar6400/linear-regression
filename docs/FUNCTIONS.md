# Functions & Methods Used in `index.ipynb`

Reference notes for every function/method called in the notebook, in the order they appear.

## Data loading

### `pathlib.Path`
Builds filesystem paths in an OS-independent way.
```python
csv_path = Path.cwd() / "datasets" / "Housing.csv"
```
`Path.cwd()` returns the current working directory; `/` joins path segments.

### `pandas.read_csv(path)`
Reads a CSV file into a `DataFrame` — a 2D labeled table (rows + named columns).
```python
df = pd.read_csv(csv_path)
```

### `DataFrame.head(n=5)`
Returns the first `n` rows of a DataFrame. Used to sanity-check that data loaded correctly.

## Feature / target split

### `DataFrame[[col1, col2, ...]]`
Selects multiple columns by name, returning a new DataFrame. Used to build the feature matrix `X`:
```python
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
```

### `DataFrame[col]`
Selects a single column, returning a `Series`. Used for the target vector:
```python
y = df['price']
```

### `DataFrame.shape`
Attribute (not a method) — a `(rows, columns)` tuple describing the DataFrame's dimensions.

## Train/test split

### `sklearn.model_selection.train_test_split(X, y, test_size, random_state)`
Randomly splits features and target into training and testing subsets.
- `test_size=0.2` — 20% of rows held out for testing, 80% for training.
- `random_state=42` — seeds the shuffle so the split is reproducible.

Returns `X_train, X_test, y_train, y_test`.

## Model

### `sklearn.linear_model.LinearRegression()`
Constructs an (unfitted) ordinary least-squares linear regression model.

### `model.fit(X_train, y_train)`
Learns the coefficients that minimize squared error between predictions and `y_train`.

### `model.coef_`
Attribute — the learned weight for each feature column (same order as `X`'s columns).

### `model.intercept_`
Attribute — the learned bias term (predicted value when all features are 0).

### `model.predict(X_test)`
Applies the learned coefficients/intercept to unseen `X_test` rows, returning predicted `y` values.

## Evaluation

### `sklearn.metrics.mean_absolute_error(y_true, y_pred)`
See [`METRICS.md`](./METRICS.md#mean-absolute-error-mae).

### `sklearn.metrics.mean_squared_error(y_true, y_pred)`
See [`METRICS.md`](./METRICS.md#mean-squared-error-mse).

### `numpy.sqrt(x)`
Element-wise square root. Used here to turn MSE into RMSE (same units as `y`).

### `sklearn.metrics.r2_score(y_true, y_pred)`
See [`METRICS.md`](./METRICS.md#r-coefficient-of-determination).
