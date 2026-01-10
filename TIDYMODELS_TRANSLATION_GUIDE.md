# Quick Reference: h2o to tidymodels Translation

## Common Patterns

### Data Splitting

#### Before (h2o):
```r
library(h2o)
h2o.init()
ames.h2o <- as.h2o(ames)
split <- h2o.splitFrame(ames.h2o, ratios = 0.7, seed = 123)
train <- split[[1]]
test <- split[[2]]
```

#### After (tidymodels):
```r
library(rsample)
set.seed(123)
split <- initial_split(ames, prop = 0.7, strata = "Sale_Price")
train <- training(split)
test <- testing(split)
```

---

### Random Forest Model

#### Before (h2o):
```r
library(h2o)
h2o_rf <- h2o.randomForest(
  x = predictors,
  y = response,
  training_frame = train_h2o,
  ntrees = 500,
  mtries = 10,
  seed = 123
)
```

#### After (tidymodels):
```r
library(parsnip)
library(workflows)

rf_spec <- rand_forest(
  trees = 500,
  mtry = 10
) %>%
  set_engine("ranger") %>%
  set_mode("regression")

rf_workflow <- workflow() %>%
  add_formula(response ~ .) %>%
  add_model(rf_spec)

rf_fit <- fit(rf_workflow, data = train)
```

---

### Hyperparameter Tuning with Grid Search

#### Before (h2o):
```r
hyper_grid <- list(
  mtries = c(5, 10, 15),
  max_depth = c(10, 20, 30)
)

search_criteria <- list(
  strategy = "RandomDiscrete",
  max_runtime_secs = 300
)

grid <- h2o.grid(
  algorithm = "randomForest",
  x = predictors,
  y = response,
  training_frame = train_h2o,
  hyper_params = hyper_grid,
  search_criteria = search_criteria
)

grid_perf <- h2o.getGrid(grid_id = grid@grid_id, sort_by = "mse")
```

#### After (tidymodels):
```r
library(tune)
library(dials)

# Define model with tuning parameters
rf_spec <- rand_forest(
  mtry = tune(),
  trees = 500,
  min_n = tune()
) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# Create workflow
rf_workflow <- workflow() %>%
  add_formula(response ~ .) %>%
  add_model(rf_spec)

# Create grid
rf_grid <- grid_random(
  mtry(range = c(5, 15)),
  min_n(range = c(5, 15)),
  size = 20
)

# Tune with cross-validation
set.seed(123)
folds <- vfold_cv(train, v = 10)

rf_tuning <- tune_grid(
  rf_workflow,
  resamples = folds,
  grid = rf_grid
)

# Get best model
best_rf <- select_best(rf_tuning, metric = "rmse")
```

---

### Gradient Boosting Machine (GBM)

#### Before (h2o):
```r
h2o_gbm <- h2o.gbm(
  x = predictors,
  y = response,
  training_frame = train_h2o,
  ntrees = 500,
  learn_rate = 0.01,
  max_depth = 5,
  nfolds = 10
)
```

#### After (tidymodels):
```r
library(parsnip)

gbm_spec <- boost_tree(
  trees = 500,
  learn_rate = 0.01,
  tree_depth = 5
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

gbm_workflow <- workflow() %>%
  add_formula(response ~ .) %>%
  add_model(gbm_spec)

# With cross-validation
folds <- vfold_cv(train, v = 10)
gbm_fit <- fit_resamples(gbm_workflow, resamples = folds)
```

---

### Principal Components Analysis (PCA)

#### Before (h2o):
```r
library(h2o)
h2o.init()
data_h2o <- as.h2o(data)

pca_model <- h2o.prcomp(
  training_frame = data_h2o,
  k = ncol(data_h2o),
  transform = "STANDARDIZE"
)

# Get loadings
loadings <- pca_model@model$eigenvectors
```

#### After (base R):
```r
# Using base R prcomp
pca_model <- prcomp(
  data,
  center = TRUE,
  scale. = TRUE
)

# Get loadings
loadings <- pca_model$rotation

# Get variance explained
pve <- pca_model$sdev^2 / sum(pca_model$sdev^2)
cve <- cumsum(pve)
```

---

### Making Predictions

#### Before (h2o):
```r
predictions <- h2o.predict(model, newdata = test_h2o)
predictions <- as.data.frame(predictions)
```

#### After (tidymodels):
```r
predictions <- predict(model_fit, new_data = test)

# With actual values
predictions <- predict(model_fit, new_data = test) %>%
  bind_cols(test %>% select(response))
```

---

### Model Performance Metrics

#### Before (h2o):
```r
perf <- h2o.performance(model, newdata = test_h2o)
h2o.rmse(perf)
h2o.r2(perf)
```

#### After (tidymodels):
```r
library(yardstick)

predictions <- predict(model_fit, new_data = test) %>%
  bind_cols(test %>% select(response))

predictions %>%
  metrics(truth = response, estimate = .pred)

# Or specific metrics
predictions %>%
  rmse(truth = response, estimate = .pred)
```

---

## Package Correspondence

| Task | h2o | tidymodels |
|------|-----|------------|
| Data splitting | `h2o.splitFrame()` | `rsample::initial_split()` |
| Cross-validation | `nfolds` parameter | `rsample::vfold_cv()` |
| Model specification | `h2o.randomForest()`, `h2o.gbm()` | `parsnip::rand_forest()`, `parsnip::boost_tree()` |
| Hyperparameter tuning | `h2o.grid()` | `tune::tune_grid()` |
| Model training | Automatic in h2o | `parsnip::fit()`, `tune::fit_resamples()` |
| Predictions | `h2o.predict()` | `stats::predict()` or `parsnip::predict()` |
| Performance metrics | `h2o.performance()` | `yardstick::metrics()` |
| Variable importance | `h2o.varimp()` | `vip::vip()` |
| PCA | `h2o.prcomp()` | `stats::prcomp()` or `recipes::step_pca()` |

---

## Key Differences

1. **Initialization:** h2o requires `h2o.init()` and `h2o.shutdown()`, tidymodels does not
2. **Data Format:** h2o requires converting to h2o objects, tidymodels works with data frames
3. **Workflow:** tidymodels uses explicit workflows combining preprocessing and modeling
4. **Engines:** tidymodels can use multiple engines (ranger, randomForest, xgboost, etc.)
5. **Unified Interface:** tidymodels provides consistent syntax across all models

---

## Additional Resources

- [tidymodels website](https://www.tidymodels.org/)
- [parsnip documentation](https://parsnip.tidymodels.org/)
- [tune documentation](https://tune.tidymodels.org/)
- [recipes documentation](https://recipes.tidymodels.org/)
- [rsample documentation](https://rsample.tidymodels.org/)
