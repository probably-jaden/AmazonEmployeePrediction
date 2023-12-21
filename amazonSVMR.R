
library(plotly)
library(readr)
library(vroom)
library(tune)
library(ranger)
library(tidymodels)
library(recipes)
library(skimr)
library(poissonreg)
library(lubridate)
library(embed)

# import data
train<-vroom("train.csv")
test<- vroom("test.csv")

glimpse(train)

# Define your recipe with target encoding
my_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(RESOURCE, MGR_ID, ROLE_ROLLUP_1, ROLE_ROLLUP_2, ROLE_DEPTNAME,
                     ROLE_TITLE, ROLE_FAMILY_DESC, ROLE_FAMILY, ROLE_CODE,
                     outcome = vars(ACTION))

prep <- prep(my_recipe)
baked <- bake(prep, new_data = test)

## Model
model <- logistic_reg() %>% 
  set_engine("glm")


## workflow
amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(model) %>% 
  fit(data = train)

amazon_predictions <- predict(amazon_workflow, new_data=test, type="class")




svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

## Grid of values to tune over
tuning_grid <- grid_regular(rbf_sigma(), cost(), levels = 5)

# Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

# Run the cross-validation
CV_results <- bike_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae, rsq))

# Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("rmse")

  
  