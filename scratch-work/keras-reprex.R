# running this entire script multiple times yields different confusion matrices
# generated from the keras neural network.
# this shows that keras creates its own random initial weights every time a neural
# network is compiled (and that set.seed() is not useful here)
library(keras)
library(tidyverse)
library(datasets)

set.seed(2)
iris_df <- iris %>% 
  filter(Species == "setosa" | Species == "versicolor") %>% 
  # 0 = setosa, 1 = versicolor
  mutate(Species = as.numeric(ifelse(Species == "setosa", 0, 1)))

# 80%: training data, 20%: test data
random_rows <- sample(1:nrow(iris_df), 0.8 * nrow(iris_df)) 
train  <- iris_df[random_rows, ]
test  <- iris_df[-random_rows, ]

# split our data into features and targets
x_train <- train %>% 
  select(-Species)
# keras requires matricies (not dataframes) as input type
x_train <- as.matrix(x_train)

y_train <- train$Species
   
x_test <- test %>% 
  select(-Species)
x_test <- as.matrix(x_test)

y_test <- test$Species

# we have four predictors
num_predictors <- ncol(x_train)

iris_model <- keras_model_sequential() %>% 
  layer_dense(units = 4, activation = "relu", input_shape = c(num_predictors)) %>%  
  layer_dense(units = 1, activation = "sigmoid")

iris_model %>% compile(
  optimizer = "rmsprop", 
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# note that setting epochs to 17 will result in 100% accuracy
iris_model %>% keras::fit(x_train, y_train, epochs = 1, batch_size = 1)
results <- iris_model %>% evaluate(x_test, y_test)

classes <- iris_model %>% predict_classes(x_test, batch_size = 1)

# Confusion matrix
table(y_test, classes) 