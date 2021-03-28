library(keras)

### DATAFRAME ########################################################################
politifact_df_raw <- read_csv("../data-raw/politifact_phase2_clean_2018_7_3.csv")  
 
# clean up data
politifact_df_cleaned <- politifact_df_raw %>% 
  # get rid of duplicate URLs 
  distinct(politifact_url_phase1, .keep_all = TRUE) %>% 
  # get rid of flip-o-meter ratings
  # filter(!(fact_tag_phase1 %in% c("Full Flop", "Half Flip", "No Flip"))) %>% 
  filter(fact_tag_phase1 %in% c("True", "Pants on Fire!")) %>% 
  # change desired targets to (categorical) numbers #### changed to numeric for keras
  # mutate(fact_tag_phase1 = as.numeric(ifelse(fact_tag_phase1 == "True", 1, 
  #                          ifelse(fact_tag_phase1 == "Mostly True", 1, 
  #                          ifelse(fact_tag_phase1 == "Half-True", 1,
  #                          ifelse(fact_tag_phase1 == "Mostly False", 0,
  #                          ifelse(fact_tag_phase1 == "False", 0, 0))))))) %>% 
  mutate(fact_tag_phase1 = as.numeric(ifelse(fact_tag_phase1 == "True", 1, 0))) %>% 
  select(politifact_url_phase1, article_claim_phase1, original_article_text_phase2, fact_tag_phase1)

# add an id 
politifact_df_cleaned$id <- seq.int(nrow(politifact_df_cleaned))
politifact_df_cleaned <- politifact_df_cleaned %>% 
  select(id, article_claim_phase1, original_article_text_phase2, fact_tag_phase1)

glimpse(politifact_df_cleaned)  


# refactor df for corpus 
politifact_corpus_df <- politifact_df_cleaned %>% 
  rename(doc_id = id,
         original_text = article_claim_phase1) %>%  
  mutate(text = lemmatize_strings(removePunctuation(original_text))) %>%   
  select(doc_id, text, fact_tag_phase1)
 
### CORPUS ########################################################################  
 
# Convert our dataframe to a volatile corpus
politifact_source <- DataframeSource(politifact_corpus_df)
politifact_corpus <- VCorpus(politifact_source) 
 
 
# a function to clean a corpus of text
clean_corpus <- function(corpus) {
  # basic text cleaning
  cleaned_corpus <- tm_map(corpus, removePunctuation)
  cleaned_corpus <- tm_map(cleaned_corpus, content_transformer(removeNumbers))
  cleaned_corpus <- tm_map(cleaned_corpus, content_transformer(tolower))

  # remove stop words
  all_stops <- c(stopwords("en"), "say")
  cleaned_corpus <- tm_map(cleaned_corpus, removeWords, c(all_stops))
  
  cleaned_corpus <- tm_map(cleaned_corpus, stripWhitespace) 
  return(cleaned_corpus)
}

politifact_corpus_cleaned <- clean_corpus(politifact_corpus)
 
### DTM ########################################################################
 
politifact_dtm <- DocumentTermMatrix(politifact_corpus_cleaned)
dim(politifact_dtm) 
length_of_dtm <- dim(politifact_dtm)[1]
# 80%: training data, 20%: test data
random_rows_for_training <- sample(1:length_of_dtm, 0.8 * length_of_dtm)  
 
#Training set
politifact_dtm_train <- politifact_dtm[random_rows_for_training, ]

# #Test set
politifact_dtm_test <- politifact_dtm[-random_rows_for_training, ] 

#Training targets
politifact_train_targets <- politifact_corpus_df[random_rows_for_training, ]$fact_tag_phase1

#Test Label
# note that hard coded values are needed here
politifact_test_targets <- politifact_corpus_df[-random_rows_for_training, ]$fact_tag_phase1
 
#Proportion for train targets
prop.table(table(politifact_train_targets))

#Proportion for test targets
prop.table(table(politifact_test_targets))
 

# only keep terms with at least 5 occurrences
politifact_freq_words <- findFreqTerms(politifact_dtm_train, 5)  

#filter the DTM sparse matrix to only contain words (columns here) with at least 5 occurence
#reducing the features in our DTM
politifact_dtm_train <- politifact_dtm_train[ , politifact_freq_words]
politifact_dtm_test <- politifact_dtm_test[ , politifact_freq_words]
dim(politifact_dtm_train)

# convert train and test features to matricies
politifact_dtm_train <- as.matrix(politifact_dtm_train)
politifact_dtm_test <- as.matrix(politifact_dtm_test)

### MLP ########################################################################

# take 25% of training (20% of all data since training is 80%) and turn it into validation 
num_validation_rows <- ceiling(num_training_rows * 0.25)
val_indices <- 1:num_validation_rows

x_validation <- politifact_dtm_train[val_indices,]
partial_x_train <- politifact_dtm_train[-val_indices,]

y_validation <- politifact_train_targets[val_indices]
partial_y_train <- politifact_train_targets[-val_indices] 

model <- keras_model_sequential() %>% 
  layer_dense(units = 460, activation = "relu", input_shape = c(dim(politifact_dtm_train)[2])) %>% 
  layer_dense(units = 460, activation = "relu") %>%  
  layer_dense(units = 460, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% keras::fit(
  partial_x_train,
  partial_y_train,
  epochs = 50,
  batch_size = 512,
  validation_data = list(x_validation, y_validation)
) 
plot(history) 

model <- keras_model_sequential() %>% 
  layer_dense(units = 400, activation = "relu", input_shape = c(dim(politifact_dtm_train)[2])) %>%  
  # layer_dense(units = 460, activation = "relu") %>%
  # layer_dense(units = 460, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = optimizer_rmsprop(lr=0.001), #optimizer_rmsprop(lr=0.001),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# for batch size info: https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
model %>% keras::fit(politifact_dtm_train, politifact_train_targets, epochs = 25, batch_size = 64)
results <- model %>% evaluate(politifact_dtm_test, politifact_test_targets)

classes <- model %>% predict_classes(politifact_dtm_test, batch_size = 64)

# Confusion matrix
table(politifact_test_targets, classes)

# save_model_hdf5(model, "my_model.h5")
# model <- load_model_hdf5("my_model.h5")


