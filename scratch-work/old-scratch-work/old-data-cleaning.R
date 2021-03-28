#Create a corpus df: 
# refactor df for corpus 
politifact_corpus_df_text <- politifact_df_cleaned %>% 
  rename(doc_id = id,
         original_text = original_article_text_phase2) %>%  
  mutate(text = lemmatize_strings(removePunctuation(original_text))) %>%   
  select(doc_id, text, fact_tag_phase1)

politifact_corpus_df_claim <- politifact_df_cleaned %>% 
  rename(doc_id = id,
         original_text = article_claim_phase1) %>%  
  mutate(text = lemmatize_strings(removePunctuation(original_text))) %>%   
  select(doc_id, text, fact_tag_phase1)

# saveRDS(politifact_corpus_df, file="full_text_lemmatized.rds")
# politifact_corpus_df <- readRDS("full_text_lemmatized.rds")
 


# create our corpus 
# Convert our dataframe to a volatile corpus
politifact_source <- DataframeSource(politifact_corpus_df)
politifact_corpus <- VCorpus(politifact_source)  

#Clean up the text in our corpus (including lemmatization and stop words): 
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



# Create a Document Term Matrix 
politifact_dtm <- DocumentTermMatrix(politifact_corpus_cleaned)
dim(politifact_dtm)
# politifact_dtm_rm_sparse <- removeSparseTerms(politifact_dtm, 0.9995) 
# politifact_m <- as.matrix(politifact_dtm) 
# dim(politifact_m)
# dim(as.matrix(DocumentTermMatrix(politifact_corpus))) this tells us that we 
# reomved 12460 - 7769 = 4691 words from our vocabulary 
 
# training_prop <- 0.8
# testing_prop <- 1 - training_prop
# total_obs <- dim(politifact_dtm)[1]
# training_num <- ceiling(training_prop * total_obs)
# test_num <- total_obs - training_num

#Training set
politifact_dtm_train <- politifact_dtm[1:6001, ]

# #Test set
politifact_dtm_test <- politifact_dtm[6002:7501, ] 

#Training targets
politifact_train_targets <- politifact_corpus_df[1:6001, ]$fact_tag_phase1

#Test Label
politifact_test_targets <- politifact_corpus_df[6002:7501, ]$fact_tag_phase1
 

 
#Proportion for train targets
prop.table(table(politifact_train_targets))

#Proportion for test targets
prop.table(table(politifact_test_targets))
 
 
# only keep terms with at least 5 occurrences
politifact_freq_words <- findFreqTerms(politifact_dtm_train, 5)  

#filter the DTM sparse matrix to only contain words with at least 5 occurence
#reducing the features in our DTM
politifact_dtm_train <- politifact_dtm_train[ , politifact_freq_words]
politifact_dtm_test <- politifact_dtm_test[ , politifact_freq_words]
dim(politifact_dtm_train)

# convert train and test features to matricies
politifact_dtm_train <- as.matrix(politifact_dtm_train)
politifact_dtm_test <- as.matrix(politifact_dtm_test)
 