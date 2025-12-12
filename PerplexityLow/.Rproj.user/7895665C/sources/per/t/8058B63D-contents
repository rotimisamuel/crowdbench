# Install necessary packages if not already installed
required_packages <- c("tm", "tidyverse", "topicmodels", "textclean", "ldatuning", "LDAvis", "textstem", "stringr", "SnowballC")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

# Load libraries
library(tm)
library(tidyverse)
library(topicmodels)
library(textclean)
library(ldatuning)
library(LDAvis)
library(textstem)
library(stringr)
library(SnowballC)

# Define file path
file_path <- "C:/Users/HP/OneDrive/Desktop/PerplexityLow"

# Get all CSV files in the folder
files <- list.files(path = file_path, pattern = "*.csv", full.names = TRUE)

# Check if file exists
if (!file.exists(file_path)) {
  stop("Error: File not found. Check the file path and try again.")
}

# Read CSV file
library(readr)
df <- read_csv("C:/Users/HP/OneDrive/Desktop/PerplexityLow/PerplexityAI_Negative.csv")

# Check if "content" column exists
if (!"content" %in% colnames(df)) {
  stop("Error: 'content' column not found in the dataset.")
}

# Remove NA values and empty rows in content column
df <- df %>% filter(!is.na(content) & content != "")

# Convert content column to character
df$content <- as.character(df$content)

# Create a VCorpus (instead of SimpleCorpus)
corpus <- VCorpus(VectorSource(df$content))

# Custom function to remove special characters
removeSpecialChars <- function(x) gsub("[^a-zA-Z ]", "", x)

# Apply transformations
corpus <- tm_map(corpus, content_transformer(tolower))  # Lowercase
corpus <- tm_map(corpus, content_transformer(removeSpecialChars))  # Remove special characters
corpus <- tm_map(corpus, removePunctuation)  # Remove punctuation
corpus <- tm_map(corpus, removeNumbers)  # Remove numbers
corpus <- tm_map(corpus, removeWords, stopwords("en"))  # Remove stopwords
corpus <- tm_map(corpus, stripWhitespace)  # Remove extra spaces
# corpus <- tm_map(corpus, stemDocument)  # Stemming
# **Apply Lemmatization**
corpus <- tm_map(corpus, content_transformer(lemmatize_strings))

# Convert back to plain text (avoids 'drops documents' warning)
corpus <- tm_map(corpus, PlainTextDocument)


# Create DTM
dtm <- DocumentTermMatrix(corpus, control = list(wordLengths = c(3, Inf)))

# Remove empty documents
rowTotals <- apply(as.matrix(dtm), 1, sum)
dtm <- dtm[rowTotals > 0, ]


# Check if DTM is empty
if (nrow(dtm) == 0) {
  stop("Error: No valid documents found after preprocessing.")
}

# Proceed with LDA modeling
library(topicmodels)
num_topics <- 5
lda_model <- LDA(dtm, k = num_topics, control = list(seed = 1234))

# View top words in each topic
terms(lda_model, 20)

# Load necessary libraries
library(tm)
library(topicmodels)
library(ggplot2)

# Set number of top words to extract
top_n_words <- 20

# Extract terms (words) for each topic
topic_terms <- terms(lda_model, top_n_words)

# Convert to data frame
topic_words_df <- data.frame(topic_terms)

# Save to CSV file
write.csv(topic_words_df, "C:/Users/HP/OneDrive/Desktop/PerplexityLow/LDA_Topics.csv", row.names = FALSE)

cat("LDA topics saved to CSV: C:/Users/HP/OneDrive/Desktop/PerplexityLow/LDA_Topics.csv\n")


# Get row sums of the Document-Term Matrix (total words per document)
doc.length <- rowSums(as.matrix(dtm))

# Ensure no empty documents (replace 0s with 1 to avoid errors)
doc.length[doc.length == 0] <- 1

# Get word frequency (total occurrences of each term across all docs)
term.frequency <- colSums(as.matrix(dtm))

# Load LDAvis
library(LDAvis)

# Convert LDA model output to JSON format for visualization
json_lda <- createJSON(
  phi = posterior(lda_model)$terms,
  theta = posterior(lda_model)$topics,
  doc.length = doc.length,    # Use the corrected doc.length
  vocab = Terms(dtm),
  term.frequency = term.frequency
)

# Visualize with LDAvis
serVis(json_lda, open.browser = TRUE)


# Save LDAvis output as an HTML file
serVis(json_lda, out.dir = "LDAvis_results", open.browser = FALSE)








#library(tidytext)
#topics <- tidy(lda_model, matrix = "beta")  # Get word probabilities per topic

# Select top 20 words per topic
#top_terms <- topics %>%
# group_by(topic) %>%
#slice_max(beta, n = 20) %>%
#ungroup()

# Plot the words in each topic
#ggplot(top_terms, aes(x = reorder_within(term, beta, topic), y = beta, fill = as.factor(topic))) +
#geom_col(show.legend = FALSE) +
#facet_wrap(~ topic, scales = "free") +
#coord_flip() +
#labs(title = "Top Terms Per Topic",
#    x = NULL, y = "Beta Value") +
#theme_minimal()


#library(topicmodels)

# Fit LDA model to the DTM
#lda_model <- LDA(dtm, k = 5, control = list(seed = 1234))  # Adjust 'k' based on your dataset

#topic_distributions <- posterior(lda_model)$topics

#pca_result <- prcomp(topic_distributions, center = TRUE, scale. = TRUE)
#pca_df <- as.data.frame(pca_result$x[, 1:3])  # Extract first 3 components
#colnames(pca_df) <- c("PC1", "PC2", "PC3")


