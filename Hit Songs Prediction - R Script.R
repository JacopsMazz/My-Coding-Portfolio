######################################################################
#############################################################
############ GROUP 8 #############################
###### Hit Songs Prediction ################
###################################

# TABLE OF CONTENTS:

# 0. PRELIMINARY OPERATIONS

# 1. PREPROCESSING AND EXPLORATORY ANALYSIS

# 2. NATURAL LANGUAGE PROCESSING AND SENTIMENT ANALYSIS
# 2.1. LYRICS CLEANING AND EXPLORATION FOR SENTIMENT ANALYSIS
# 2.2 TEXT ANALYSIS
# 2.3 SENTIMENT ANALYSIS
# 2.3.1 SENTIMENT ANALYSIS WITH AFINN (NEGATIVE AND POSITIVE)
# 2.3.2 SENTIMENT ANALYSIS WITH NRC
# 2.4 TOPIC MODELLING (LDA)
# 2.4.1 CREATION OF THE CORPUS
# 2.4.2 TOKENIZATION AND PREPROCESSING
# 2.4.3 CREATE A DFM
# 2.4.4 LDA

# 3. RANDOM FOREST
# 3.1 EXPLORATORY MODEL
# 3.2 TUNING OF THE PARAMETERS
# 3.3 CROSS VALIDATION and FINAL MODEL
# 3.4 AUROC PLOT

########### 0. PRELIMINARY OPERATIONS

library(dummies)
library(data.table)
library(ggplot2)
library(ggpubr)
library(ggridges)
library(grid)
library(gridExtra)
library(GGally)
library(matrixStats) 
library(reshape2)
library(wordcloud)
library(tidytext)
library(tidyverse)
library(lexicon)
library(dplyr)
library(ROCR)
library(rsample)
library(caret)
library(ggrepel)
library(knitr) 
library(kableExtra) 
library(formattable)
library(ggraph)
library(circlize)
library(yarrr)
library(quanteda)
library(textdata)
library(tm)
library(SnowballC)
library(caTools)
library(rpart)
library(rpart.plot)
library(ROCR)
library(randomForest)
library(janitor)
library(dplyr) 
library(readtext)
library(stringr)
library(topicmodels)
library(gghighlight)
library(recipes)
library(corrplot)
library(RColorBrewer)
library(reprex)

mytheme <- theme_bw() + theme(
  plot.title = element_text(face="bold",hjust = 0.5),
  axis.title.x =element_text(face="bold"),
  axis.title.y =element_text(face="bold"),
  axis.text = element_text(size = rel(1)),
  legend.position = "bottom",
  legend.title=element_text(face="bold"))

########### 1. PREPROCESSING AND EXPLORATORY ANALYSIS

# Import the Spotify dataset with songs and relative characteristics (obtained from Kaggle)
spotify = fread("spotify_songs.csv")
names(spotify)

# Remove duplicates
spotify = distinct(spotify, track_id, .keep_all= TRUE)

# Import Last.fm dataset with artists popularity (Dataset obtained from Kaggle)
Last.fm_database = read.csv('Artists.csv', header = TRUE, sep = ',')
names(Last.fm_database)

# Remove duplicates in the columns "artist_mb" and then "artist_lastfm"
Last.fm_database = distinct(Last.fm_database, artist_mb, .keep_all= TRUE)
Last.fm_database = distinct(Last.fm_database, artist_lastfm, .keep_all= TRUE)

# Keep the relevant columns of the dataset
# The number of scrobbles is considered as a proxy of the artist popularity
Last.fm_database = Last.fm_database[c("artist_mb","scrobbles_lastfm","country_mb")]
colnames(Last.fm_database)[colnames(Last.fm_database) == "scrobbles_lastfm"] <- "artist_popularity"

# Merge the 2 datasets, so that you have all the relevant information in a single dataset
complete_dataset = merge.data.frame(spotify, Last.fm_database, by.x = 'track_artist', by.y = 'artist_mb')

# Drop irrelevant variables
complete_dataset = subset(complete_dataset, select = -c(track_album_id, track_album_name, playlist_id, playlist_name, track_id))

# remove missing values  
complete_dataset <-na.omit(complete_dataset)
complete_dataset <- complete_dataset %>% dplyr::filter(!(country_mb == ""))

# Log transformation of artist popularity and track popularity
complete_dataset$log_artist_popularity <- log(complete_dataset$artist_popularity+1)
complete_dataset$log_track_popularity <- log(complete_dataset$track_popularity+1)

# Preliminary analysis of track popularity and artist popularity
plot(density(complete_dataset$log_artist_popularity), main = "Density of log_artist_popularity")
polygon(density(complete_dataset$log_artist_popularity), col = "#1b98f0")
abline(v = mean(complete_dataset$log_artist_popularity), col = "red")
abline(v = median(complete_dataset$log_artist_popularity), col = "orange")
abline(v = quantile(complete_dataset$log_artist_popularity, 0.25), col = "green")
abline(v = quantile(complete_dataset$log_artist_popularity, 0.75), col = "darkgreen")
plot(density(complete_dataset$track_popularity), main = "Density of track_popularity")
polygon(density(complete_dataset$track_popularity), col = "#1b98f1")
abline(v = mean(complete_dataset$track_popularity), col = "red")
abline(v = median(complete_dataset$track_popularity), col = "orange")
abline(v = quantile(complete_dataset$track_popularity, 0.25), col = "green")
abline(v = quantile(complete_dataset$track_popularity, 0.75), col = "darkgreen")

summary(complete_dataset["track_popularity"])

# Plotting the distribution of track_popularity (10 classes histogram)
plot.H<-ggplot(complete_dataset, aes(x=track_popularity)) + mytheme + xlab("track_popularity") +
  geom_histogram(bins=10,fill="dodgerblue",color="black")
plot.H

# Dataset contains too many nationalities, for simplicity we summarize them according to macro-areas (Africa, Asia, Oceania, Center and South America)
# Create macro-categories for the nationality of the artist 
complete_dataset	<-	complete_dataset	%>%	
  mutate(artist_region	=	ifelse(country_mb=="Austria"|country_mb=="Germany"|country_mb=="Hungary"|country_mb=="Kosovo"|country_mb=="Monaco"|country_mb=="Poland"|country_mb=="Serbia"|country_mb=="Switzerland"|country_mb=="Turkey",	"Rest of Europe",
                                ifelse(country_mb=="Belgium"|country_mb=="France"|country_mb=="Netherlands"|country_mb=="Ireland","West Europe",
                                       ifelse(country_mb =="Greece"|country_mb == "Italy"|country_mb == "Portugal"|country_mb=="Romania"|country_mb == "Spain","South Europe",
                                              ifelse(country_mb =="Estonia"|country_mb =="Lithuania"|country_mb =="Moldova"|country_mb =="Russia","East and Baltic Europe",
                                                     ifelse(country_mb	=="United Kingdom",	"UK",
                                                            ifelse(country_mb=="Antigua and Barbuda"|country_mb=="Argentina"|country_mb=="Brazil"|country_mb=="Colombia"|country_mb=="Dominican Republic"|country_mb=="Jamaica"|country_mb=="Mexico"|country_mb=="Peru"|country_mb=="Trinidad and Tobago","Centre and south America",
                                                                   ifelse(country_mb=="Ghana"|country_mb=="Kenya"|country_mb=="Mali"|country_mb=="Nigeria"|country_mb=="Senegal"|country_mb=="South Africa"|country_mb=="Zambia","Africa",
                                                                          ifelse(country_mb=="United States","USA",
                                                                                 ifelse(country_mb=="Canada","Canada",
                                                                                        ifelse(country_mb=="","",
                                                                                               ifelse(country_mb=="Australia"|country_mb=="New Zealand","Oceania",
                                                                                                      ifelse(country_mb=="Denmark"|country_mb=="Faroe Islands"|country_mb=="Finland"|country_mb=="Iceland"|country_mb=="Norway"|country_mb=="Sweden","Scandinavia","Asia")))))))))))))
levels(complete_dataset$artist_region)

# Normalization of the variables
complete_dataset <- complete_dataset %>%
  recipe() %>% 
  step_normalize( danceability,
                  energy,
                  key,
                  loudness,
                  speechiness,
                  acousticness,
                  instrumentalness,
                  liveness,
                  valence,
                  tempo,
                  duration_ms ) %>% 
  prep() %>% 
  juice()

# Create a binary variable for popularity with 2 levels: popular, unpopular
complete_dataset	<-	complete_dataset	%>%	
  mutate(popularity_binary	=	ifelse(track_popularity	<	quantile(track_popularity,	0.5),	"unpopular", "popular"))
table(complete_dataset$popularity_binary)
complete_dataset$popularity_binary <- as.factor(complete_dataset$popularity_binary)

# Keep only English songs
complete_dataset <- complete_dataset[!(complete_dataset$language!="en"),]
complete_dataset = subset(complete_dataset, select = -c(language))

# Create a copy of the dataset to work on
Original_spotify <- complete_dataset

# Create a new column containing the release year of the song
Original_spotify$release_year <- as.numeric(substring(Original_spotify$track_album_release_date,1,4))

# Create a column indicating the decade of release
Original_spotify <- Original_spotify %>%
  mutate(decade=ifelse(release_year<1960, 1950,
                       ifelse(release_year<1970, 1960,
                              ifelse(release_year<1980, 1970,
                                     ifelse(release_year<1990, 1980,
                                            ifelse(release_year<2000, 1990,
                                                   ifelse(release_year<2010, 2000,
                                                          ifelse(release_year<2020, 2010,
                                                                 2020))))))))

# Distribution of genre popularity by decade
genre_plot <- Original_spotify %>% 
  group_by(decade, playlist_genre) %>%
  summarize(mean_popularity = mean(track_popularity, na.rm=TRUE))
ggplot(genre_plot,aes(x=decade,y=mean_popularity,colour=playlist_genre)) +
  geom_line(lwd=1.5) + labs(title = "Distribution of genre popularity by decade") +
  scale_x_continuous(breaks = c(1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020))

# Density plots of the song features
ggplot(Original_spotify) +
  geom_density(aes(energy, fill ="energy", alpha = 0.1), bw = 0.2) + 
  geom_density(aes(danceability, fill ="danceability", alpha = 0.1), bw = 0.2) + 
  geom_density(aes(loudness, fill ="loudness", alpha = 0.1), bw = 0.2) + 
  geom_density(aes(acousticness, fill ="acousticness", alpha = 0.1), bw = 0.2) + 
  geom_density(aes(instrumentalness, fill ="instrumentalness", alpha = 0.1), bw = 0.2) + 
  geom_density(aes(liveness, fill ="liveness", alpha = 0.1), bw = 0.2) + 
  labs(title = "Density plot of the song features") +
  xlab("Features") +
  theme_bw() +
  theme(plot.title = element_text(size = 10, face = "bold"), text = element_text(size = 10)) +
  theme(legend.title=element_blank()) +
  scale_fill_brewer(palette="Accent") + xlim(c(-5,5))

Original_spotify$decade <- as.factor(Original_spotify$decade)

# Correlation plot
songs_correlation <- cor(Original_spotify[,c(8:20)])
corrplot(songs_correlation, type = "upper", tl.srt = 45)

# Factorize categorical variables
complete_dataset$mode <- as.factor(complete_dataset$mode)

# playlist_genre / track_popularity violin plots
violins <- ggplot(Original_spotify, aes(x = playlist_genre, y = track_popularity))
violins + geom_violin(aes(fill = playlist_genre), trim = FALSE) + 
  geom_boxplot(width = 0.2)+
  ylim(0,100)+
  stat_summary(fun = mean, geom = "point",
               shape = 18, size = 2.5, color = "#FC4E07") +
  scale_fill_manual(values = c("#00AFBB", "#E7B800", "#FC4E07", "#FF33FF", "#FF9933","#D2691E"))+
  theme(legend.position = "none")


########### 2. NATURAL LANGUAGE PROCESSING AND SENTIMENT ANALYSIS

# 2.1. LYRICS CLEANING AND EXPLORATION FOR SENTIMENT ANALYSIS

# Fix multiple spacing
Original_spotify$lyrics = str_replace_all( Original_spotify$lyrics, 
                                           "\\s{2,100}", 
                                           " " )
# Fix pattern "word ,"
Original_spotify$lyrics = str_replace_all( Original_spotify$lyrics, 
                                           "\\s\\,", 
                                           "," )
# Fix "( " and " )"
Original_spotify$lyrics = str_replace_all( Original_spotify$lyrics, 
                                           "\\(\\s", 
                                           "(" )
Original_spotify$lyrics = str_replace_all( Original_spotify$lyrics, 
                                           "\\s\\)", 
                                           ")" )
# Final fix of multiple spacing
Original_spotify$lyrics = str_replace_all( Original_spotify$lyrics, 
                                           "\\s{2,100}", 
                                           " " )
# All cahracters lowercase
Original_spotify$lyrics <- tolower(Original_spotify$lyrics)

# Removal of numbers
Original_spotify$lyrics <- removeNumbers(Original_spotify$lyrics)

# Removal of punctuation
Original_spotify$lyrics <- removePunctuation(Original_spotify$lyrics)

# Removal of white spaces at the beginning and at the end of the lyric
Original_spotify$lyrics <- stripWhitespace(Original_spotify$lyrics)

# Breaking the phrases into single words with 1-gram
stopwords_en <- data.frame(word = tm::stopwords("english"))
class(stopwords("english"))
stopwords_text = c("cant","dont","take","this", "that", "make","can","cause","aint","give","your","youre","gonna","wanna","will","keep")
word <- stopwords_text
stopwords_lyrics <- data.frame(word)

unnested <- Original_spotify %>%
  select(lyrics, track_name, playlist_genre, decade) %>%
  unnest_tokens(word, lyrics, token = "ngrams", n = 1) %>%
  filter(!nchar(word) <= 3) %>% # Remove words shorter than 3 chars
  dplyr::anti_join(stopwords_en, by = c("word" = "word")) %>% # Removing stopwords
  dplyr::anti_join(stopwords_lyrics, by = c("word" = "word")) %>%
  mutate(word=wordStem(word)) # stem words

# 2.2 TEXT ANALYSIS

### Text analysis
# Count of each word appearing in the songs
unnested %>%
  dplyr::count(word) %>%
  arrange(desc(n)) %>%
  slice(1:20)

# Word count: 30 most common words (with graph)
unnested %>%
  group_by(playlist_genre) %>%
  dplyr::count(word) %>%
  # removing the top 0,1% of the words
  dplyr::filter(n < quantile(n, 0.9999)) %>%
  dplyr::top_n(n = 15) %>%
  ggplot(aes(reorder(word, n), n)) +
  geom_linerange(aes(ymin = min(n), ymax = n, x = reorder(word, n)),
                 position = position_dodge(width = 0.2), size = 1,
                 colour = 'lightblue') +
  geom_point(colour = 'dodgerblue4', size = 3, alpha = 0.9) +
  facet_wrap(~playlist_genre, scales = "free") +
  coord_flip() +
  labs(x = 'Top 15 most common words per genre', y = 'Count') +
  theme_bw(14)

# Word cloud
unnested %>%
  count(word) %>%
  dplyr::filter(n < quantile(n, 0.9999)) %>%
  with(wordcloud(word, n, family = "serif",
                 random.order = FALSE, max.words = 50,
                 colors = c("lightblue","dodgerblue4")))

# Word count: line graphs for bigrams
Bigrams <- Original_spotify %>%
  group_by(playlist_genre) %>%
  select(lyrics) %>%
  unnest_tokens(bigram, lyrics, token = "ngrams", n = 2) %>% 
  separate(bigram, c("word1", "word2"), sep = " ") %>% 
  filter(!word1 %in% stopwords_en$word,
         !is.na(word1), !is.na(word2),
         !word2 %in% stopwords_en$word) %>% 
  count(word1, word2, sort = TRUE) %>% 
  mutate(word = paste(word1, word2)) %>% 
  filter(n < quantile(n, 0.9999)) %>% arrange(desc(n)) %>%
  slice(1:15) %>%
  ggplot(aes(reorder(word, n), n)) +
  geom_linerange(aes(ymin = min(n), ymax = n, x = reorder(word, n)),
                 position = position_dodge(width = 0.2), size = 1,
                 colour = 'lightblue') +
  geom_point(colour = 'dodgerblue4', size = 3, alpha = 0.9) +
  facet_wrap(~playlist_genre, scales = "free") +
  coord_flip() +
  labs(x = 'Top 15 most common 2-grams', y = 'Count') +
  theme_bw(18)
Bigrams

# 2.3 SENTIMENT ANALYSIS

# 2.3.1 SENTIMENT ANALYSIS WITH AFINN (NEGATIVE AND POSITIVE)

# Words with sentiment going on a scale from -5 to +5
add_sentiments_afinn <- unnested %>%
  dplyr::inner_join(get_sentiments("afinn"), by = c("word" = "word"))

# To see the 15 most common words distinguishing per sentiment
add_sentiments_afinn %>%
  group_by(value) %>%
  count(word) %>%
  filter(n < quantile(n, 0.999)) %>%
  top_n(n = 15) %>%
  ggplot(aes(reorder(word, n), n)) +
  geom_linerange(aes(ymin = min(n), ymax = n, x = reorder(word, n)),
                 position = position_dodge(width = 0.2), size = 1,
                 colour = 'lightblue') +
  geom_point(colour = 'dodgerblue4', size = 3, alpha = 0.9) +
  facet_wrap(~value, scales = "free") +
  coord_flip() +
  labs(x = 'Top 15 most common words', y = 'Counts', title = 'Sentiments') +
  theme_bw(14)

# Songs' sentiment distribution from -5 to 5 (density plot)
summ_afinn <- add_sentiments_afinn %>%
  group_by(track_name) %>%
  summarise(mean_pol = mean(value))
summ_afinn %>%
  ggplot(aes(mean_pol)) + ggtitle("Sentiment Density") +
  geom_density(colour = 'dodgerblue4', fill = "lightblue",alpha = 0.8) +
  labs(y = 'Density', x = 'Sentiment') +
  theme_bw(14)

# Sentiment distribution per music genre (bar plot)
summ_afinn1 <- add_sentiments_afinn %>%
  group_by(playlist_genre) %>%
  summarise(mean_pol = mean(value))
plot_afinn <- summ_afinn1 %>%
  ggplot( aes(playlist_genre, mean_pol, fill = mean_pol)) +
  geom_col() +
  labs(x = "Genre", y = "Sentiment") +
  ggtitle("Sentiment per genre") +
  guides(fill = FALSE) +
  theme_bw(14)
plot_afinn

Original_spotify<-merge(Original_spotify,summ_afinn,by="track_name")

# 2.3.2 SENTIMENT ANALYSIS WITH NRC

add_sentiments_nrc <- unnested %>%
  dplyr::inner_join(get_sentiments("nrc"), by = c("word" = "word"))

add_sentiments_nrc %>%
  group_by(sentiment) %>%
  count(word) %>%
  filter(n < quantile(n, 0.999)) %>%
  top_n(n = 15) %>%
  ggplot(aes(reorder(word, n), n)) +
  geom_linerange(aes(ymin = min(n), ymax = n, x = reorder(word, n)),
                 position = position_dodge(width = 0.2), size = 1,
                 colour = 'darksalmon') +
  geom_point(colour = 'dodgerblue4', size = 3, alpha = 0.9) +
  facet_wrap(~sentiment, scales = "free") +
  coord_flip() +
  labs(x = 'Top 15 most common words', y = 'Counts', title = 'Sentiments') +
  theme_bw(14)

nrc_sentiments <- get_sentiments("nrc")

spotify_nrc <- unnested %>%
  inner_join(nrc_sentiments)

# by genre
spotify_genre_nrc <- spotify_nrc %>%
  count(sentiment, playlist_genre) %>%
  spread(playlist_genre, n, fill = 0)

spotify_genre_nrc = spotify_genre_nrc[-c(6,7), ]

plot_edm <- spotify_genre_nrc %>%
  ggplot( aes(sentiment, edm, fill = sentiment)) +
  geom_col() +
  labs(x = NULL, y = NULL) +
  ggtitle("EDM") +
  guides(fill = FALSE) +
  theme_bw()

plot_latin <- spotify_genre_nrc %>%
  ggplot( aes(sentiment, latin, fill = sentiment)) +
  geom_col() +
  labs(x = NULL, y = NULL) +
  ggtitle("Latin") +
  guides(fill = FALSE)+
  theme_bw()

plot_pop <- spotify_genre_nrc %>%
  ggplot( aes(sentiment, pop, fill = sentiment)) +
  geom_col() +
  labs(x = NULL, y = NULL) +
  ggtitle("Pop") +
  guides(fill = FALSE)+
  theme_bw()

names(spotify_genre_nrc)[names(spotify_genre_nrc) == "r&b"] <- "reb"
plot_reb <- spotify_genre_nrc %>%
  ggplot( aes(sentiment, reb, fill = sentiment)) +
  geom_col() +
  labs(x = NULL, y = NULL) +
  ggtitle("R&B") +
  guides(fill = FALSE)+
  theme_bw()

plot_rap <- spotify_genre_nrc %>%
  ggplot( aes(sentiment, rap, fill = sentiment)) +
  geom_col() +
  labs(x = NULL, y = NULL) +
  ggtitle("Rap") +
  guides(fill = FALSE)+
  theme_bw()

plot_rock <- spotify_genre_nrc %>%
  ggplot( aes(sentiment, rock, fill = sentiment)) +
  geom_col() +
  labs(x = NULL, y = NULL) +
  ggtitle("Rock") +
  guides(fill = FALSE)+
  theme_bw()

ggarrange(plot_edm, plot_latin, plot_pop, plot_reb, plot_rap, plot_rock, ncol=3, nrow=2)

# 2.4 TOPIC MODELLING (LDA)

# 2.4.1 CREATION OF THE CORPUS

lyrics = corpus (Original_spotify$lyrics)
summary(lyrics, 10)

N = ndoc( lyrics )
N

# Assigns a unique identifier to each text called "id_doc"
docvars( lyrics, "id_lyrics" ) = str_pad( 1L:N, width =5, pad = "0" )
docvars(lyrics, "playlist_genre") = Original_spotify$playlist_genre
docvars(lyrics, "decade") = Original_spotify$decade
summary(lyrics, 10)

# 2.4.2 TOKENIZATION AND PREPROCESSING
lyrics_tokens = tokens( lyrics,
                        remove_numbers = TRUE,
                        remove_punct = TRUE,
                        remove_symbols = TRUE,
                        remove_url = TRUE,
                        split_hyphens = TRUE)

lyrics_tokens = tokens_select( lyrics_tokens,
                               c("[\\d-]", "[[:punct:]]", "^.{1,2}$"),
                               selection = "remove",
                               valuetype = "regex",
                               verbose = TRUE )

lyrics_tokens=tokens_remove(lyrics_tokens, "\\b\\w{1,3}\\b", valuetype = "regex")

# 2.4.3 CREATE A DFM
stopwords()


lyrics_dfm = dfm( lyrics_tokens, # character, corpus, token, or another dfm object
                  tolower = TRUE,
                  stem = TRUE,
                  remove = stopwords("english"))

class( lyrics_dfm )

lyrics_dfm_trim = dfm_trim( lyrics_dfm,
                            # min 5%
                            min_docfreq = 0.05,
                            #  max 70%
                            max_docfreq = 0.95,
                            docfreq_type = "prop" ) 

class( lyrics_dfm_trim )

head( lyrics_dfm_trim, n = 5, nf = 10 )

# 2.4.4 LDA

n_topics = 3
dfmLDA = convert( lyrics_dfm, to = "topicmodels" )
LDAmodel = LDA( dfmLDA, control = list(seed=1970), k = n_topics )
perplexity(LDAmodel)
get_terms(LDAmodel, 15)
gamma = as.data.table( LDAmodel@gamma )

doc_vars = docvars( lyrics_dfm )
gamma[ , decade := doc_vars$decade]
gamma[ , playlist_genre := doc_vars$playlist_genre ]
gamma_molten = melt( gamma, 
                     id.vars = c( "decade", "playlist_genre" ),
                     value.name = "gamma",
                     variable.name = "topic",
                     measure.vars = paste0("V", 1:n_topics) )

gamma_molten = as.data.table(gamma_molten)

gamma_molten[ , `:=` (gamma_prop = gamma / sum(gamma ),
                      topic = as.factor( topic ) ), 
              by = .(playlist_genre) ]

# Plot proportions per genre
gamma_molten <- gamma_molten %>% mutate(topic=recode(topic, 
                                                     "V1"="uplifting",
                                                     "V2"="explicit",
                                                     "V3"="sentimental"))

# Plot distribution of topics in the Spotify Songs' Lyrics corpus
plot1 <- ggplot(gamma_molten, 
                aes( playlist_genre, gamma_prop, color = topic, fill = topic ) ) +
  geom_bar(stat = "identity") + 
  ggtitle("Distribution of topics in the Spotify Songs' Lyrics corpus") +
  xlab("") +
  ylab("Topic share (%)") +
  theme(legend.position = "right", 
        axis.text.x = element_text( angle = 90, vjust = .2, size = 6),
        axis.ticks.x = element_blank()) +
  theme_bw(14)
plot1

gamma_molten[ , `:=` (gamma_prop = gamma / sum(gamma ),
                      topic = as.factor( topic ) ), 
              by = .(decade) ]

#Distribution of topics by decade
plot2 <- gamma_molten %>% 
  group_by(decade, topic) %>%
  summarize(mean_gamma = mean(gamma, na.rm=TRUE))
ggplot(plot2,aes(x=decade,y=mean_gamma,group=topic, colour=topic)) +
  geom_line(lwd=1.5) + labs(title = "Distribution of topics by decade", y = "Topic share (%)") +
  scale_x_discrete(breaks = c(1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020))+
  theme_bw(14)

# insert the topics proportions as new vars in the dataset
Original_spotify <- cbind(Original_spotify, gamma)
Original_spotify = Original_spotify[, 1:31]
colnames(Original_spotify)[colnames(Original_spotify) == "V1"] <- "explicit"
colnames(Original_spotify)[colnames(Original_spotify) == "V2"] <- "sentimental"
colnames(Original_spotify)[colnames(Original_spotify) == "V3"] <- "uplifting"

########### 3. RANDOM FOREST

complete_dataset <- Original_spotify

## Let's plot songs popularity depending on their features
table(complete_dataset$popularity_binary)

spotify_songs <- complete_dataset %>% dplyr::select(-track_name, -track_artist, -lyrics, 
                                                    -track_popularity, -log_track_popularity, -artist_popularity,
                                                    -track_album_release_date, -country_mb, -playlist_genre, -decade)

# 3.1 EXPLORATORY MODEL

# Run an exploratory model in order to better understand what is the importance of the different variables
# Set train and test (80%-20%)
set.seed(2)
spotify_songs<-spotify_songs%>%dplyr::mutate(part	=	ifelse(runif(n())	>	0.20,	"train",	"test"))
spotify_songs	%>%	
  janitor::tabyl(part)

train	<-	spotify_songs	%>%	
  filter(part	==	"train")	%>%	
  select(-part)
test	<-	spotify_songs	%>%	
  filter(part	==	"test")	%>%	
  select(-part)

exploratory_model= randomForest(popularity_binary~.,data=train ,
                                importance =TRUE,ntree=100)
exploratory_model

importance_dat = exploratory_model$importance
importance_dat

# Exploration of the features importance
imp_exploratory	<-	randomForest::importance(exploratory_model)
imp_exploratory	<-	data.frame(var	=	dimnames(imp_exploratory)[[1]],	
                              value	=	imp_exploratory[,4])

imp_exploratory	%>%	
  arrange(var,	value)	%>%	
  mutate(var	=	fct_reorder(factor(var),		value,		min))	%>%	
  ggplot(aes(var,	value))	+
  geom_point(size	=	3.5,	colour	=	"darksalmon")	+
  coord_flip()	+
  labs(x	=	"Variables",	y	=	"Decrease in Gini criteria")	+
  theme_bw()

imp_exploratory 

# Prediction on the test
pred	<-	predict(exploratory_model,	test)
sum(pred	==	test$popularity_binary)/nrow(test)

# 3.2 TUNING OF THE PARAMETERS

# Grid search to understand what is the most promising combination of hyperparameters
# Set train and test (80%-20%)
set.seed(123)
spotify_songs<-spotify_songs%>%dplyr::mutate(part	=	ifelse(runif(n())	>	0.20,	"train",	"test"))
spotify_songs	%>%	
  janitor::tabyl(part)
train	<-	spotify_songs	%>%	
  filter(part	==	"train")	%>%	
  select(-part)
test	<-	spotify_songs	%>%	
  filter(part	==	"test")	%>%	
  select(-part)

models_error_rate <-data.frame()
test_error <-data.frame()
i=1
k=20
for (ntree in c(100,250,500,900,1300)) {
  for (mtry in (4:k)) {
    bag = randomForest(popularity_binary~.,data=train ,
                       mtry=mtry,importance =TRUE,ntree=ntree)
    
    models_error_rate[mtry,i]=bag$err.rate[ntree,1]
    
    pred	<-	predict(bag,	test)
    test_error[mtry,i] <- 1-sum(pred	==	test$popularity_binary)/nrow(test)
  }
  names(models_error_rate)[i]<-paste(ntree,"trees")
  names(test_error)[i]<-paste(ntree,"trees")
  i=i+1
}
models_error_rate["mtry"] = (1:k) #adds a column with mtry to the table, it will be used as X in the plot
test_error["mtry"] = (1:k)

# Plot of the models' error on the training set
ggplot(models_error_rate, aes(mtry, y)) +
  geom_line(aes(y = `100 trees`, col ="ntree = 100")) +
  geom_line(aes(y = `250 trees`,col ="ntree = 250")) +
  geom_line(aes(y = `500 trees`,col ="ntree = 500")) +
  geom_line(aes(y = `900 trees`,col ="ntree = 900")) +
  geom_line(aes(y = `1300 trees`,col ="ntree = 1300")) +
  labs(title = "Models' error on the training set", x = "mtry", y = "OOB Error") +
  scale_colour_manual("N* of trees", breaks = c("ntree = 100", "ntree = 250", "ntree = 500", "ntree = 900", "ntree = 1300"),
                      values = c("red", "green", "blue", "violet", "brown")) +
#  theme(legend.position = c(0.9, 0.8)) +
  scale_x_continuous(breaks=seq(1,k, by=1), limits = c(4,20), labels=c(1:k))+theme_bw()

# Plot of the models's error on the test
ggplot(test_error, aes(mtry, y)) +
  geom_line(aes(y = `100 trees`, col ="ntree = 100")) +
  geom_line(aes(y = `250 trees`,col ="ntree = 250")) +
  geom_line(aes(y = `500 trees`,col ="ntree = 500")) +
  geom_line(aes(y = `900 trees`,col ="ntree = 900")) +
  geom_line(aes(y = `1300 trees`,col ="ntree = 1300")) +
  labs(title = "Models' error on the test", x = "mtry", y = "Test Error") +
  scale_colour_manual("N* of trees", breaks = c("ntree = 100", "ntree = 250", "ntree = 500", "ntree = 900", "ntree = 1300"),
                      values = c("red", "green", "blue", "violet", "brown")) +
 # theme(legend.position = c(0.9, 0.8)) +
  scale_x_continuous(breaks=seq(1,k, by=1), limits = c(4, 20), labels=c(1:k))+theme_bw()

# 3.3 CROSS VALIDATION and FINAL MODEL

# Use the best combination of hyperparameters to run a random forest model with cross-validation
spotify_songs <- complete_dataset %>% dplyr::select(-track_name, -track_artist, -lyrics, 
                                                    -track_popularity, -log_track_popularity, -artist_popularity,
                                                    -track_album_release_date, -country_mb, -playlist_genre, -decade)

set.seed(283)
spotify_songs<-spotify_songs%>%dplyr::mutate(part	=	ifelse(runif(n())	>	0.20,	"train",	"test"))
spotify_songs	%>%	
  janitor::tabyl(part)
train	<-	spotify_songs	%>%	
  filter(part	==	"train")	%>%	
  select(-part)
test	<-	spotify_songs	%>%	
  filter(part	==	"test")	%>%	
  select(-part)

control <- trainControl(method='repeatedcv', 
                        number=5, 
                        repeats=5)

set.seed(212)

tunegrid <- expand.grid(.mtry=5)
final_model <- train(popularity_binary~., 
                     data=train, 
                     method='rf', 
                     metric='Accuracy', 
                     tuneGrid=tunegrid, 
                     trControl=control,
                     ntree=900)
print(final_model)

# Check of the prediction performances of the best model
final_prediction = predict(final_model, test)
final_test_error <- 1-sum(final_prediction	==	test$popularity_binary)/nrow(test)
  final_test_error
  
confusionMatrix(test$popularity_binary, final_prediction)
  
# 3.4 AUROC PLOT (Area Under the Receiver Operating Characteristics)

prediction_for_roc_curve <- predict(final_model,test,type="prob")

# Specify the different classes 
classes <- levels(test$popularity_binary)

# Define which observations belong to class[i]
true_values <- ifelse(test$popularity_binary==classes[1],1,0)

pred <- prediction(prediction_for_roc_curve[,1],true_values)
perf <- performance(pred, "tpr", "fpr")

plot(perf,main="ROC Curve",col="blue") + abline(coef = c(0,1))

# Calculate the AUC and print it to screen
auc.perf <- performance(pred, measure = "auc")
print(auc.perf@y.values)