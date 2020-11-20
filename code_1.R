
library(adabag)
library(caret)
library(reticulate)
library(tidyverse)
library(tidytable)
library(Hmisc)
library(skimr)
library(kableExtra)
library(summarytools)

library(tableplot)
library(GGally)

library(e1071) 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()



# In this data, categorical vars are encoded and numericals are scaled
# Date variables are also included as numeric --> 'host_since_num' and 'last_review_num'
train_clean <- read.csv('train_clean.csv')
test_clean <- read.csv('test_clean.csv')

head(train_clean)

# Check how predictors relate to target variable 
featurePlot(x = train_clean[,-ncol(train_clean)], 
            y = as.factor(train_clean$price), 
            strip=strip.custom(par.strip.text=list(cex=.7)),
            plot = 'box',
            scales = list(x=list(relation="free"),
                          y=list(relation="free")))

####################################
######## LOAD DATA - RAW ###########
####################################

train <- read.csv('train.csv')
test <- read.csv('test.csv')


head(train)

###########################################
######## CLEAN OUTLIERS - TRAIN DATA ######
###########################################

# Numerical cols (but without 'host_since_num' and 'last_review_num' since we haven't created them yet)
# We deal with bathrooms, bedrooms and beds later
out_cols <- c("minimum_nights", "number_of_reviews", "reviews_per_month", 
            "calculated_host_listings_count", "availability_365", "bathrooms", 
            "cleaning_fee", "guests_included", "extra_people",
            "maximum_nights")

# Lets compute quantiles and IQR for each column in 'out_cols'
idx_num <- match(out_cols, colnames(train))

colnames(train)[idx_num]

cond_out <- lapply(idx_num, function(j){
  q <- quantile(train[,j], probs=c(.10, .90), na.rm = FALSE)
  iqr <- IQR(train[,j])
  low <- q[1]-1.5*iqr
  up <-  q[2]+1.5*iqr 
  train[,j] >= low & train[,j] <= up
})

cond_out_2 <- cond_out %>% reduce(`&`)


# We filter out all the outliers
train_no_out <- filter(train, cond_out_2)

write.csv(train_no_out, file = "train_no_out.csv", row.names = FALSE)

###########################################
######## CLEAN OUTLIERS - TEST DATA #######
###########################################

# Lets compute quantiles and IQR for each column in 'out_cols'
idx_num_test <- match(out_cols, colnames(test))

colnames(test)[idx_num_test]

cond_out <- lapply(idx_num_test, function(j){
  q <- quantile(test[,j], probs=c(.10, .90), na.rm = FALSE)
  iqr <- IQR(test[,j])
  low <- q[1]-1.5*iqr
  up <-  q[2]+1.5*iqr 
  test[,j] >= low & test[,j] <= up
})

cond_out_2 <- cond_out %>% reduce(`&`)


# We filter out all the outliers
test_no_out <- filter(test, cond_out_2)

write.csv(test_no_out, file = "test_no_out.csv", row.names = FALSE)

#### HER WE LOAD THE DATA TO PYTHON AND CLEAN (RESCALE AND ENCODE) #####


##################################################
######## LOAD DATA - NO OUTLIERS AND CLEAN #######
##################################################

train_no_out_clean <- read.csv('train_no_out_clean.csv')
test_no_out_clean <- read.csv('test_no_out_clean.csv')


##########################
######## BOXPLOTS ########
##########################

cols <- c("minimum_nights", "number_of_reviews", "reviews_per_month", 
         "calculated_host_listings_count", "availability_365", "bathrooms",
         "bedrooms", "beds", "cleaning_fee", "guests_included", "extra_people",
         "maximum_nights", "last_review_num", "host_since_num")

feat <- train_no_out_clean %>% select(cols)
target <- train_no_out_clean$price

# Check how predictors relate to target variable 
featurePlot(x = feat, 
            y = as.factor(target), 
            strip=strip.custom(par.strip.text=list(cex=.7)),
            plot = 'box',
            scales = list(x=list(relation="free"),
                          y=list(relation="free")))
  

# FeaturePlot(object = obj, "gene", split.by = "orig.ident") &
#   scale_colour_gradientn(colours =rev(brewer.pal(n = 10, name = "RdBu"))) & 
#   theme(text = element_text(face = "bold"),
#         axis.text.x=element_text(angle=45, hjust=1, size=30),
#         axis.title = element_text(size=30,face="bold"),
#         axis.title.y.right = element_text(size = SIZE HERE),
#         legend.text=element_text(size=50),
#         legend.title=element_text(size=50),
#         axis.line = element_line(size=2))

###################################
######## FANCY CORR MATRIX ########
###################################

# Restrictions / availability
cols_1 <- c("minimum_nights","availability_365", "guests_included", "extra_people",
          "maximum_nights")

g <- ggpairs(train_no_out_clean[,c(cols_1, 'price')], aes(colour = as.factor(price), alpha = 0.4), lower = "blank") +
  ggtitle("Group of Features 1 - Restrictions")
g



# Reviews
cols_2 <- c("number_of_reviews", "reviews_per_month","calculated_host_listings_count","last_review_num", "host_since_num")

g <- ggpairs(train_no_out_clean[,c(cols_2, 'price')], aes(colour = as.factor(price), alpha = 0.4), lower = "blank") +
  ggtitle("Group of Features 2 - Reviews")
g



# Characteristics
cols_3 <- c("bathrooms", "bedrooms", "beds", "cleaning_fee")

g <- ggpairs(train_no_out_clean[,c(cols_3, 'price')], aes(colour = as.factor(price), alpha = 0.4), lower = "blank") +
  ggtitle("Group of Features 3 - Characteristics")
g

#EXPLANATION OF ASTERISKS
# "***" if the p-value is < 0.001
# "**" if the p-value is < 0.01
# "*" if the p-value is < 0.05
# "." if the p-value is < 0.10
# "" otherwise

# MENTION THAT THE DENSITIES OF PRICE ARE NOT RELEVANT --> CATEGORICAL

########################################################
######## DISTRIBUTION OF PRICE PER NEIGHBORHOOD ########
########################################################

# With transparency (right)



g <- ggplot(data=train_no_out_clean, aes(x=cleaning_fee, group=price, fill=price)) +
  geom_density(adjust=1.5, alpha=.4) +
  scale_fill_viridis_c() + 
  ggtitle("Density - Cleaning Fee", subtitle = "After adjusting for outliers")
g






g <- ggplot(data=train_no_out_clean, aes(x=number_of_reviews, group=price, fill=price)) +
  geom_density(adjust=1.5, alpha=.4) +
  scale_fill_viridis_c() + 
  ggtitle("Density - Number of Reviews", subtitle = "After adjusting for outliers")
g






g <- ggplot(data=train_no_out_clean, aes(x=reviews_per_month, group=price, fill=price)) +
  geom_density(adjust=1.5, alpha=.4) +
  scale_fill_viridis_c() + 
  ggtitle("Density - Reviews per Month", subtitle = "After adjusting for outliers")
g




g <- ggplot(data=train_no_out_clean, aes(x=availability_365, group=price, fill=price)) +
  geom_density(adjust=1.5, alpha=.4) +
  scale_fill_viridis_c() + 
  ggtitle("Density - Availability_365", subtitle = "After adjusting for outliers")
g



g <- ggplot(data=train_no_out_clean, aes(x=maximum_nights, group=price, fill=price)) +
  geom_density(adjust=1.5, alpha=.4) +
  scale_fill_viridis_c() + 
  ggtitle("Density - Maximum Nights", subtitle = "After adjusting for outliers")
g


g <- ggplot(data=train_no_out_clean, aes(x=minimum_nights, group=price, fill=price)) +
  geom_density(adjust=1.5, alpha=.4) +
  scale_fill_viridis_c() + 
  ggtitle("Density - Minimum Nights", subtitle = "After adjusting for outliers")
g


########################################################
######## SVM from the 'ready' data from Python #########
########################################################
# https://medium.com/@ODSC/build-a-multi-class-support-vector-machine-in-r-abcdd4b7dab6#:~:text=An%20SVM%20performs%20classification%20tasks,support%20vector%20machine%20in%20R.

x <- read.csv('x_ready.csv')
y <- read.csv('y_ready.csv')
x_t <- read.csv('x_t_ready.csv')

data <- cbind(x, y$price)
colnames(data)[length(colnames(data))] <- "price"

n_train <- nrow(x)
n_test <- nrow(x_t)

train_indx <- createDataPartition(data$price, p = .8, 
                                  list = FALSE, 
                                  times = 1)

data_train <- data[train_indx,]
data_test  <- data[-train_indx,]


svm1 <- svm(as.factor(price)~., data=data_train, 
            method="C-classification", kernal="radial", 
            gamma=0.1, cost=10)

summary(svm1)

# No funca
c <- plot(svm1, data_train, cleaning_fee ~ number_of_reviews)
c

# Test with the test split of the train data
prediction <- predict(svm1, data_test)
xtab <- table(data_test$price, prediction)
xtab

# Compute prediction accuracy
confusionMatrix(as.factor(data_test$price), as.factor(prediction))


