## Bias Estimation in Classification assignement

## When you fit a model, like a logistic regression, what happens with class inbalance? We'll consider a data sets whose class balance will change over time, in a way that we might not anticipate, and investigate how the balance of predicted positives and negatives has a bias w.r.t. the actual class balance.
## You'll examine this bias as introduced by classification algorithms, e.g. logistoc regression. We'll examine two ways to correct the bias, one using a confusion matrix and a Bayesian approach.
## What you will learn The first method only works when the negative and positive predicted values remain constant, whereas the second method works when these ratios change.
## Imports and Definitions Beside importing several libraries, we'll define a function create_data that can create a supervised data set of N samples with a specific distribution, but with a different proportion ppos positive samples.
## Both the negative and positive samples are distributed standard-normal, except for the mean of the positive samples which is d_prime to allow for seperation between both.
## Author: Marco Puts and Piet Daas

##load libraries
library(ggplot2)

##Function to generate data
create_data <- function(d_prime, N, ppos){
  ##generate y
  y <- runif(N) <= ppos
  ##generate X
  X <- rnorm(N) + d_prime * y
  ##create df
  df <- data.frame("X"=X, "y"=y)
  return(df)
}

##function that creates a confusion matrix from 2 vectors
create_confusion_matrix <- function(yt, yp){
  ##make vectors
  actual_values <- as.vector(yt)
  predict_values <- as.vector(yp)
  ##create table  with structure TN, FN, FP, TP
  con_m <- table("Actual values"= factor(actual_values, levels=c("FALSE", "TRUE")), "Predicted values" = factor(predict_values, levels = c("FALSE", "TRUE")))
  return(con_m)
}

##plot nice looking confusion matrix
confusion_matrix_plot <- function(cm, title = ""){
  ##Create layout for data in cm
  TClass <- factor(c(FALSE, TRUE, FALSE, TRUE))
  PClass <- factor(c(FALSE, FALSE, TRUE, TRUE))
  Y      <- cm[c(1:4)]
  df <- data.frame(TClass, PClass, Y)
  
  ##show a nice coloured plot
  ggplot(data =  df, mapping = aes(x = factor(PClass, ordered=TRUE, levels = c(FALSE, TRUE)), y = factor(TClass, ordered=TRUE, levels = c(TRUE, FALSE)))) +
    geom_tile(aes(fill = Y), colour = "white") +
    geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
    scale_fill_gradient(low = "blue", high = "yellow") +
    labs(x = "Predicted class", y = "Actual class", title = title) +
    theme_bw() + theme(legend.position = "none")
}



## PART 1 (60% of the grade) ######################################################################################################################################################

## 1.1.  Example code: Create data, 100000 datapoints, positive ration of 0.5 and d_prime = 1
##generate data
data <- create_data(1, 100000, 0.5)
##show result
ggplot(data = data) +
  geom_histogram(data = subset(data, y == FALSE), aes(x = X, fill="False"), bins = 100) + 
  geom_histogram(data = subset(data, y == TRUE), aes(x = X, fill="True"), bins = 100) + 
  scale_fill_manual("y", values=c("blue", "orange"))


## 1.2. We are going to generate a training set X, y. X are the features (only one feature per case) and y gives the class where the case belongs to. 
# We will fit and test it against a test set X_t, y_t, which has a different (actual) proportion ppos.
## generate data
data2 <- create_data(1, 500, 0.5)
###Fit the balanced data set
log.model <- glm(y ~ ., data = data2, family = binomial)

##generate new data (with a different distribution)
data2_t <- create_data(1, 200, 0.3)
##predict y based on fitted log.model
y_prob <- predict(log.model, data2_t, type="response")
##Convert probability to TRUE andr FALSE
y_tp <- ifelse(y_prob > 0.5, TRUE, FALSE)
## Compare prediction with actual y via confusion matrix
cm <- create_confusion_matrix(data2_t$y, y_tp)
##show cm
cm

##Generate nice plot of confusion matrix
confusion_matrix_plot(cm)

## START OF THE ASSIGMENTS OF PART 1

## Question A: Why do the false posities and false negatives differ?

## Exercise B: Sum the rows and columns of the confusion matrix.

## Question C: What do the four resulting values mean?

## Question D: Why don't the rummed rows match the summed columns?


## 1.3. As shown in the presentation, we would like to correct for this difference. For that we need to calculate a matrix of
##  - the Negative Predicted Value,
##  - the False Discovery rate,
##  - the False Omission rate, and
##  - the Positive Predicted Value.
## Their definitions can be found on Wikipedia (link: https://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion).

## Exercise A: Calculate the four values according to their definitions.

##put the values in matrix M (you will need this later on)
M <- matrix(c(negative_predicted_value, false_omission_rate, false_discovery_rate, positive_predicted_value), byrow = FALSE, nrow = 2, ncol = 2)


## 1.4. Now, we generate a new dataset. we will not look at y_, since it will give us already the right answer. We will take y_p and see if we can reconstruct the right answer
data4 <- create_data(1,1000,0.3)
##Exercise A: predict probablity of y based on log.model, convert to TRUE and FALSE and compare true and estimated y


## 1.5. For the bias correction, we need to multiply matrix 𝑀 with the predictions made. For that we will create a diagonal matrix 𝑃 with the number of negative and positive predictions.
P <- matrix(c(sum(1-y_p), 0, 0, sum(y_p)), byrow = FALSE, nrow = 2, ncol = 2)

## Exercise A: Estimate the confusion matrix between y and y_p based on M and P.
## Note that this is an estimation, since M and P are based on different data sets from the same distribution.

## ExerciseB : Calculate the reconstructed
##  number of true positives, and
##  number of true negatives,
##  based on the confusion matrix cm_reconstructed.

## Exercise C: Estimate the threshold that belongs to the reconstructed negatives and positives and apply it to the model


##  1.6. Bonus questions and excersises
## Exercise A: Check what happens when you change the proportions of positives and negatives when generating X_ and y_.
  
##generate range of probability numbers (we use 41 between 0.1 and 0.9)
ppos <- seq(from = 0.1, to = 0.9, length.out = 41)
d_prime <- 1
N <- 10000

## Extra Bonus: Also add the original model to the plot.
## Question B: Does the logistic regression model behave properly? What does this imply?




## PART 2 (40% of the assignement) ######################################################################################################################################################

## 2.1. Bayesian Correction
## In this second part of the exercise, we are going to look at a Bayesian correction. We will follow the discription in https://arxiv.org/pdf/2102.08659.pdf
## For this, we first need to calculate the scores based on the result of the classifier.
##get probabilities
scores <- predict(log.model, data4, type="response")
##get probabilities for pos and neg cases
scores_n <- scores[!data4$y]
scores_p <- scores[data4$y]
##printe means
print(paste(mean(scores_p), mean(scores_n), sep = " "))

##Excersise A: In the subsequent step, we calculate the histograms of the positive and negative results. We will only use three bins for this.

##Excersise B: plot both histograms (barplot)


##2.2.  We will use X_ of the previous exercise. 
## Exercise A: Please calculate the probabilities scores using the log.model and only select the proportions for class 1.
## (i.e. do the same as before when using LR.predict_proba)


## Exercise B: Now we'll assign the scores to the right bins of the histogram and calculate the probabilities.
## Investigate how prop_n is calculated and mirror for prob_p.
## Note: The code first assigns bins to all scores for negative and then the positive samples:
##    scores between 0-0.33 go to bin 0,
##    scores between 0.33-0.66 to bin 1,
##    and the rest to bin 2.
## With the aforecalculated histograms on the same bins, prop_n estimates the probability that of the test sample scores in a particular bin according to the train set score distributions for negative and positive samples.

##generate 99 bins from 0.01 to 0.99
##Calculate log sums

##Question C: At which value of pi is the likelihood at a maximum? Also calculate the proportion negatives.

##Exercise D: Estimate the threshold that belongs to the found proportion positives.

##Question E: Compare this threshold with the confusion matrix, you calculated in the previous part. How do they compare?


##2.3 Bonus
##Exercise A: Check what happens when you change the proportions of positives and negatives when generating X_ and y_.

##Extra Bonus B: Also add the original model to the plot.

##Question C: How does the function you plotted here compare to the method of the previous part? Which one performs better?
  
