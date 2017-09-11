library(nlme)

# This basic R script prepares data for predictions,
# runs a basic LME and then writes the predictions to a .csv

# set working directory
setwd("/home/anerdi/Desktop/Zillow")

### load data
properties <- read.csv('./data/properties_2016.csv/properties_2016.csv')
logerror <- read.csv('./data/train_2016_v2.csv/train_2016_v2.csv')
data <- merge(properties, logerror, by="parcelid")

# numerical variables
num_atts = c('bathroomcnt','bedroomcnt','buildingqualitytypeid','calculatedbathnbr','finishedfloor1squarefeet',
'calculatedfinishedsquarefeet','finishedsquarefeet12','finishedsquarefeet13',
'finishedsquarefeet15','finishedsquarefeet50','finishedsquarefeet6','fireplacecnt',
'fullbathcnt','garagecarcnt','garagetotalsqft','latitude','longitude','lotsizesquarefeet',
'poolcnt','poolsizesum','censustractandblock','roomcnt','threequarterbathnbr','unitcnt',
'yardbuildingsqft17','yardbuildingsqft26','numberofstories',
'structuretaxvaluedollarcnt','taxvaluedollarcnt','landtaxvaluedollarcnt','taxamount')

# categorical variables
cat_atts = c('airconditioningtypeid','architecturalstyletypeid',
            'buildingclasstypeid','heatingorsystemtypeid','pooltypeid10','pooltypeid2',
            'pooltypeid7','propertylandusetypeid','regionidcounty',
            'storytypeid','typeconstructiontypeid','yearbuilt','fireplaceflag',
            'taxdelinquencyflag')

### Dealing with missing values
# fill in missing values with mean for numeric variables
for (i in 1:length(num_atts)) {
  data[[num_atts[i]]][is.na(data[[num_atts[i]]])] <- mean(data[[num_atts[i]]], na.rm=TRUE)
}

# fill in missing values of categorical variables with label -1 and factorize 
for (i in 1:length(cat_atts)) {
  data[[cat_atts[i]]] <- as.numeric(data[[cat_atts[i]]])
  data[[cat_atts[i]]][is.na(data[[cat_atts[i]]])] <- -1
  data[[cat_atts[i]]] <- as.factor(data[[cat_atts[i]]])
}

### Training a linear mixed-effects model
lme.fit = lme(logerror ~ calculatedfinishedsquarefeet + bedroomcnt + bathroomcnt + structuretaxvaluedollarcnt, data=data,
              random = ~ 1 | regionidcounty)

### Predictions on test set
# prepare test set
for (i in 1:length(num_atts)) {
  properties[[num_atts[i]]][is.na(properties[[num_atts[i]]])] <- mean(data[[num_atts[i]]], na.rm=TRUE)
}
for (i in 1:length(cat_atts)) {
  properties[[cat_atts[i]]] <- as.numeric(properties[[cat_atts[i]]])
  properties[[cat_atts[i]]][is.na(properties[[cat_atts[i]]])] <- -1
  properties[[cat_atts[i]]] <- as.factor(properties[[cat_atts[i]]])
}

# predict on test set and create out dataframe
preds <- predict(lme.fit, properties[c('calculatedfinishedsquarefeet','bedroomcnt','bathroomcnt','structuretaxvaluedollarcnt','regionidcounty')])
preds[is.na(preds)] <- lme.fit$coefficients$fixed["(Intercept)"] # some test observations don't even have regionidcounty
testout <- data.frame(properties[['parcelid']],preds,preds,preds,rep(0,length(preds)),rep(0,length(preds)),rep(0,length(preds)))
colnames(testout) <- c("ParcelId",'201610',"201611","201612","201710","201711","201712")
# write dataframe
write.csv(testout, file="test_predictions.csv", row.names = FALSE)
