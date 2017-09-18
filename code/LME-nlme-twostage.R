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
'structuretaxvaluedollarcnt','taxvaluedollarcnt','landtaxvaluedollarcnt','taxamount','yearbuilt')

# categorical variables
cat_atts = c('airconditioningtypeid','architecturalstyletypeid',
            'buildingclasstypeid','heatingorsystemtypeid','pooltypeid10','pooltypeid2',
            'pooltypeid7','propertylandusetypeid','regionidcounty',
            'storytypeid','typeconstructiontypeid','fireplaceflag',
            'taxdelinquencyflag')

### Dealing with missing values
# fill in missing values with mean for numeric variables
for (i in 1:length(num_atts)) {
  data[[num_atts[i]]][is.na(data[[num_atts[i]]])] <- mean(data[[num_atts[i]]], na.rm=TRUE)
}

# fill in missing values of categorical variables with label -1 and factorize 
for (i in 1:length(cat_atts)) {
  if (class(data[[cat_atts[i]]]) == "factor") {
    levels(data[[cat_atts[i]]])[levels(data[[cat_atts[i]]]) == ""] <- "-1"
  } else {
    data[[cat_atts[i]]][is.na(data[[cat_atts[i]]])] <- "-1"
    data[[cat_atts[i]]] <- factor(data[[cat_atts[i]]])    
  }
}

num_vars = c('calculatedfinishedsquarefeet','bathroomcnt','structuretaxvaluedollarcnt','bedroomcnt')

### Training a linear mixed-effects model
traindata_overestimate <- data[data$logerror >= 0,]
lme_over.fit = lme(logerror ~ calculatedfinishedsquarefeet + bedroomcnt + bathroomcnt + structuretaxvaluedollarcnt, data=traindata_overestimate,
              random = ~ 1 | regionidcounty)

traindata_underestimate <- data[data$logerror < 0,]
lme_under.fit = lme(logerror ~ calculatedfinishedsquarefeet + bedroomcnt + bathroomcnt + structuretaxvaluedollarcnt, data=traindata_underestimate,
                   random = ~ 1 | regionidcounty)


### Predictions on test set
# prepare test set
for (i in 1:length(num_atts)) {
  properties[[num_atts[i]]][is.na(properties[[num_atts[i]]])] <- mean(data[[num_atts[i]]], na.rm=TRUE)
}
for (i in 1:length(cat_atts)) {
  if (class(properties[[cat_atts[i]]]) == "factor") {
    levels(properties[[cat_atts[i]]])[levels(properties[[cat_atts[i]]]) == ""] <- "-1"
  } else {
    properties[[cat_atts[i]]][is.na(properties[[cat_atts[i]]])] <- "-1"
    properties[[cat_atts[i]]] <- factor(properties[[cat_atts[i]]], levels=levels(data[[cat_atts[i]]]))    
  }
}
# something funny goes on when filling in missing values for yearbuilt variable
properties[[cat_atts[12]]][is.na(properties[[cat_atts[12]]])] <- "-1"


logerror_mean_over = mean(traindata_overestimate$logerror)
logerror_mean_under = mean(traindata_underestimate$logerror)
remove(traindata_overestimate)
remove(traindata_underestimate)

# predict on test set and create out dataframe
# Making predictions for overestimated errors model
preds_over <- predict(lme_over.fit, properties, na.action = na.pass) # use na.pass to skip through NA test obs
preds_over[is.na(preds_over)] <- logerror_mean_over # some test observations don't even have regionidcounty

# Making predictions for underestimated errors model
preds_under <- predict(lme_under.fit, properties, na.action = na.pass) # use na.pass to skip through NA test obs
preds_under[is.na(preds_under)] <- logerror_mean_under

preds <- data.frame(properties[['parcelid']],preds_over,preds_under)
colnames(preds) <- c("parcelid",'lme_over',"lme_under")
# write dataframe
write.csv(preds, file="LME-nlme-two-stage-preds.csv", row.names = FALSE)

