library(nlme)

# This basic R script prepares data for predictions,
# runs a basic LME and then writes the predictions to a .csv

# set working directory
setwd("/home/anerdi/Desktop/Zillow")

### load data
properties <- read.csv('./data/properties_2016.csv/properties_2016.csv')
logerror <- read.csv('./data/train_2016_v2.csv/train_2016_v2.csv')
properties['sqft_bucket'] = cut(properties$calculatedfinishedsquarefeet, c(0,1000,1500,2000,2500,5000,30000), labels=c(1:6))
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

### Training a linear mixed-effects model
# lme.fit = lme(logerror ~ calculatedfinishedsquarefeet + bedroomcnt + bathroomcnt + structuretaxvaluedollarcnt, data=data,
#               random = ~ 1 | regionidcounty)
# final model after first iteration of stepwise

lme.fit = lmer(logerror ~  poly(calculatedfinishedsquarefeet,2) + bathroomcnt + 
                structuretaxvaluedollarcnt + bedroomcnt,
              data=data, random = ~ 1 + calculatedfinishedsquarefeet | regionidcounty)



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

# predict on test set and create out dataframe
preds <- predict(lme.fit, properties, na.action = na.pass) # use na.pass to skip through NA test obs
preds[is.na(preds)] <- mean(data$logerror) # some test observations don't even have regionidcounty
testout <- data.frame(properties[['parcelid']],preds,preds,preds,rep(0,length(preds)),rep(0,length(preds)),rep(0,length(preds)))
colnames(testout) <- c("ParcelId",'201610',"201611","201612","201710","201711","201712")
# write dataframe
write.csv(testout, file="LME.csv", row.names = FALSE)

