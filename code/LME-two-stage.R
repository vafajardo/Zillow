require(lme4)

# This basic R script prepares data for predictions,
# runs a basic LME and then writes the predictions to a .csv

# set working directory
setwd("/home/anerdi/Desktop/Zillow")

### load data
properties <- read.csv('./data/properties_2016.csv/properties_2016.csv')
logerror <- read.csv('./data/train_2016_v2.csv/train_2016_v2.csv')
properties['sqft_bucket'] = cut(properties$calculatedfinishedsquarefeet, c(0,1000,1500,2000,2500,5000,30000), labels=c(1:6))
properties['age_bucket'] = cut(properties$calculatedfinishedsquarefeet, c(1800,1950,1975,2000,2025), labels=c(1:4))
properties['ValueRatio'] = properties['taxvaluedollarcnt']/properties['taxamount']
properties['ValueProp'] = properties['structuretaxvaluedollarcnt']/properties['landtaxvaluedollarcnt']
properties['LivingAreaProp'] = properties['calculatedfinishedsquarefeet']/properties['lotsizesquarefeet']
data <- merge(properties, logerror, by="parcelid")
remove(logerror)

# numerical variables
num_atts = c('bathroomcnt','bedroomcnt','buildingqualitytypeid','calculatedbathnbr','finishedfloor1squarefeet',
             'calculatedfinishedsquarefeet','finishedsquarefeet12','finishedsquarefeet13',
             'finishedsquarefeet15','finishedsquarefeet50','finishedsquarefeet6','fireplacecnt',
             'fullbathcnt','garagecarcnt','garagetotalsqft','latitude','longitude','lotsizesquarefeet',
             'poolcnt','poolsizesum','censustractandblock','roomcnt','threequarterbathnbr','unitcnt',
             'yardbuildingsqft17','yardbuildingsqft26','numberofstories','yearbuilt',
             'structuretaxvaluedollarcnt','taxvaluedollarcnt','landtaxvaluedollarcnt','taxamount')

# categorical variables
cat_atts = c('airconditioningtypeid','architecturalstyletypeid',
             'buildingclasstypeid','heatingorsystemtypeid','pooltypeid10','pooltypeid2',
             'pooltypeid7','propertylandusetypeid','regionidcounty',
             'storytypeid','typeconstructiontypeid','fireplaceflag',
             'taxdelinquencyflag','sqft_bucket','age_bucket')

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

num_vars = c('calculatedfinishedsquarefeet','bathroomcnt','structuretaxvaluedollarcnt',
             'bedroomcnt','ValueRatio','ValueProp','LivingAreaProp')

traindata_overestimate <- data[data$logerror >= 0,]
traindata_overestimate[num_vars] <- scale(traindata_overestimate[num_vars])
lme_overestimate <- lmer(logerror ~ calculatedfinishedsquarefeet + bathroomcnt +  structuretaxvaluedollarcnt + bedroomcnt + 
              (1 + | regionidcounty) + (1 | sqft_bucket),
              data=traindata_overestimate)

traindata_underestimate <- data[data$logerror < 0,]
traindata_underestimate[num_vars] <- scale(traindata_underestimate[num_vars])
lme_underestimate <- lmer(logerror ~ calculatedfinishedsquarefeet + bathroomcnt +  structuretaxvaluedollarcnt + bedroomcnt + 
                           (1 + | regionidcounty) + (1 | sqft_bucket),
                         data=traindata_underestimate)

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


logerror_mean_over = mean(traindata_overestimate$logerror)
logerror_mean_under = mean(traindata_underestimate$logerror)
remove(traindata_overestimate)
remove(traindata_underestimate)

properties[num_vars] <- scale(properties[num_vars])

# Making predictions for overestimated errors model
preds_over <- predict(lme_overestimate, properties, na.action = na.pass, allow.new.levels = TRUE) # use na.pass to skip through NA test obs
preds_over[is.na(preds_over)] <- logerror_mean_over # some test observations don't even have regionidcounty

# Making predictions for underestimated errors model
preds_under <- predict(lme_underestimate, properties, na.action = na.pass, allow.new.levels = TRUE) # use na.pass to skip through NA test obs
preds_under[is.na(preds_under)] <- logerror_mean_under

preds <- data.frame(properties[['parcelid']],preds_over,preds_under)
colnames(preds) <- c("parcelid",'lme_over',"lme_under")
# write dataframe
write.csv(preds, file="LME-two-stage-preds.csv", row.names = FALSE)