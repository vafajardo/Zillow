library(nlme)
library(MASS)

# This R script executes the stepwise selection procedure on Zillow data

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

# model selection
fit <- lm(as.formula(paste('logerror ~ ',paste(paste(cat_atts, collapse=" + "),paste(num_atts, collapse=" + "), sep=" + "), sep="")), data=data)
step <- stepAIC(fit, direction="both")
