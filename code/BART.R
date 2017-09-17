options(java.parameters="-Xmx32g")
library(bartMachine)
set_bart_machine_num_cores(8)

# set working directory
setwd("/home/ubuntu/Zillow")

### load data
properties <- read.csv('./data/properties_2016.csv')
properties['ValueRatio'] = properties['taxvaluedollarcnt']/properties['taxamount']
properties['ValueProp'] = properties['structuretaxvaluedollarcnt']/properties['landtaxvaluedollarcnt']
properties['LivingAreaProp'] = properties['calculatedfinishedsquarefeet']/properties['lotsizesquarefeet']

logerror <- read.csv('./data/train_2016_v2.csv')
data <- merge(properties, logerror, by="parcelid")
remove(properties)

# numerical variables
num_atts = c('calculatedfinishedsquarefeet','bathroomcnt','structuretaxvaluedollarcnt',
             'bedroomcnt','regionidcounty','yearbuilt')

# categorical variables
cat_atts = c('airconditioningtypeid','heatingorsystemtypeid','pooltypeid10','pooltypeid2',
             'pooltypeid7','propertylandusetypeid','taxdelinquencyflag')

### Dealing with missing values
# fill in missing values with mean for numeric variables
for (i in 1:length(num_atts)) {
  data[[num_atts[i]]][is.na(data[[num_atts[i]]])] <- mean(data[[num_atts[i]]], na.rm=TRUE)
}

# garbage collection
gc()

### Training the bart machine
bart_machine <- bartMachine(data[1:nrow(data),c(num_atts,cat_atts)],data$logerror, serialize = TRUE,
                            use_missing_data = TRUE, use_missing_data_dummies_as_covars = TRUE)
save.image("bart_second.RData")
q("no")