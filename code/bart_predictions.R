options(java.parameters="-Xmx5g")
library(bartMachine)
set_bart_machine_num_cores(4)

# set working directory
setwd("/home/anerdi/Desktop/Zillow")

# load BART model
load("bart_first.RData")

### load data
properties <- read.csv('./data/properties_2016.csv/properties_2016.csv')
properties['ValueRatio'] = properties['taxvaluedollarcnt']/properties['taxamount']
properties['ValueProp'] = properties['structuretaxvaluedollarcnt']/properties['landtaxvaluedollarcnt']
properties['LivingAreaProp'] = properties['calculatedfinishedsquarefeet']/properties['lotsizesquarefeet']
logerror <- read.csv('./data/train_2016_v2.csv/train_2016_v2.csv')
data <- merge(properties, logerror, by="parcelid")

### Prepare properties for predictions
for (i in 1:length(num_atts)) {
  properties[[num_atts[i]]][is.na(properties[[num_atts[i]]])] <- mean(data[[num_atts[i]]], na.rm=TRUE)
}

# numerical variables
num_atts = c('calculatedfinishedsquarefeet','bathroomcnt','structuretaxvaluedollarcnt',
             'bedroomcnt','regionidcounty')

## Make predictions
print("Making predictions")
filenames <- paste(rep("pred_BART_"),c(1:30),rep(".csv"),sep="")
for (i in 0:28) {
  lower <- i*100000 + 1
  upper <- (i+1)*100000
  print(upper)
  test = properties[lower:upper, num_atts]
  preds <- predict(bart_machine, test, na.action = na.pass) # use na.pass to skip through NA test obs
  testout <- data.frame(properties[lower:upper,'parcelid'],preds,preds,preds,rep(0,length(preds)),rep(0,length(preds)),rep(0,length(preds)))
  colnames(testout) <- c("ParcelId",'201610',"201611","201612","201710","201711","201712")
  write.csv(testout, file=filenames[i+1], row.names = FALSE)
}

# set working directory
setwd("/home/anerdi/Desktop/Zillow/BART_preds")

# fencepost problem
test = properties[2900001:nrow(properties), num_atts]
preds <- predict(bart_machine, test, na.action = na.pass) # use na.pass to skip through NA test obs
testout <- data.frame(properties[2900001:nrow(properties),'parcelid'],preds,preds,preds,rep(0,length(preds)),rep(0,length(preds)),rep(0,length(preds)))
colnames(testout) <- c("ParcelId",'201610',"201611","201612","201710","201711","201712")
# write dataframe
write.csv(testout, file=filenames[30], row.names = FALSE)
length(preds)
