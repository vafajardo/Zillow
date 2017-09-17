#!/bin/bash

rm BART_submission.csv
echo "ParcelId,201610,201611,201612,201710,201711,201712" > BART_submission.csv

for f in pred_BART*
do
  tail -n +2 $f >> BART_submission.csv
done
