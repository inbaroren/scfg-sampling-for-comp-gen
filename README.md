# Sampling Structurally-diverse Training Sets from Synthetic Data for Compositional Generalization in Semantic Parsing
Code for the paper: Finding needles in a haystack:Sampling Structurally-diverse Training Sets from Synthetic Data forCompositional Generalization.
The instructions will be updated soon. 

## Generate synthetic data
Please refer to https://github.com/stanford-oval/genie-toolkit/tree/master/starter/schemaorg to generate synthetic question, thingtalk query pairs.
The generated data will be in the "augmented.tsv" file.

## Sample UAT training sets w.r.t a compositional evaluation set
To sample training sets from your generated synthetic data w.r.t a compositional evaluation set, use scripts/create_splits.py.
The script performs 2 main operations: 
1. Clean examples that overlap with the compositional evaluation set in terms of templates 
2. Sample with UAT 

The synthetic data, evaluation development and test sets are passed as paths.
Use python scripts/create_splits.py -h for more details. 
 For example, to create UAT samples w.r.t the evaluation data in the data directory (program_dev.tsv and program_test.tsv), run:
 ```
python scripts/create_splits.py --augmented_path <path to your augmented.tsv file>   --training_size <choose training size for each sample>  --save_uat_samples
```  

When using the save_uat_samples flag the output file will contain the file in augmented_path but with additional 5 columns, each represents a sample. For example, the column "-1_1000_0_false" represents the first sample of size 1K, and has the value 1 for each example in the sample's training set, and 2 for each example in the sample's validation set.  
Possible error: during the splitting process validation examples that contain schema properties or constants that do not appear in the training set are removed. It is possible that when the training size is small this would cause the validation set to be empty. 