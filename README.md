# Sampling Structurally-diverse Training Sets from Synthetic Data for Compositional Generalization in Semantic Parsing
Code for the paper: Finding needles in a haystack:Sampling Structurally-diverse Training Sets from Synthetic Data forCompositional Generalization.
The instructions will be updated soon. 

## Generate synthetic data
Please refer to https://github.com/stanford-oval/genie-toolkit/tree/master/starter/schemaorg to generate synthetic question, thingtalk query pairs.
The generated data will be in the "augmented.tsv" file.

## Sample UAT training sets w.r.t a compositional evaluation set
To sample training sets from your generated synthetic data w.r.t a compositional evaluation set, use ```scripts/create_splits.py```.
The script performs 2 main operations: 
1. Remove examples with templates that appear in the compositional evaluation set
2. Sample with UAT

The synthetic data, evaluation development and test sets are passed as paths.
Use  ```python scripts/create_splits.py -h  ``` for more details. 
 For example, to create UAT samples w.r.t the evaluation data in the data directory (program_dev.tsv and program_test.tsv), run:
 ```
python scripts/create_splits.py --augmented_path <path to your augmented.tsv file>   --training_size <choose training size for each sample>  --create_training_pool --save_uat_samples
```  

When using the save_uat_samples flag the output file will contain the file in augmented_path but with additional 5 columns, each represents a sample. 
For example, the column "-1_1000_0_false" represents the first sample of size 1K, and has the value 1 for each example in the sample's training set, and 2 for each example in the sample's validation set.  

Validation set is empty error: during the splitting process validation examples that contain schema properties or constants that do not appear in the training set are removed. It is possible that when the training size is small this would cause the validation set to be empty.

To run a test example using the file ```data/small_synthetic_data.tsv```:
```
python scripts/create_splits.py --save_uat_samples
```  

When the create_training_pool flag is on the input data is reduced by removing any example with abstract template from the compositional evaluation set (data/program_*.tsv).
To save the result use the save_training_pool flag. 

## Train a model
We use allennlp BART implementation as the parser and use the allennlp framework to train it. 
To train the test example, use:
```
allennlp train experiments/bart-synthetic-data.json -s experiments/test_experiment --include-package models_code
```
You can use it to train on your data by using the overrides parameter, or edit ```experiments/bart-synthetic-data.json```. For example, the following command specifies which sample to use for training:
```
allennlp train experiments/bart-synthetic-data.json -s experiments/test_experiment --include-package models_code --overrides '{"dataset_reader":{"condition_name":"-1_20_0_false"}}'
```

#### Train the baseline parsers
To train one parser:
```
python training_scripts/train_baseline_models.py --seeds --absolute-path-project-root <path-to-project-root>
```
Use ```-h``` for more options. 
To train multiple models with different hyper parameters / seeds, you can use:
```
simple_hypersearch "python training_scripts/train_baseline_models.py --seeds --absolute-path-project-root <path-to-project-root> --seed-num {seed}" -p seed 0 1 2 3 4 | simple_gpu_scheduler --gpus 0,1,2,3
```

## Predict
To get models predictions use the command below. Replace the parameters 
```
allennlp predict <path>/model.tar.gz <path-to-input-tsv-file> --output-file <path-to-output-json-file> --use-dataset-reader --predictor my_seq2seq --include-package models_code
```
For example:
```
allennlp predict experiments/test_experiment/model.tar.gz data/small_uat_splits.tsv --use-dataset-reader --output-file data/small_uat_splits.json --predictor my_seq2seq  --include-package models_code
```

## Fine-tune
To fine-tune your parser on the manually annotated data use the following command. Replace the parameters ```<experiment-path>, <output-dir>```.
```
allennlp train <experiment-path>/config.json -s <output-dir> --include-package models_code --overrides  '{"train_data_path":"data/program_train.tsv","validation_data_path":"data/iid_dev.tsv","trainer":{"learning_rate_scheduler":{"warmup_steps":1500,"power":1},"optimizer":{"lr":0.000020},"num_epochs":30,"patience":10},"data_loader":{"batch_size":8},"dataset_reader":{"type":"my_seq2seq_pd","example_id_col":0,"utterance_col":1,"program_col":2,"read_header":null,"condition_name":null,"condition_value":null},"validation_dataset_reader":{"type":"my_seq2seq_pd","read_header":null,"condition_name":null,"condition_value":null,"example_id_col":0,"utterance_col":1,"program_col":2},"model":{"experiment_name":"test_experiment_finetuned","beam_size":4,"load_weights":true,"initializer":{"regexes":[[".*",{"type":"pretrained","weights_file_path":"<experiment-path>/best.th","parameter_name_overrides":{}}],]}}}'  
```
For example:
```
allennlp train experiments/bart-synthetic-data.json -s experiments/test_experiment_finetune  --include-package models_code --overrides  '{"train_data_path":"data/program_train.tsv","validation_data_path":"data/iid_dev.tsv","trainer":{"learning_rate_scheduler":{"warmup_steps":1500,"power":1},"optimizer":{"lr":0.000020},"num_epochs":30,"patience":10},"data_loader":{"batch_size":8},"dataset_reader":{"type":"my_seq2seq_pd","example_id_col":0,"utterance_col":1,"program_col":2,"read_header":null,"condition_name":null,"condition_value":null},"validation_dataset_reader":{"type":"my_seq2seq_pd","read_header":null,"condition_name":null,"condition_value":null,"example_id_col":0,"utterance_col":1,"program_col":2},"model":{"experiment_name":"test_experiment_finetuned","beam_size":4,"load_weights":true,"initializer":{"regexes":[[".*",{"type":"pretrained","weights_file_path":"experiments/test_experiment/best.th","parameter_name_overrides":{}}],]}}}'  
```
