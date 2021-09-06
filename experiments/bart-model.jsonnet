local exp_path = "../thingtalk/crowdsourced_baselines/kbfree_untyped_noops";
local train_data = "/program_dev.tsv";
local dev_data = "/program_dev.tsv";

local target_namespace = "tokens";
local transformer_model = "facebook/bart-base";
local hidden_size = 768;

{
    "dataset_reader": {
        "type": "my_seq2seq_pd",
        "source_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model,
        },
        "source_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": transformer_model,
            },
        },
        "start_symbol": "<s>",
        "end_symbol": "</s>",
        "source_add_end_token": false,
        "source_add_start_token": false,
        "target_add_end_token": false,
        "target_add_start_token": false
    },
    "validation_dataset_reader": {
        "type": "my_seq2seq_pd",
        "source_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model,
        },
        "source_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": transformer_model,
            },
        },
        "start_symbol": "<s>",
        "end_symbol": "</s>",
        "source_add_end_token": false,
        "source_add_start_token": false,
        "target_add_end_token": false,
        "target_add_start_token": false
    },
    "train_data_path": exp_path + train_data,
    "validation_data_path": exp_path + dev_data,
    "model": {
        "type": "my_bart",
        "model_name": transformer_model,
        "max_decoding_steps": 140,
    },
    "data_loader": {
        "batch_size": 4,
        "shuffle": true
    },

    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 3e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "correct_bias": true
        },
        "learning_rate_scheduler": {
            "type": "polynomial_decay",
            "warmup_steps": 500
        },
        "tensorboard_writer": {
            "should_log_learning_rate": true
        },
        "patience": 2,
        "num_epochs": 2,
        "cuda_device": 0,
        "validation_metric": "+id_seq_acc",
        "checkpointer": {"num_serialized_models_to_keep": 1}
    }
}