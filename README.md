# Medical-LLM

# MedLlama Model Finetuning 


Dataset info: 

1. Chatdoctor - 10k 
2. Meadow Medical flashcards - 33k 
3. Medical wikidoc - 10k 
4. Medical Patient - 6k 
5. MediQA - 2.2k 
6. Cord-19 - 20k
7. Medical MMLU - 3.7k
8. Pubmed - 10k 

Total dataset size = 95k 

Dataset on huggingface: [medical-instruction-dataset](https://huggingface.co/datasets/anchorblock/medical_instruct_dataset) (Private Dataset)

Format of the dataset is given: ```medical_dataset.json```

# How to train 

1. Install requirements: 
    ```
    pip install -r requirements.txt
    ```
2. Train model using following command:
    ```
    python medalpaca/train.py     --model "/llm/model/llama-7b-hf"     --data_path /home/asif/llm/chatdoctor_10k.json     --output_dir 'medalpaca-model-2'   --num_epochs 5 --use_lora False --bf16 true --fp16 False --train_in_8bit False --global_batch_size 64 --warmup_steps 0 --per_device_batch_size 1 --group_by_length True --gradient_checkpointing True --optim "adafactor" --use_wandb False --save_total_limit 10 --model_max_length 512
    ```

# Hyperparameters 

For fine-tuning Llama-2 7b chat model, the hyperparameters used were: 

```
--num_epochs 1 
--use_lora False 
--bf16 true 
--fp16 False 
--train_in_8bit False 
--global_batch_size 64 
--warmup_steps 0 
--per_device_batch_size 1 
--group_by_length True 
--gradient_checkpointing True 
--optim "adafactor" 
--use_wandb False 
--save_total_limit 10 
--model_max_length 512

``` 

# Training details 

| Model Name  | Epochs | Training Time | VRAM
| ------------- | ------------- |-------- | --------- |
| Llama-2-7b-chat  | 1  |  22 hrs |  35 GB   |
| Llama-2-7b  |  1    |   20 hrs      |     35 GB     |


# Results 

**Llama-2-7b chat model:**

Validation loss after 1 epoch: 1.11 

Inference time (avg): 2s 

**Llama-2-7b base model:**

Validation loss after 1 epoch: 1.09 

Inference time (avg): 2-3s 

**Llama-2-7b finetuned model after 1 epoch can be found :** [here](https://huggingface.co/anchorblock/medllama-2-finetuned-1epoch) (Private repo)