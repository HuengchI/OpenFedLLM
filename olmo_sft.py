import os
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from peft import get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training, LoraConfig

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args

# ===== Define the arguments =====
script_args = get_config()
training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        gradient_checkpointing=script_args.gradient_checkpointing,
        deepspeed=script_args.deepspeed,
        local_rank=script_args.local_rank,
        optim=script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=script_args.warmup_ratio,
        run_name=script_args.run_name,
    )
peft_config = LoraConfig(
    r=script_args.peft_lora_r,
    lora_alpha=script_args.peft_lora_alpha,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["att_proj", "attn_output"] # for OLMO only
)

if is_main_process(script_args):
    save_config(script_args)
print(script_args)

# ===== Load the dataset =====

dataset = get_custom_local_dataset(script_args.custom_local_dataset,
                                       script_args.dataset_sample,
                                       post_df_loading_process_fn=lambda df: df.rename(columns={script_args.train_set_source_column: "template_source",
                                                                                                script_args.train_set_target_column: "template_target"}))

# ===== Get model config =====
device_map, quantization_config, torch_dtype, other_kwargs = get_model_config(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    **other_kwargs,
)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ===== Start training =====

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=data_collator,
)

print(f">> ==================== Training Start ====================")
results = trainer.train()

training_loss = results.training_loss

# ===== Save the model =====
if is_main_process(script_args):
    trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-final"))
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
