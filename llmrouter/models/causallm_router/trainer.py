import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from llmrouter.models.base_trainer import BaseTrainer


class CausalLMTrainer(BaseTrainer):
    """
    CausalLMTrainer: Trainer for finetuning a Causal Language Model for LLM routing.

    Supports:
    - Full finetuning
    - LoRA finetuning (recommended for efficiency)
    """

    def __init__(self, router, optimizer=None, device="cuda"):
        """
        Initialize CausalLMTrainer.

        Args:
            router: CausalLMRouter instance
            optimizer: Optional custom optimizer
            device: Device to use for training
        """
        super().__init__(router=router, optimizer=optimizer, device=device)

        self.router = router
        self.device = device

        # Get config
        self.model_config = router.cfg.get("hparam", {})
        self.base_model_name = self.model_config.get("base_model", "meta-llama/Llama-2-7b-hf")

        # Get model paths
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_path_config = router.cfg.get("model_path", {})

        self.save_model_path = os.path.join(
            project_root,
            model_path_config.get("save_model_path", "saved_models/causallm_router")
        )

        # Initialize tokenizer and model
        self._init_model_and_tokenizer()

        print(f"[CausalLMTrainer] Initialized with base model: {self.base_model_name}")

    def _init_model_and_tokenizer(self):
        """Initialize tokenizer and model."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )

        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None
        )

        # Apply LoRA if enabled
        if self.model_config.get("use_lora", True):
            self._apply_lora()

    def _apply_lora(self):
        """Apply LoRA adapter to the model."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.model_config.get("lora_r", 16),
            lora_alpha=self.model_config.get("lora_alpha", 32),
            lora_dropout=self.model_config.get("lora_dropout", 0.1),
            target_modules=self.model_config.get(
                "lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj"]
            )
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def _prepare_dataset(self) -> Dataset:
        """Prepare dataset for training."""
        training_data = self.router.get_training_data()

        # Tokenize data
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["full_text"],
                truncation=True,
                max_length=self.model_config.get("max_length", 512),
                padding="max_length"
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        # Create dataset
        dataset = Dataset.from_list(training_data)
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        return tokenized_dataset

    def train(self):
        """
        Train the causal LM using HuggingFace Trainer.
        """
        # Prepare dataset
        print("[CausalLMTrainer] Preparing dataset...")
        train_dataset = self._prepare_dataset()

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.save_model_path,
            num_train_epochs=self.model_config.get("num_epochs", 3),
            per_device_train_batch_size=self.model_config.get("batch_size", 4),
            gradient_accumulation_steps=self.model_config.get("gradient_accumulation_steps", 4),
            learning_rate=self.model_config.get("learning_rate", 2e-5),
            weight_decay=self.model_config.get("weight_decay", 0.01),
            warmup_ratio=self.model_config.get("warmup_ratio", 0.1),
            logging_steps=self.model_config.get("logging_steps", 10),
            save_steps=self.model_config.get("save_steps", 100),
            save_total_limit=2,
            fp16=torch.cuda.is_available() and self.model_config.get("fp16", True),
            report_to=self.model_config.get("report_to", "none"),
            remove_unused_columns=False
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )

        # Train
        print("[CausalLMTrainer] Starting training...")
        trainer.train()

        # Save model
        print(f"[CausalLMTrainer] Saving model to {self.save_model_path}")
        trainer.save_model(self.save_model_path)
        self.tokenizer.save_pretrained(self.save_model_path)

        # Merge LoRA weights if using LoRA
        if self.model_config.get("use_lora", True) and self.model_config.get("merge_lora", True):
            self._merge_and_save_lora()

        print("[CausalLMTrainer] Training completed!")

    def _merge_and_save_lora(self):
        """Merge LoRA weights and save the full model."""
        print("[CausalLMTrainer] Merging LoRA weights...")

        merged_model = self.model.merge_and_unload()
        merged_path = os.path.join(self.save_model_path, "merged")

        merged_model.save_pretrained(merged_path)
        self.tokenizer.save_pretrained(merged_path)

        print(f"[CausalLMTrainer] Merged model saved to {merged_path}")



