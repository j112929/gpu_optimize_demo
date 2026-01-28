"""
Post-Training Optimization Module - Fine-tuning and alignment.

Provides:
- LoRA / QLoRA for efficient fine-tuning
- PEFT methods (Adapter, Prefix Tuning, Prompt Tuning)
- RLHF / DPO for alignment
"""

from src.post_training.lora import (
    LoRAConfig,
    LoRAModel,
    apply_lora,
    merge_lora,
    QuantizedLoRA,
)
from src.post_training.peft import (
    PEFTConfig,
    AdapterModel,
    PrefixTuning,
    PromptTuning,
)
from src.post_training.alignment import (
    RLHFTrainer,
    DPOTrainer,
    RewardModel,
    PPOConfig,
    DPOConfig,
)

__all__ = [
    # LoRA
    "LoRAConfig",
    "LoRAModel",
    "apply_lora",
    "merge_lora",
    "QuantizedLoRA",
    # PEFT
    "PEFTConfig",
    "AdapterModel",
    "PrefixTuning",
    "PromptTuning",
    # Alignment
    "RLHFTrainer",
    "DPOTrainer",
    "RewardModel",
    "PPOConfig",
    "DPOConfig",
]
