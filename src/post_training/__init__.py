"""
Post-Training Optimization Module - Fine-tuning and alignment.

Provides:
- LoRA / QLoRA / Advanced LoRA variants for efficient fine-tuning
- PEFT methods (Adapter, Prefix Tuning, Prompt Tuning)
- RLHF / DPO / ORPO / SimPO for alignment
- NEFTune for embedding noise regularization
- GaLore for memory-efficient full-parameter training
- Model Merging (TIES, DARE, SLERP)
- Continual Learning (EWC, Replay, Distillation)
"""

# Base LoRA
from src.post_training.lora import (
    LoRAConfig,
    LoRAModel,
    LoRALinear,
    apply_lora,
    merge_lora,
    QuantizedLoRA,
    QLoRAConfig,
)

# Advanced LoRA variants
from src.post_training.lora_advanced import (
    AdaLoRAConfig,
    AdaLoRALayer,
    AdaLoRATrainer,
    LoRAPlusConfig,
    create_lora_plus_optimizer,
    VeRAConfig,
    VeRALayer,
    LoRAXSConfig,
    LoRAXSLayer,
    apply_advanced_lora,
    count_lora_parameters,
)

# PEFT
from src.post_training.peft import (
    PEFTConfig,
    AdapterModel,
    PrefixTuning,
    PromptTuning,
)

# Alignment - Original
from src.post_training.alignment import (
    RLHFTrainer,
    DPOTrainer,
    DPOConfig,
    RewardModel,
    RewardConfig,
    PPOTrainer,
    PPOConfig,
    KTOTrainer,
    KTOConfig,
)

# Alignment - Advanced
from src.post_training.preference import (
    ORPOConfig,
    ORPOTrainer,
    SimPOConfig,
    SimPOTrainer,
    IPOConfig,
    IPOTrainer,
    select_preference_method,
)

# NEFTune
from src.post_training.neftune import (
    NEFTuneConfig,
    NEFTuneEmbedding,
    NEFTuneTrainer,
    apply_neftune,
    get_optimal_noise_alpha,
)

# GaLore
from src.post_training.galore import (
    GaLoreConfig,
    GaLoreProjector,
    GaLoreAdamW,
    GaLoreAdaFactor,
    create_galore_optimizer,
    estimate_galore_memory_savings,
)

# Model Merging
from src.post_training.merging import (
    MergeConfig,
    ModelMerger,
    linear_merge,
    slerp_merge,
    ties_merge,
    dare_merge,
    compute_model_similarity,
    compute_task_vector_stats,
)

# Continual Learning
from src.post_training.continual import (
    ContinualLearningConfig,
    EWCRegularizer,
    ReplayBuffer,
    ReplayTrainer,
    TaskAdapter,
    MultiTaskAdapterManager,
    DistillationLoss,
    ContinualDistillationTrainer,
    ContinualLearner,
)


__all__ = [
    # LoRA
    "LoRAConfig",
    "LoRAModel",
    "LoRALinear",
    "apply_lora",
    "merge_lora",
    "QuantizedLoRA",
    "QLoRAConfig",
    
    # Advanced LoRA
    "AdaLoRAConfig",
    "AdaLoRALayer",
    "AdaLoRATrainer",
    "LoRAPlusConfig",
    "create_lora_plus_optimizer",
    "VeRAConfig",
    "VeRALayer",
    "LoRAXSConfig",
    "LoRAXSLayer",
    "apply_advanced_lora",
    "count_lora_parameters",
    
    # PEFT
    "PEFTConfig",
    "AdapterModel",
    "PrefixTuning",
    "PromptTuning",
    
    # Alignment - Original
    "RLHFTrainer",
    "DPOTrainer",
    "DPOConfig",
    "RewardModel",
    "RewardConfig",
    "PPOTrainer",
    "PPOConfig",
    "KTOTrainer",
    "KTOConfig",
    
    # Alignment - Advanced
    "ORPOConfig",
    "ORPOTrainer",
    "SimPOConfig",
    "SimPOTrainer",
    "IPOConfig",
    "IPOTrainer",
    "select_preference_method",
    
    # NEFTune
    "NEFTuneConfig",
    "NEFTuneEmbedding",
    "NEFTuneTrainer",
    "apply_neftune",
    "get_optimal_noise_alpha",
    
    # GaLore
    "GaLoreConfig",
    "GaLoreProjector",
    "GaLoreAdamW",
    "GaLoreAdaFactor",
    "create_galore_optimizer",
    "estimate_galore_memory_savings",
    
    # Model Merging
    "MergeConfig",
    "ModelMerger",
    "linear_merge",
    "slerp_merge",
    "ties_merge",
    "dare_merge",
    "compute_model_similarity",
    "compute_task_vector_stats",
    
    # Continual Learning
    "ContinualLearningConfig",
    "EWCRegularizer",
    "ReplayBuffer",
    "ReplayTrainer",
    "TaskAdapter",
    "MultiTaskAdapterManager",
    "DistillationLoss",
    "ContinualDistillationTrainer",
    "ContinualLearner",
]
