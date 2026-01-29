"""
Advanced Post-Training Examples

Demonstrates the new post-training optimization techniques:
- NEFTune (Noisy Embeddings)
- ORPO/SimPO (No Reference Model Preference Optimization)
- GaLore (Memory-Efficient Full-Parameter Training)
- Model Merging (TIES, DARE, SLERP)
- Advanced LoRA (AdaLoRA, VeRA, LoRA+)
- Continual Learning (EWC, Replay)

Usage:
    python examples/advanced_post_training.py --mode neftune
    python examples/advanced_post_training.py --mode orpo
    python examples/advanced_post_training.py --mode galore
    python examples/advanced_post_training.py --mode merging
    python examples/advanced_post_training.py --mode continual
"""

import argparse
import torch
import torch.nn as nn
from dataclasses import dataclass


# =============================================================================
# Example Model
# =============================================================================

class SimpleTransformer(nn.Module):
    """A simple transformer for demonstration."""
    
    def __init__(self, vocab_size=10000, hidden_size=256, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # Named projections for LoRA targeting
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        self.lm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        
        # Apply projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        x = self.o_proj(q + k + v)  # Simplified
        
        for layer in self.layers:
            x = layer(x)
        
        logits = self.lm_head(x)
        return type('Output', (), {'logits': logits})()


def create_dummy_data(batch_size=8, seq_len=32, vocab_size=10000):
    """Create dummy data for demonstration."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    return input_ids, labels


# =============================================================================
# NEFTune Example
# =============================================================================

def neftune_example():
    """Demonstrate NEFTune for instruction fine-tuning."""
    print("\n" + "="*60)
    print("🔊 NEFTune Example - Noisy Embeddings")
    print("="*60)
    
    from src.post_training import (
        apply_neftune,
        NEFTuneConfig,
        get_optimal_noise_alpha,
    )
    
    # Create model
    model = SimpleTransformer()
    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Get recommended noise alpha
    alpha = get_optimal_noise_alpha("medium", "instruction")
    print(f"Recommended noise alpha: {alpha}")
    
    # Apply NEFTune
    neftune_trainer = apply_neftune(model, noise_alpha=alpha)
    
    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print("\nTraining with NEFTune...")
    for step in range(3):
        input_ids, labels = create_dummy_data()
        
        model.train()
        outputs = model(input_ids)
        loss = nn.functional.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1),
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Step {step+1}: Loss = {loss.item():.4f}")
    
    # Disable NEFTune for evaluation
    neftune_trainer.disable()
    print("NEFTune disabled for evaluation")


# =============================================================================
# ORPO/SimPO Example
# =============================================================================

def preference_example():
    """Demonstrate ORPO and SimPO preference optimization."""
    print("\n" + "="*60)
    print("🎯 ORPO/SimPO Example - Preference Optimization")
    print("="*60)
    
    from src.post_training import (
        ORPOTrainer, ORPOConfig,
        SimPOTrainer, SimPOConfig,
        select_preference_method,
    )
    
    # Create model
    model = SimpleTransformer()
    
    # Recommend method based on constraints
    method = select_preference_method(
        reference_model_available=False,
        data_quality="high",
        memory_constrained=True,
    )
    print(f"Recommended method: {method}")
    
    # === ORPO Training ===
    print("\n--- ORPO Training ---")
    orpo_config = ORPOConfig(lambda_orpo=0.1, learning_rate=1e-5)
    orpo_trainer = ORPOTrainer(model, config=orpo_config)
    
    # Simulate preference data
    chosen_ids, _ = create_dummy_data()
    rejected_ids, _ = create_dummy_data()
    
    for step in range(3):
        metrics = orpo_trainer.step(chosen_ids, rejected_ids)
        print(f"  Step {step+1}: Loss = {metrics['loss']:.4f}, "
              f"Accuracy = {metrics['accuracy']:.2%}")
    
    # === SimPO Training ===
    print("\n--- SimPO Training ---")
    model2 = SimpleTransformer()
    simpo_config = SimPOConfig(beta=2.0, gamma=0.5)
    simpo_trainer = SimPOTrainer(model2, config=simpo_config)
    
    for step in range(3):
        metrics = simpo_trainer.step(chosen_ids, rejected_ids)
        print(f"  Step {step+1}: Loss = {metrics['loss']:.4f}, "
              f"Reward Margin = {metrics['reward_margin']:.4f}")


# =============================================================================
# GaLore Example
# =============================================================================

def galore_example():
    """Demonstrate GaLore optimizer for memory-efficient training."""
    print("\n" + "="*60)
    print("📉 GaLore Example - Memory-Efficient Training")
    print("="*60)
    
    from src.post_training import (
        create_galore_optimizer,
        estimate_galore_memory_savings,
        GaLoreConfig,
    )
    
    # Create larger model for more noticeable savings
    model = SimpleTransformer(hidden_size=512, num_layers=6)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Estimate memory savings
    savings = estimate_galore_memory_savings(model, rank=128)
    print(f"\nMemory Estimates:")
    print(f"  Full Optimizer: {savings['full_optimizer_gb']:.3f} GB")
    print(f"  GaLore Optimizer: {savings['galore_optimizer_gb']:.3f} GB")
    print(f"  Savings Ratio: {savings['memory_savings_ratio']:.1f}x")
    
    # Create GaLore optimizer
    optimizer = create_galore_optimizer(
        model,
        lr=1e-4,
        rank=128,
        update_proj_gap=100,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    print("\nTraining with GaLore...")
    for step in range(3):
        input_ids, labels = create_dummy_data()
        
        model.train()
        outputs = model(input_ids)
        loss = nn.functional.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1),
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Step {step+1}: Loss = {loss.item():.4f}")


# =============================================================================
# Model Merging Example
# =============================================================================

def merging_example():
    """Demonstrate model merging techniques."""
    print("\n" + "="*60)
    print("🔀 Model Merging Example")
    print("="*60)
    
    from src.post_training import (
        ModelMerger,
        MergeConfig,
        linear_merge,
        slerp_merge,
        ties_merge,
        dare_merge,
        compute_model_similarity,
    )
    
    # Create base model and "fine-tuned" variants
    base_model = SimpleTransformer()
    
    # Simulate fine-tuned models by adding noise
    model_a = SimpleTransformer()
    model_b = SimpleTransformer()
    
    with torch.no_grad():
        for p in model_a.parameters():
            p.add_(torch.randn_like(p) * 0.01)
        for p in model_b.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    
    # Compute similarity
    sim = compute_model_similarity(model_a, model_b)
    print(f"\nModel A vs Model B:")
    print(f"  Cosine Similarity: {sim['cosine_similarity']:.4f}")
    print(f"  L2 Distance: {sim['l2_distance']:.4f}")
    
    # === Linear Merge ===
    print("\n--- Linear Merge ---")
    merged = linear_merge([model_a, model_b], weights=[0.6, 0.4])
    sim_merged = compute_model_similarity(merged, model_a)
    print(f"  Merged model similarity to A: {sim_merged['cosine_similarity']:.4f}")
    
    # === SLERP Merge ===
    print("\n--- SLERP Merge ---")
    slerp_merged = slerp_merge(model_a, model_b, t=0.5)
    sim_slerp = compute_model_similarity(slerp_merged, model_a)
    print(f"  SLERP merged similarity to A: {sim_slerp['cosine_similarity']:.4f}")
    
    # === TIES Merge ===
    print("\n--- TIES Merge ---")
    merger = ModelMerger(base_model)
    ties_merged = merger.ties_merge([model_a, model_b], threshold=0.2)
    sim_ties = compute_model_similarity(ties_merged, base_model)
    print(f"  TIES merged similarity to base: {sim_ties['cosine_similarity']:.4f}")
    
    # === DARE Merge ===
    print("\n--- DARE Merge ---")
    dare_merged = merger.dare_merge([model_a, model_b], drop_rate=0.9)
    sim_dare = compute_model_similarity(dare_merged, base_model)
    print(f"  DARE merged similarity to base: {sim_dare['cosine_similarity']:.4f}")


# =============================================================================
# Advanced LoRA Example
# =============================================================================

def advanced_lora_example():
    """Demonstrate advanced LoRA variants."""
    print("\n" + "="*60)
    print("🔧 Advanced LoRA Example")
    print("="*60)
    
    from src.post_training import (
        AdaLoRAConfig, AdaLoRATrainer,
        VeRALayer, VeRAConfig,
        LoRAXSLayer, LoRAXSConfig,
        create_lora_plus_optimizer,
        apply_advanced_lora,
        count_lora_parameters,
    )
    
    # === AdaLoRA ===
    print("\n--- AdaLoRA (Adaptive Rank) ---")
    model_adalora = SimpleTransformer()
    
    adalora_config = AdaLoRAConfig(
        init_r=12,
        target_r=8,
        target_modules=["q_proj", "v_proj"],
    )
    adalora_trainer = AdaLoRATrainer(model_adalora, adalora_config)
    adalora_trainer.apply_adalora()
    
    stats = count_lora_parameters(model_adalora)
    print(f"  Total params: {stats['total_parameters']:,}")
    print(f"  Trainable params: {stats['trainable_parameters']:,}")
    print(f"  Trainable ratio: {stats['trainable_ratio']:.2%}")
    
    # === VeRA ===
    print("\n--- VeRA (Shared Random Projections) ---")
    model_vera = SimpleTransformer()
    
    # Replace q_proj with VeRA
    vera_q = VeRALayer(model_vera.q_proj, r=256)
    model_vera.q_proj = vera_q
    
    # VeRA uses much fewer trainable parameters
    vera_trainable = sum(p.numel() for p in vera_q.parameters() if p.requires_grad)
    print(f"  VeRA trainable params (single layer): {vera_trainable:,}")
    
    # === LoRA-XS ===
    print("\n--- LoRA-XS (Ultra-Low Rank) ---")
    model_loraxs = SimpleTransformer()
    
    # Replace with r=1 LoRA
    loraxs_q = LoRAXSLayer(model_loraxs.q_proj, r=1, scale=4.0)
    model_loraxs.q_proj = loraxs_q
    
    loraxs_trainable = sum(p.numel() for p in loraxs_q.parameters() if p.requires_grad)
    print(f"  LoRA-XS trainable params (single layer): {loraxs_trainable:,}")
    
    # === LoRA+ ===
    print("\n--- LoRA+ (Different Learning Rates) ---")
    # Would need a model with LoRA applied first
    print("  LoRA+ uses lr_B = 16x * lr_A for faster convergence")


# =============================================================================
# Continual Learning Example
# =============================================================================

def continual_example():
    """Demonstrate continual learning techniques."""
    print("\n" + "="*60)
    print("📚 Continual Learning Example")
    print("="*60)
    
    from src.post_training import (
        ContinualLearner,
        ContinualLearningConfig,
        EWCRegularizer,
        ReplayBuffer,
    )
    
    model = SimpleTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # === EWC Example ===
    print("\n--- EWC (Elastic Weight Consolidation) ---")
    ewc = EWCRegularizer(model, lambda_=1000.0)
    
    # Train on Task 1
    print("Training on Task 1...")
    for step in range(3):
        input_ids, labels = create_dummy_data()
        
        outputs = model(input_ids)
        loss = nn.functional.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1),
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Consolidate (would normally compute Fisher on task 1 data)
    ewc.consolidate()
    print("Task 1 consolidated")
    
    # Train on Task 2 with EWC regularization
    print("Training on Task 2 with EWC...")
    for step in range(3):
        input_ids, labels = create_dummy_data()
        
        outputs = model(input_ids)
        task_loss = nn.functional.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1),
        )
        
        # Add EWC penalty
        ewc_penalty = ewc.penalty()
        total_loss = task_loss + ewc_penalty
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"  Step {step+1}: Task Loss = {task_loss.item():.4f}, "
              f"EWC Penalty = {ewc_penalty.item():.4f}")
    
    # === Replay Buffer Example ===
    print("\n--- Experience Replay ---")
    buffer = ReplayBuffer(max_size=1000)
    
    # Add samples
    for _ in range(10):
        data = create_dummy_data()
        buffer.add(data)
    
    print(f"Buffer size: {len(buffer)}")
    
    # Sample for replay
    replay_samples = buffer.sample(batch_size=4)
    print(f"Sampled {len(replay_samples)} samples for replay")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Advanced Post-Training Examples")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["neftune", "orpo", "galore", "merging", "lora", "continual", "all"],
        help="Which example to run",
    )
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 Advanced Post-Training Optimization Demo")
    print("="*60)
    
    examples = {
        "neftune": neftune_example,
        "orpo": preference_example,
        "galore": galore_example,
        "merging": merging_example,
        "lora": advanced_lora_example,
        "continual": continual_example,
    }
    
    if args.mode == "all":
        for name, func in examples.items():
            try:
                func()
            except Exception as e:
                print(f"\n❌ Error in {name}: {e}")
    else:
        examples[args.mode]()
    
    print("\n" + "="*60)
    print("✅ Demo complete!")
    print("="*60)


if __name__ == "__main__":
    main()
