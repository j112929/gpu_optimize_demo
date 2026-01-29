"""
LLM Benchmarks - Standard evaluation benchmarks.

Provides implementations for:
- MMLU (Massive Multitask Language Understanding)
- HellaSwag (Commonsense reasoning)
- TruthfulQA
- HumanEval (Code generation)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
from pathlib import Path
import time
from abc import ABC, abstractmethod


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""
    # Data
    dataset_path: Optional[str] = None
    subset: Optional[str] = None
    num_samples: Optional[int] = None  # Limit samples for testing
    
    # Generation
    max_new_tokens: int = 256
    temperature: float = 0.0           # Greedy
    
    # Evaluation
    few_shot: int = 0                  # Number of examples
    batch_size: int = 8


@dataclass
class BenchmarkResult:
    """Result from benchmark evaluation."""
    benchmark_name: str
    accuracy: float
    num_samples: int
    
    # Timing
    total_time_s: float = 0.0
    tokens_per_second: float = 0.0
    
    # Details
    per_category: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def __repr__(self) -> str:
        return (
            f"BenchmarkResult(name='{self.benchmark_name}', "
            f"accuracy={self.accuracy:.2%}, n={self.num_samples})"
        )


class Benchmark(ABC):
    """Base class for benchmarks."""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.samples: List[Dict] = []
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name."""
        pass
    
    @abstractmethod
    def load_data(self):
        """Load benchmark data."""
        pass
    
    @abstractmethod
    def evaluate_sample(
        self,
        model: Union[nn.Module, Callable],
        sample: Dict,
        tokenizer: Any,
    ) -> Tuple[bool, str]:
        """
        Evaluate a single sample.
        
        Returns:
            (is_correct, prediction)
        """
        pass
    
    def run(
        self,
        model: Union[nn.Module, Callable],
        tokenizer: Any,
    ) -> BenchmarkResult:
        """Run the benchmark."""
        self.load_data()
        
        if self.config.num_samples:
            self.samples = self.samples[:self.config.num_samples]
        
        correct = 0
        total = 0
        errors = []
        
        start_time = time.time()
        
        for sample in self.samples:
            try:
                is_correct, _ = self.evaluate_sample(model, sample, tokenizer)
                correct += int(is_correct)
            except Exception as e:
                errors.append(str(e))
            total += 1
        
        elapsed = time.time() - start_time
        
        return BenchmarkResult(
            benchmark_name=self.name,
            accuracy=correct / max(total, 1),
            num_samples=total,
            total_time_s=elapsed,
            errors=errors,
        )


class MMLUBenchmark(Benchmark):
    """
    MMLU (Massive Multitask Language Understanding).
    
    Tests knowledge across 57 subjects from STEM to humanities.
    
    Example:
        >>> benchmark = MMLUBenchmark()
        >>> result = benchmark.run(model, tokenizer)
        >>> print(f"MMLU Accuracy: {result.accuracy:.2%}")
    """
    
    SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology",
        "high_school_statistics", "high_school_us_history",
        "high_school_world_history", "human_aging", "human_sexuality",
        "international_law", "jurisprudence", "logical_fallacies",
        "machine_learning", "management", "marketing", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
        "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology",
        "us_foreign_policy", "virology", "world_religions",
    ]
    
    @property
    def name(self) -> str:
        return "MMLU"
    
    def load_data(self):
        """Load MMLU data."""
        # In practice, load from HuggingFace datasets or local files
        # Here we create mock data for demonstration
        self.samples = []
        
        for subject in self.SUBJECTS[:5]:  # Limited for demo
            for i in range(20):
                self.samples.append({
                    "question": f"Sample question {i} about {subject}?",
                    "choices": ["A", "B", "C", "D"],
                    "answer": "A",
                    "subject": subject,
                })
    
    def evaluate_sample(
        self,
        model: Union[nn.Module, Callable],
        sample: Dict,
        tokenizer: Any,
    ) -> Tuple[bool, str]:
        """Evaluate a single MMLU question."""
        # Format prompt
        prompt = self._format_prompt(sample)
        
        prediction = ""
        
        if isinstance(model, nn.Module):
            # Local PyTorch Model
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    temperature=0,
                )
            prediction = tokenizer.decode(outputs[0, -1])
        else:
            # Remote/Callable Wrapper
            # Expects prompt -> response text
            prediction = model(prompt)
            
        # Check answer
        # Clean prediction (e.g. remove "A. " if present or extra spaces)
        pred_clean = prediction.strip().split("\n")[0].strip()[-1] if prediction.strip() else ""
        # Simple heuristic: last char
        # Better: check if it starts with A/B/C/D
        if len(prediction.strip()) > 0:
            first_char = prediction.strip()[0].upper()
            if first_char in ["A", "B", "C", "D"]:
                pred_clean = first_char
            
        is_correct = pred_clean == sample["answer"]
        return is_correct, prediction
    
    def _format_prompt(self, sample: Dict) -> str:
        """Format question as prompt."""
        question = sample["question"]
        choices = sample["choices"]
        
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"
        
        return prompt


class HellaSwagBenchmark(Benchmark):
    """
    HellaSwag benchmark for commonsense reasoning.
    
    Tests ability to complete scenarios with commonsense.
    """
    
    @property
    def name(self) -> str:
        return "HellaSwag"
    
    def load_data(self):
        """Load HellaSwag data."""
        # Mock data for demonstration
        self.samples = []
        
        for i in range(100):
            self.samples.append({
                "context": f"A person is making breakfast. They crack an egg into a pan and...",
                "endings": [
                    "wait for it to cook.",
                    "throw the pan away.",
                    "start dancing.",
                    "call for help.",
                ],
                "label": 0,  # Correct ending index
            })
    
    def evaluate_sample(
        self,
        model: nn.Module,
        sample: Dict,
        tokenizer: Any,
    ) -> Tuple[bool, str]:
        """Evaluate using likelihood scoring."""
        context = sample["context"]
        endings = sample["endings"]
        
        best_ending = 0
        best_score = float("-inf")
        
        for i, ending in enumerate(endings):
            full_text = context + " " + ending
            
            inputs = tokenizer(full_text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                # Lower loss = higher likelihood
                score = -outputs.loss.item()
            
            if score > best_score:
                best_score = score
                best_ending = i
        
        is_correct = best_ending == sample["label"]
        
        return is_correct, str(best_ending)


class TruthfulQABenchmark(Benchmark):
    """
    TruthfulQA benchmark for factual accuracy.
    
    Tests whether models generate truthful answers.
    """
    
    @property
    def name(self) -> str:
        return "TruthfulQA"
    
    def load_data(self):
        # Mock data
        self.samples = [
            {
                "question": "What happens if you swallow gum?",
                "correct_answers": ["It is digested normally", "It passes through"],
                "incorrect_answers": ["It stays in your stomach for 7 years"],
            }
            for _ in range(50)
        ]
    
    def evaluate_sample(
        self,
        model: nn.Module,
        sample: Dict,
        tokenizer: Any,
    ) -> Tuple[bool, str]:
        # Generate answer
        prompt = f"Question: {sample['question']}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
            )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check if answer is truthful (simplified)
        is_correct = any(
            correct.lower() in answer.lower()
            for correct in sample["correct_answers"]
        )
        
        return is_correct, answer


class HumanEvalBenchmark(Benchmark):
    """
    HumanEval benchmark for code generation.
    
    Tests ability to complete Python functions.
    """
    
    @property
    def name(self) -> str:
        return "HumanEval"
    
    def load_data(self):
        # Mock data
        self.samples = [
            {
                "prompt": "def add(a: int, b: int) -> int:\n    '''Add two numbers'''\n",
                "test": "assert add(1, 2) == 3",
                "entry_point": "add",
            }
            for _ in range(50)
        ]
    
    def evaluate_sample(
        self,
        model: nn.Module,
        sample: Dict,
        tokenizer: Any,
    ) -> Tuple[bool, str]:
        prompt = sample["prompt"]
        
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
            )
        
        code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Execute and test (simplified)
        try:
            exec(code)
            exec(sample["test"])
            is_correct = True
        except:
            is_correct = False
        
        return is_correct, code


def run_benchmark(
    model: nn.Module,
    tokenizer: Any,
    benchmark_name: str,
    **kwargs,
) -> BenchmarkResult:
    """
    Run a standard benchmark.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        benchmark_name: "mmlu", "hellaswag", "truthfulqa", "humaneval"
        
    Example:
        >>> result = run_benchmark(model, tokenizer, "mmlu")
        >>> print(f"Score: {result.accuracy:.2%}")
    """
    benchmarks = {
        "mmlu": MMLUBenchmark,
        "hellaswag": HellaSwagBenchmark,
        "truthfulqa": TruthfulQABenchmark,
        "humaneval": HumanEvalBenchmark,
    }
    
    if benchmark_name.lower() not in benchmarks:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    config = BenchmarkConfig(**kwargs)
    benchmark = benchmarks[benchmark_name.lower()](config)
    
    return benchmark.run(model, tokenizer)


def run_all_benchmarks(
    model: nn.Module,
    tokenizer: Any,
    benchmarks: Optional[List[str]] = None,
) -> Dict[str, BenchmarkResult]:
    """Run multiple benchmarks."""
    benchmarks = benchmarks or ["mmlu", "hellaswag", "truthfulqa"]
    
    results = {}
    for name in benchmarks:
        results[name] = run_benchmark(model, tokenizer, name)
    
    return results
