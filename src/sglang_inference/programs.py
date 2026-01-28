"""
SGLang Programs - Composable LLM programs.

Provides building blocks for complex LLM workflows including
multi-turn chat, chain-of-thought, and few-shot learning.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from abc import ABC, abstractmethod


@dataclass
class Message:
    """Chat message."""
    role: str  # system, user, assistant
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class SGLProgram:
    """
    Composable LLM program using SGLang.
    
    Provides a fluent API for building complex LLM workflows.
    
    Example:
        >>> program = SGLProgram(client)
        >>> result = (program
        ...     .system("You are a helpful assistant")
        ...     .user("What is Python?")
        ...     .generate()
        ...     .user("Give me an example")
        ...     .generate()
        ...     .get_response())
    """
    
    def __init__(self, client=None, url: str = "http://localhost:30000"):
        if client is None:
            from src.sglang_inference.client import SGLangClient
            client = SGLangClient(url)
        
        self.client = client
        self.messages: List[Message] = []
        self.variables: Dict[str, str] = {}
        self._last_response: Optional[str] = None
    
    def system(self, content: str) -> "SGLProgram":
        """Add system message."""
        self.messages.append(Message("system", self._format(content)))
        return self
    
    def user(self, content: str) -> "SGLProgram":
        """Add user message."""
        self.messages.append(Message("user", self._format(content)))
        return self
    
    def assistant(self, content: str) -> "SGLProgram":
        """Add assistant message."""
        self.messages.append(Message("assistant", self._format(content)))
        return self
    
    def _format(self, content: str) -> str:
        """Format content with variables."""
        for key, value in self.variables.items():
            content = content.replace(f"{{{key}}}", value)
        return content
    
    def set(self, key: str, value: str) -> "SGLProgram":
        """Set a variable."""
        self.variables[key] = value
        return self
    
    def generate(
        self,
        max_tokens: int = 256,
        temperature: float = 0.7,
        store_as: Optional[str] = None,
    ) -> "SGLProgram":
        """
        Generate assistant response.
        
        Args:
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            store_as: Store response in this variable
        """
        from src.sglang_inference.client import GenerationConfig
        
        config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        
        messages_dict = [m.to_dict() for m in self.messages]
        result = self.client.chat(messages_dict, config)
        
        self._last_response = result.text
        self.messages.append(Message("assistant", result.text))
        
        if store_as:
            self.variables[store_as] = result.text
        
        return self
    
    def get_response(self) -> str:
        """Get last response."""
        return self._last_response or ""
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages."""
        return [m.to_dict() for m in self.messages]
    
    def get_variable(self, key: str) -> str:
        """Get variable value."""
        return self.variables.get(key, "")
    
    def fork(self) -> "SGLProgram":
        """Create a copy of this program."""
        new_program = SGLProgram(self.client)
        new_program.messages = self.messages.copy()
        new_program.variables = self.variables.copy()
        return new_program
    
    def clear(self) -> "SGLProgram":
        """Clear all messages and variables."""
        self.messages.clear()
        self.variables.clear()
        self._last_response = None
        return self


# =============================================================================
# Pre-built Programs
# =============================================================================

def multi_turn_chat(
    client,
    system_prompt: str = "You are a helpful assistant.",
    max_turns: int = 10,
) -> Callable:
    """
    Create a multi-turn chat function.
    
    Example:
        >>> chat = multi_turn_chat(client)
        >>> response = chat("Hello!")
        >>> response = chat("Tell me more")
    """
    program = SGLProgram(client).system(system_prompt)
    turn_count = [0]  # Use list to allow mutation in closure
    
    def chat(user_input: str) -> str:
        nonlocal program
        
        if turn_count[0] >= max_turns:
            # Start fresh after max turns
            program = SGLProgram(client).system(system_prompt)
            turn_count[0] = 0
        
        program.user(user_input).generate()
        turn_count[0] += 1
        
        return program.get_response()
    
    return chat


def chain_of_thought(
    client,
    question: str,
    examples: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, str]:
    """
    Chain-of-thought reasoning.
    
    Args:
        client: SGLang client
        question: Question to answer
        examples: Optional few-shot examples
        
    Returns:
        Dict with "reasoning" and "answer" keys
        
    Example:
        >>> result = chain_of_thought(client, "If I have 3 apples and buy 2 more, how many do I have?")
        >>> print(result["reasoning"])
        >>> print(result["answer"])
    """
    program = SGLProgram(client)
    
    system_msg = """You are a helpful assistant that thinks step by step.
When given a question, first explain your reasoning, then provide the final answer.

Format:
Reasoning: [your step-by-step thinking]
Answer: [final answer]"""
    
    program.system(system_msg)
    
    # Add examples if provided
    if examples:
        for ex in examples:
            program.user(ex["question"])
            program.assistant(f"Reasoning: {ex['reasoning']}\nAnswer: {ex['answer']}")
    
    program.user(question)
    program.generate(max_tokens=512, temperature=0.3)
    
    response = program.get_response()
    
    # Parse response
    reasoning = ""
    answer = ""
    
    if "Reasoning:" in response:
        parts = response.split("Answer:")
        reasoning = parts[0].replace("Reasoning:", "").strip()
        if len(parts) > 1:
            answer = parts[1].strip()
    else:
        answer = response
    
    return {"reasoning": reasoning, "answer": answer}


def few_shot_learning(
    client,
    examples: List[Dict[str, str]],
    input_key: str = "input",
    output_key: str = "output",
) -> Callable:
    """
    Create a few-shot learning function.
    
    Args:
        client: SGLang client
        examples: List of {"input": ..., "output": ...} examples
        input_key: Key for input in examples
        output_key: Key for output in examples
        
    Returns:
        Function that applies few-shot learning
        
    Example:
        >>> examples = [
        ...     {"input": "hello", "output": "HELLO"},
        ...     {"input": "world", "output": "WORLD"},
        ... ]
        >>> uppercase = few_shot_learning(client, examples)
        >>> result = uppercase("python")  # Returns "PYTHON"
    """
    def apply(new_input: str) -> str:
        program = SGLProgram(client)
        
        program.system("Follow the pattern shown in the examples exactly.")
        
        for ex in examples:
            program.user(ex[input_key])
            program.assistant(ex[output_key])
        
        program.user(new_input)
        program.generate(max_tokens=256, temperature=0.0)
        
        return program.get_response()
    
    return apply


# =============================================================================
# Advanced Programs
# =============================================================================

def self_consistency(
    client,
    question: str,
    num_samples: int = 5,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Self-consistency reasoning: sample multiple times and vote.
    
    Args:
        client: SGLang client
        question: Question to answer
        num_samples: Number of samples
        temperature: Sampling temperature
        
    Returns:
        Dict with "answer", "confidence", and "all_answers" keys
    """
    from collections import Counter
    
    answers = []
    
    for _ in range(num_samples):
        result = chain_of_thought(client, question)
        answers.append(result["answer"])
    
    # Vote
    counter = Counter(answers)
    most_common = counter.most_common(1)[0]
    
    return {
        "answer": most_common[0],
        "confidence": most_common[1] / num_samples,
        "all_answers": answers,
    }


def tree_of_thought(
    client,
    question: str,
    num_branches: int = 3,
    depth: int = 2,
) -> Dict[str, Any]:
    """
    Tree-of-thought reasoning with branching exploration.
    
    Args:
        client: SGLang client
        question: Question to explore
        num_branches: Branches per node
        depth: Tree depth
        
    Returns:
        Dict with best path and all explored paths
    """
    
    def evaluate_thought(thought: str) -> float:
        """Evaluate how promising a thought is."""
        program = SGLProgram(client)
        program.system("Rate the following reasoning step from 0-10. Respond with just a number.")
        program.user(f"Question: {question}\nThought: {thought}")
        program.generate(max_tokens=10, temperature=0.0)
        
        try:
            return float(program.get_response().strip())
        except:
            return 5.0
    
    def explore(current_thoughts: List[str], current_depth: int) -> List[Dict]:
        if current_depth >= depth:
            return [{"path": current_thoughts, "score": 0}]
        
        # Generate branches
        program = SGLProgram(client)
        program.system("Generate the next reasoning step. Be specific and logical.")
        
        context = f"Question: {question}\n"
        if current_thoughts:
            context += "Previous thoughts:\n" + "\n".join(f"- {t}" for t in current_thoughts)
        
        program.user(context + "\n\nNext thought:")
        
        branches = []
        for _ in range(num_branches):
            program_copy = program.fork()
            program_copy.generate(max_tokens=100, temperature=0.8)
            thought = program_copy.get_response()
            score = evaluate_thought(thought)
            branches.append((thought, score))
        
        # Sort by score and explore best branches
        branches.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for thought, score in branches[:num_branches]:
            sub_results = explore(current_thoughts + [thought], current_depth + 1)
            for r in sub_results:
                r["score"] += score
            results.extend(sub_results)
        
        return results
    
    all_paths = explore([], 0)
    all_paths.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "best_path": all_paths[0]["path"] if all_paths else [],
        "best_score": all_paths[0]["score"] if all_paths else 0,
        "all_paths": all_paths,
    }
