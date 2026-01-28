"""
SGLang Structured Generation - Constrained output generation.

Provides tools for generating structured outputs like JSON,
choices, and regex-constrained text.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union
from abc import ABC, abstractmethod


@dataclass
class StructuredConfig:
    """Configuration for structured generation."""
    max_tokens: int = 1024
    temperature: float = 0.1  # Lower for structured output
    

class StructuredGenerator(ABC):
    """Base class for structured generation."""
    
    @abstractmethod
    def generate(self, client, prompt: str) -> Any:
        """Generate structured output."""
        pass
    
    @abstractmethod
    def get_constraint(self) -> Dict[str, Any]:
        """Get SGLang constraint config."""
        pass


class JsonGenerator(StructuredGenerator):
    """
    Generate JSON output with schema validation.
    
    Example:
        >>> generator = JsonGenerator({
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"},
        ...         "age": {"type": "integer"}
        ...     }
        ... })
        >>> result = generator.generate(client, "Generate a person profile")
    """
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
    
    def get_constraint(self) -> Dict[str, Any]:
        return {"json_schema": self.schema}
    
    def generate(self, client, prompt: str) -> Dict[str, Any]:
        """Generate JSON output."""
        from src.sglang_inference.client import GenerationConfig
        
        config = GenerationConfig(
            max_new_tokens=1024,
            temperature=0.1,
        )
        
        # Add JSON instruction
        full_prompt = f"{prompt}\n\nRespond with valid JSON only:\n"
        
        result = client.generate(full_prompt, config)
        
        # Parse JSON
        try:
            return json.loads(result.text.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from response
            text = result.text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
            raise ValueError(f"Failed to parse JSON: {result.text}")


class ChoiceGenerator(StructuredGenerator):
    """
    Generate output constrained to specific choices.
    
    Example:
        >>> generator = ChoiceGenerator(["positive", "negative", "neutral"])
        >>> sentiment = generator.generate(client, "Classify: I love this!")
    """
    
    def __init__(self, choices: List[str]):
        self.choices = choices
    
    def get_constraint(self) -> Dict[str, Any]:
        return {"choices": self.choices}
    
    def generate(self, client, prompt: str) -> str:
        """Generate from choices."""
        from src.sglang_inference.client import GenerationConfig
        
        choices_str = ", ".join(f'"{c}"' for c in self.choices)
        full_prompt = f"{prompt}\n\nRespond with exactly one of: {choices_str}\n"
        
        config = GenerationConfig(
            max_new_tokens=50,
            temperature=0.0,  # Deterministic
        )
        
        result = client.generate(full_prompt, config)
        text = result.text.strip().strip('"\'')
        
        # Find best match
        for choice in self.choices:
            if choice.lower() in text.lower():
                return choice
        
        return text


class RegexGenerator(StructuredGenerator):
    """
    Generate output matching a regex pattern.
    
    Example:
        >>> generator = RegexGenerator(r"\d{3}-\d{3}-\d{4}")
        >>> phone = generator.generate(client, "Generate a phone number")
    """
    
    def __init__(self, pattern: str):
        self.pattern = pattern
    
    def get_constraint(self) -> Dict[str, Any]:
        return {"regex": self.pattern}
    
    def generate(self, client, prompt: str) -> str:
        """Generate matching regex."""
        import re
        from src.sglang_inference.client import GenerationConfig
        
        full_prompt = f"{prompt}\n\nOutput format: {self.pattern}\n"
        
        config = GenerationConfig(
            max_new_tokens=100,
            temperature=0.1,
        )
        
        result = client.generate(full_prompt, config)
        
        # Extract matching portion
        match = re.search(self.pattern, result.text)
        if match:
            return match.group()
        
        return result.text.strip()


# =============================================================================
# Pydantic Integration
# =============================================================================

def json_schema_from_pydantic(model_class) -> Dict[str, Any]:
    """
    Generate JSON schema from Pydantic model.
    
    Example:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> schema = json_schema_from_pydantic(Person)
    """
    try:
        return model_class.model_json_schema()
    except AttributeError:
        # Pydantic v1
        return model_class.schema()


class PydanticGenerator(StructuredGenerator):
    """
    Generate output as a Pydantic model.
    
    Example:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> generator = PydanticGenerator(Person)
        >>> person = generator.generate(client, "Generate a person")
        >>> print(person.name, person.age)
    """
    
    def __init__(self, model_class: Type):
        self.model_class = model_class
        self.schema = json_schema_from_pydantic(model_class)
    
    def get_constraint(self) -> Dict[str, Any]:
        return {"json_schema": self.schema}
    
    def generate(self, client, prompt: str):
        """Generate Pydantic model instance."""
        json_gen = JsonGenerator(self.schema)
        data = json_gen.generate(client, prompt)
        return self.model_class(**data)


# =============================================================================
# Function Calling
# =============================================================================

@dataclass
class FunctionDef:
    """Function definition for function calling."""
    name: str
    description: str
    parameters: Dict[str, Any]


class FunctionCallGenerator:
    """
    Generate function calls from natural language.
    
    Example:
        >>> functions = [
        ...     FunctionDef(
        ...         name="get_weather",
        ...         description="Get weather for a city",
        ...         parameters={"city": {"type": "string"}}
        ...     )
        ... ]
        >>> generator = FunctionCallGenerator(functions)
        >>> call = generator.generate(client, "What's the weather in Tokyo?")
        >>> print(call)  # {"name": "get_weather", "arguments": {"city": "Tokyo"}}
    """
    
    def __init__(self, functions: List[FunctionDef]):
        self.functions = functions
    
    def _build_schema(self) -> Dict[str, Any]:
        """Build JSON schema for function call."""
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [f.name for f in self.functions]
                },
                "arguments": {"type": "object"}
            },
            "required": ["name", "arguments"]
        }
    
    def generate(self, client, prompt: str) -> Dict[str, Any]:
        """Generate function call."""
        # Build function descriptions
        func_descs = "\n".join([
            f"- {f.name}: {f.description}\n  Parameters: {json.dumps(f.parameters)}"
            for f in self.functions
        ])
        
        full_prompt = f"""Available functions:
{func_descs}

User request: {prompt}

Generate a function call as JSON with "name" and "arguments":
"""
        
        json_gen = JsonGenerator(self._build_schema())
        return json_gen.generate(client, full_prompt)


# =============================================================================
# List and Array Generation
# =============================================================================

class ListGenerator(StructuredGenerator):
    """Generate a list of items."""
    
    def __init__(self, item_schema: Dict[str, Any], min_items: int = 1, max_items: int = 10):
        self.item_schema = item_schema
        self.min_items = min_items
        self.max_items = max_items
    
    def get_constraint(self) -> Dict[str, Any]:
        return {
            "json_schema": {
                "type": "array",
                "items": self.item_schema,
                "minItems": self.min_items,
                "maxItems": self.max_items,
            }
        }
    
    def generate(self, client, prompt: str) -> List[Any]:
        """Generate list of items."""
        json_gen = JsonGenerator(self.get_constraint()["json_schema"])
        return json_gen.generate(client, prompt)
