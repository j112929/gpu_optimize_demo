"""
Ray Serve - Model serving and batch inference.

Provides scalable model deployment with auto-scaling.
"""

import ray
from ray import serve
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass 
class ServeConfig:
    """Configuration for Ray Serve deployment."""
    num_replicas: int = 1
    max_concurrent_queries: int = 100
    ray_actor_options: Dict[str, Any] = None
    autoscaling_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.ray_actor_options is None:
            self.ray_actor_options = {"num_gpus": 1}
        if self.autoscaling_config is None:
            self.autoscaling_config = {
                "min_replicas": 1,
                "max_replicas": 10,
                "target_num_ongoing_requests_per_replica": 5,
            }


class RayModelServer:
    """
    Ray Serve-based model server.
    
    Features:
    - Auto-scaling based on load
    - Batching for efficiency
    - GPU acceleration
    - Health checks
    
    Example:
        >>> server = RayModelServer(model, config)
        >>> server.deploy()
        >>> result = server.predict({"input": data})
        >>> server.shutdown()
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[ServeConfig] = None,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config or ServeConfig()
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self._handle = None
    
    def deploy(self, name: str = "model_server"):
        """Deploy the model as a Ray Serve endpoint."""
        if not ray.is_initialized():
            ray.init()
        
        serve.start()
        
        model = self.model
        preprocess = self.preprocess_fn
        postprocess = self.postprocess_fn
        
        @serve.deployment(
            name=name,
            num_replicas=self.config.num_replicas,
            max_concurrent_queries=self.config.max_concurrent_queries,
            ray_actor_options=self.config.ray_actor_options,
            autoscaling_config=self.config.autoscaling_config,
        )
        class ModelDeployment:
            def __init__(self):
                self.model = model.cuda().eval()
                self.preprocess = preprocess
                self.postprocess = postprocess
            
            async def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
                # Preprocess
                if self.preprocess:
                    inputs = self.preprocess(request)
                else:
                    inputs = request.get("input")
                
                # Inference
                with torch.no_grad():
                    if isinstance(inputs, torch.Tensor):
                        inputs = inputs.cuda()
                    outputs = self.model(inputs)
                
                # Postprocess
                if self.postprocess:
                    result = self.postprocess(outputs)
                else:
                    result = outputs.cpu().numpy().tolist()
                
                return {"output": result}
        
        self._handle = serve.run(ModelDeployment.bind())
        
        return self
    
    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction."""
        if self._handle is None:
            raise RuntimeError("Model not deployed. Call deploy() first.")
        
        return ray.get(self._handle.remote(request))
    
    async def predict_async(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Make an async prediction."""
        if self._handle is None:
            raise RuntimeError("Model not deployed. Call deploy() first.")
        
        return await self._handle.remote(request)
    
    def batch_predict(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make batch predictions."""
        futures = [self._handle.remote(req) for req in requests]
        return ray.get(futures)
    
    def shutdown(self):
        """Shutdown the server."""
        serve.shutdown()
        self._handle = None


def deploy_model(
    model: nn.Module,
    name: str = "model",
    num_replicas: int = 1,
    num_gpus: float = 1.0,
) -> Any:
    """
    Simple API to deploy a model.
    
    Args:
        model: PyTorch model
        name: Deployment name
        num_replicas: Number of replicas
        num_gpus: GPUs per replica
        
    Returns:
        Deployment handle
    """
    config = ServeConfig(
        num_replicas=num_replicas,
        ray_actor_options={"num_gpus": num_gpus},
    )
    
    server = RayModelServer(model, config)
    server.deploy(name)
    
    return server


# =============================================================================
# Batch Inference
# =============================================================================

@ray.remote(num_gpus=1)
class BatchInferenceActor:
    """Actor for batch inference on GPU."""
    
    def __init__(self, model_fn: Callable):
        self.model = model_fn().cuda().eval()
    
    def predict_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Run inference on a batch."""
        with torch.no_grad():
            batch = batch.cuda()
            outputs = self.model(batch)
        return outputs.cpu()


def batch_inference(
    model_fn: Callable,
    data: List[torch.Tensor],
    batch_size: int = 32,
    num_actors: int = 4,
) -> List[torch.Tensor]:
    """
    Run batch inference using multiple GPUs.
    
    Args:
        model_fn: Function that returns the model
        data: List of input tensors
        batch_size: Batch size
        num_actors: Number of GPU actors
        
    Returns:
        List of output tensors
    """
    if not ray.is_initialized():
        ray.init()
    
    # Create actors
    actors = [BatchInferenceActor.remote(model_fn) for _ in range(num_actors)]
    
    # Create batches
    batches = []
    for i in range(0, len(data), batch_size):
        batch = torch.stack(data[i:i+batch_size])
        batches.append(batch)
    
    # Distribute batches
    futures = []
    for i, batch in enumerate(batches):
        actor = actors[i % num_actors]
        futures.append(actor.predict_batch.remote(batch))
    
    # Collect results
    results = []
    for future in futures:
        batch_result = ray.get(future)
        results.extend([batch_result[i] for i in range(len(batch_result))])
    
    return results


# =============================================================================
# Streaming Inference
# =============================================================================

@serve.deployment
class StreamingModelDeployment:
    """Streaming model deployment for LLM inference."""
    
    def __init__(self, model_fn: Callable, tokenizer_fn: Callable):
        self.model = model_fn().cuda().eval()
        self.tokenizer = tokenizer_fn()
    
    async def __call__(self, request: Dict[str, Any]):
        """Stream tokens one at a time."""
        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 100)
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
        
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids)
                next_token_id = outputs.logits[:, -1, :].argmax(dim=-1)
            
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
            
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
            
            # Yield token
            yield self.tokenizer.decode(next_token_id)
