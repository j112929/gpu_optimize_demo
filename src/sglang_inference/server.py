"""
SGLang Server - High-performance LLM serving.

Provides model serving with RadixAttention, continuous batching,
and efficient KV cache management.
"""

import subprocess
import time
import requests
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path


@dataclass
class ServerConfig:
    """Configuration for SGLang server."""
    # Model
    model_path: str = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer_path: Optional[str] = None
    
    # Server
    host: str = "127.0.0.1"
    port: int = 30000
    
    # Performance
    tp_size: int = 1  # Tensor parallelism
    dp_size: int = 1  # Data parallelism
    max_total_tokens: int = 32768
    max_prefill_tokens: int = 8192
    max_running_requests: int = 256
    
    # Memory
    mem_fraction_static: float = 0.88
    context_length: Optional[int] = None
    
    # Features
    enable_flashinfer: bool = True
    disable_radix_cache: bool = False
    chunked_prefill_size: int = 8192
    
    # Quantization
    quantization: Optional[str] = None  # awq, gptq, fp8
    
    def to_args(self) -> List[str]:
        """Convert config to command line arguments."""
        args = [
            "--model-path", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--tp-size", str(self.tp_size),
            "--dp-size", str(self.dp_size),
            "--max-total-tokens", str(self.max_total_tokens),
            "--max-prefill-tokens", str(self.max_prefill_tokens),
            "--max-running-requests", str(self.max_running_requests),
            "--mem-fraction-static", str(self.mem_fraction_static),
            "--chunked-prefill-size", str(self.chunked_prefill_size),
        ]
        
        if self.tokenizer_path:
            args.extend(["--tokenizer-path", self.tokenizer_path])
        
        if self.context_length:
            args.extend(["--context-length", str(self.context_length)])
        
        if self.disable_radix_cache:
            args.append("--disable-radix-cache")
        
        if not self.enable_flashinfer:
            args.append("--disable-flashinfer")
        
        if self.quantization:
            args.extend(["--quantization", self.quantization])
        
        return args


class SGLangServer:
    """
    SGLang server manager for LLM inference.
    
    Features:
    - RadixAttention for efficient prefix caching
    - Continuous batching for high throughput
    - FlashInfer backend for fast attention
    - Multi-GPU tensor parallelism
    
    Example:
        >>> config = ServerConfig(model_path="meta-llama/Llama-2-7b-chat-hf")
        >>> server = SGLangServer(config)
        >>> server.start()
        >>> # Use server...
        >>> server.stop()
    """
    
    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()
        self._process: Optional[subprocess.Popen] = None
        self._is_running = False
    
    @property
    def url(self) -> str:
        """Get server URL."""
        return f"http://{self.config.host}:{self.config.port}"
    
    def start(self, wait: bool = True, timeout: int = 300) -> "SGLangServer":
        """
        Start the SGLang server.
        
        Args:
            wait: Wait for server to be ready
            timeout: Timeout in seconds
        """
        if self._is_running:
            print("Server is already running")
            return self
        
        cmd = ["python", "-m", "sglang.launch_server"] + self.config.to_args()
        
        print(f"Starting SGLang server: {' '.join(cmd)}")
        
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        
        if wait:
            self._wait_for_ready(timeout)
        
        self._is_running = True
        return self
    
    def _wait_for_ready(self, timeout: int):
        """Wait for server to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.url}/health")
                if response.status_code == 200:
                    print(f"✅ Server ready at {self.url}")
                    return
            except requests.exceptions.ConnectionError:
                pass
            
            time.sleep(2)
            print("Waiting for server to start...")
        
        raise TimeoutError(f"Server did not start within {timeout} seconds")
    
    def stop(self):
        """Stop the server."""
        if self._process:
            self._process.terminate()
            self._process.wait()
            self._process = None
        self._is_running = False
        print("Server stopped")
    
    def is_healthy(self) -> bool:
        """Check if server is healthy."""
        try:
            response = requests.get(f"{self.url}/health")
            return response.status_code == 200
        except:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        try:
            response = requests.get(f"{self.url}/get_model_info")
            return response.json()
        except:
            return {}
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


def launch_server(
    model_path: str,
    port: int = 30000,
    tp_size: int = 1,
    **kwargs,
) -> SGLangServer:
    """
    Quick function to launch an SGLang server.
    
    Args:
        model_path: HuggingFace model path
        port: Server port
        tp_size: Tensor parallelism size
        **kwargs: Additional ServerConfig options
        
    Returns:
        Running SGLangServer instance
    """
    config = ServerConfig(
        model_path=model_path,
        port=port,
        tp_size=tp_size,
        **kwargs,
    )
    
    server = SGLangServer(config)
    server.start()
    
    return server


def stop_server(server: SGLangServer):
    """Stop an SGLang server."""
    server.stop()


# =============================================================================
# Server Pool for Load Balancing
# =============================================================================

class ServerPool:
    """
    Pool of SGLang servers for load balancing.
    
    Example:
        >>> pool = ServerPool()
        >>> pool.add_server(config1)
        >>> pool.add_server(config2)
        >>> server = pool.get_server()  # Round-robin
    """
    
    def __init__(self):
        self.servers: List[SGLangServer] = []
        self._current_idx = 0
    
    def add_server(self, config: ServerConfig) -> SGLangServer:
        """Add a server to the pool."""
        server = SGLangServer(config)
        server.start()
        self.servers.append(server)
        return server
    
    def get_server(self) -> SGLangServer:
        """Get next server (round-robin)."""
        if not self.servers:
            raise RuntimeError("No servers in pool")
        
        server = self.servers[self._current_idx]
        self._current_idx = (self._current_idx + 1) % len(self.servers)
        return server
    
    def get_healthy_server(self) -> Optional[SGLangServer]:
        """Get a healthy server."""
        for server in self.servers:
            if server.is_healthy():
                return server
        return None
    
    def stop_all(self):
        """Stop all servers."""
        for server in self.servers:
            server.stop()
        self.servers.clear()
