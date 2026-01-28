"""
Ray Cluster - Cluster management and monitoring utilities.

Provides tools for managing Ray clusters and resources.
"""

import ray
from ray.util.state import list_nodes, list_actors, list_tasks
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import time


@dataclass
class NodeInfo:
    """Information about a Ray node."""
    node_id: str
    node_ip: str
    state: str
    resources: Dict[str, float]
    resources_available: Dict[str, float]


@dataclass
class ClusterResources:
    """Cluster resource summary."""
    total_cpus: float
    available_cpus: float
    total_gpus: float
    available_gpus: float
    total_memory: float
    available_memory: float
    num_nodes: int


class RayCluster:
    """
    Ray cluster management utilities.
    
    Features:
    - Cluster status monitoring
    - Resource tracking
    - Node management
    - Auto-scaling control
    
    Example:
        >>> cluster = RayCluster()
        >>> cluster.connect("ray://head-node:10001")
        >>> print(cluster.status())
        >>> cluster.scale(num_workers=8)
    """
    
    def __init__(self, address: Optional[str] = None):
        self.address = address
        self._connected = False
    
    def connect(self, address: Optional[str] = None):
        """Connect to a Ray cluster."""
        addr = address or self.address
        
        if addr:
            ray.init(address=addr)
        elif not ray.is_initialized():
            ray.init()
        
        self._connected = True
        return self
    
    def disconnect(self):
        """Disconnect from cluster."""
        if ray.is_initialized():
            ray.shutdown()
        self._connected = False
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return ray.is_initialized()
    
    def get_nodes(self) -> List[NodeInfo]:
        """Get information about all nodes."""
        nodes = []
        for node in ray.nodes():
            if node["Alive"]:
                nodes.append(NodeInfo(
                    node_id=node["NodeID"],
                    node_ip=node["NodeManagerAddress"],
                    state="alive",
                    resources=node["Resources"],
                    resources_available=node.get("AvailableResources", {}),
                ))
        return nodes
    
    def get_resources(self) -> ClusterResources:
        """Get cluster resource summary."""
        resources = ray.cluster_resources()
        available = ray.available_resources()
        
        return ClusterResources(
            total_cpus=resources.get("CPU", 0),
            available_cpus=available.get("CPU", 0),
            total_gpus=resources.get("GPU", 0),
            available_gpus=available.get("GPU", 0),
            total_memory=resources.get("memory", 0) / (1024**3),  # GB
            available_memory=available.get("memory", 0) / (1024**3),
            num_nodes=len(self.get_nodes()),
        )
    
    def status(self) -> Dict[str, Any]:
        """Get cluster status."""
        resources = self.get_resources()
        nodes = self.get_nodes()
        
        return {
            "connected": self.is_connected(),
            "num_nodes": resources.num_nodes,
            "resources": {
                "cpus": f"{resources.available_cpus}/{resources.total_cpus}",
                "gpus": f"{resources.available_gpus}/{resources.total_gpus}",
                "memory_gb": f"{resources.available_memory:.1f}/{resources.total_memory:.1f}",
            },
            "nodes": [
                {
                    "id": n.node_id[:8],
                    "ip": n.node_ip,
                    "state": n.state,
                }
                for n in nodes
            ],
        }
    
    def print_status(self):
        """Print cluster status."""
        status = self.status()
        
        print("=" * 50)
        print("RAY CLUSTER STATUS")
        print("=" * 50)
        print(f"Connected: {status['connected']}")
        print(f"Nodes: {status['num_nodes']}")
        print(f"CPUs: {status['resources']['cpus']}")
        print(f"GPUs: {status['resources']['gpus']}")
        print(f"Memory: {status['resources']['memory_gb']} GB")
        print("-" * 50)
        print("Nodes:")
        for node in status["nodes"]:
            print(f"  {node['id']} | {node['ip']} | {node['state']}")
        print("=" * 50)
    
    def wait_for_nodes(self, num_nodes: int, timeout: int = 300):
        """Wait for nodes to be available."""
        start = time.time()
        while time.time() - start < timeout:
            current_nodes = len(self.get_nodes())
            if current_nodes >= num_nodes:
                print(f"✅ {current_nodes} nodes available")
                return True
            print(f"Waiting for nodes: {current_nodes}/{num_nodes}")
            time.sleep(5)
        
        raise TimeoutError(f"Timeout waiting for {num_nodes} nodes")


def get_cluster_info() -> Dict[str, Any]:
    """Get quick cluster info."""
    if not ray.is_initialized():
        ray.init()
    
    cluster = RayCluster()
    return cluster.status()


def scale_cluster(num_workers: int, wait: bool = True, timeout: int = 300):
    """
    Scale the cluster to specified number of workers.
    
    Note: This requires Ray Autoscaler or Kubernetes Ray Operator.
    """
    cluster = RayCluster()
    
    # Request scaling through placement group
    # This triggers autoscaler if configured
    bundles = [{"CPU": 1, "GPU": 1} for _ in range(num_workers)]
    pg = ray.util.placement_group(bundles, strategy="SPREAD")
    
    if wait:
        ready = ray.get(pg.ready())
        print(f"✅ Scaled to {num_workers} workers")
    
    return pg


# =============================================================================
# Monitoring Utilities
# =============================================================================

def get_running_tasks() -> List[Dict]:
    """Get list of running tasks."""
    try:
        tasks = list_tasks(filters=[("state", "=", "RUNNING")])
        return [
            {
                "task_id": t.task_id,
                "name": t.name,
                "state": t.state,
                "node_id": t.node_id,
            }
            for t in tasks
        ]
    except Exception:
        return []


def get_running_actors() -> List[Dict]:
    """Get list of running actors."""
    try:
        actors = list_actors(filters=[("state", "=", "ALIVE")])
        return [
            {
                "actor_id": a.actor_id,
                "class_name": a.class_name,
                "state": a.state,
                "node_id": a.node_id,
            }
            for a in actors
        ]
    except Exception:
        return []


class ResourceMonitor:
    """Monitor cluster resources over time."""
    
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self._history = []
        self._running = False
    
    def start(self):
        """Start monitoring."""
        import threading
        
        self._running = True
        
        def monitor_loop():
            cluster = RayCluster()
            while self._running:
                resources = cluster.get_resources()
                self._history.append({
                    "timestamp": time.time(),
                    "cpus_used": resources.total_cpus - resources.available_cpus,
                    "gpus_used": resources.total_gpus - resources.available_gpus,
                    "memory_used": resources.total_memory - resources.available_memory,
                })
                time.sleep(self.interval)
        
        self._thread = threading.Thread(target=monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self._running = False
    
    def get_history(self) -> List[Dict]:
        """Get resource history."""
        return self._history
    
    def get_summary(self) -> Dict[str, float]:
        """Get resource usage summary."""
        if not self._history:
            return {}
        
        cpu_usage = [h["cpus_used"] for h in self._history]
        gpu_usage = [h["gpus_used"] for h in self._history]
        
        return {
            "avg_cpu_usage": sum(cpu_usage) / len(cpu_usage),
            "max_cpu_usage": max(cpu_usage),
            "avg_gpu_usage": sum(gpu_usage) / len(gpu_usage),
            "max_gpu_usage": max(gpu_usage),
            "num_samples": len(self._history),
        }
