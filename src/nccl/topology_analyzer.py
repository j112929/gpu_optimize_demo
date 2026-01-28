"""
Topology Analyzer - Analyze GPU topology for optimal communication.

This module provides tools for analyzing GPU interconnect topology
(NVLink, PCIe, etc.) to optimize communication patterns.
"""

import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    
    index: int
    name: str
    uuid: str
    compute_capability: Tuple[int, int]
    memory_total_mb: int
    memory_free_mb: int
    
    # Interconnect info
    pcie_gen: int = 0
    pcie_width: int = 0
    nvlink_count: int = 0


@dataclass
class GPULink:
    """Link between two GPUs."""
    
    gpu_0: int
    gpu_1: int
    link_type: str  # 'NVLink', 'PCIe', 'SYS'
    bandwidth_gbps: float
    
    # NVLink specific
    nvlink_version: int = 0
    nvlink_lanes: int = 0


@dataclass
class GPUTopology:
    """Full GPU topology information."""
    
    num_gpus: int
    gpus: List[GPUInfo]
    links: List[GPULink]
    
    # Adjacency matrix: topology[i][j] = link type
    matrix: List[List[str]] = field(default_factory=list)
    
    # P2P access matrix
    p2p_access: List[List[bool]] = field(default_factory=list)
    
    def get_link(self, gpu_0: int, gpu_1: int) -> Optional[GPULink]:
        """Get link between two GPUs."""
        for link in self.links:
            if (link.gpu_0 == gpu_0 and link.gpu_1 == gpu_1) or \
               (link.gpu_0 == gpu_1 and link.gpu_1 == gpu_0):
                return link
        return None
    
    def best_pairing(self) -> List[Tuple[int, int]]:
        """Get best GPU pairings based on topology."""
        if self.num_gpus < 2:
            return []
        
        # Sort links by bandwidth
        sorted_links = sorted(self.links, key=lambda x: x.bandwidth_gbps, reverse=True)
        
        used_gpus = set()
        pairs = []
        
        for link in sorted_links:
            if link.gpu_0 not in used_gpus and link.gpu_1 not in used_gpus:
                pairs.append((link.gpu_0, link.gpu_1))
                used_gpus.add(link.gpu_0)
                used_gpus.add(link.gpu_1)
        
        return pairs
    
    def summary(self) -> str:
        """Generate topology summary."""
        lines = [
            "=" * 70,
            "GPU TOPOLOGY",
            "=" * 70,
            f"Number of GPUs: {self.num_gpus}",
            "",
            "GPUs:",
        ]
        
        for gpu in self.gpus:
            lines.append(
                f"  [{gpu.index}] {gpu.name}"
                f"\n      Memory: {gpu.memory_total_mb}MB"
                f"\n      PCIe: Gen{gpu.pcie_gen} x{gpu.pcie_width}"
                f"\n      NVLinks: {gpu.nvlink_count}"
            )
        
        lines.append("\nTopology Matrix:")
        
        # Header
        header = "    " + " ".join(f"GPU{i}" for i in range(self.num_gpus))
        lines.append(header)
        
        # Matrix rows
        for i, row in enumerate(self.matrix):
            row_str = f"GPU{i} " + " ".join(f"{cell:>4}" for cell in row)
            lines.append(row_str)
        
        # Best NVLink connections
        nvlink_pairs = [(l.gpu_0, l.gpu_1) for l in self.links if l.link_type == 'NVLink']
        if nvlink_pairs:
            lines.append("\nNVLink Connections:")
            for g0, g1 in nvlink_pairs:
                link = self.get_link(g0, g1)
                if link:
                    lines.append(f"  GPU{g0} <-> GPU{g1}: {link.bandwidth_gbps:.1f} Gbps")
        
        lines.append("=" * 70)
        return "\n".join(lines)


class TopologyAnalyzer:
    """
    Analyze GPU topology for communication optimization.
    
    Example:
        >>> analyzer = TopologyAnalyzer()
        >>> topology = analyzer.analyze()
        >>> print(topology.summary())
    """
    
    def __init__(self, device_ids: Optional[List[int]] = None):
        """
        Initialize topology analyzer.
        
        Args:
            device_ids: List of GPU device IDs to analyze. None for all.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for TopologyAnalyzer")
        
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self._topology: Optional[GPUTopology] = None
    
    def analyze(self) -> GPUTopology:
        """
        Analyze GPU topology.
        
        Returns:
            GPUTopology with complete topology information
        """
        num_gpus = len(self.device_ids)
        
        # Gather GPU info
        gpus = []
        for idx in self.device_ids:
            gpu_info = self._get_gpu_info(idx)
            gpus.append(gpu_info)
        
        # Build link matrix
        links = []
        matrix = [['X'] * num_gpus for _ in range(num_gpus)]
        p2p_access = [[False] * num_gpus for _ in range(num_gpus)]
        
        for i, gpu_i in enumerate(self.device_ids):
            for j, gpu_j in enumerate(self.device_ids):
                if i == j:
                    matrix[i][j] = 'X'
                    p2p_access[i][j] = True
                    continue
                
                if i > j:
                    continue  # Already analyzed
                
                # Check P2P access
                can_access = torch.cuda.can_device_access_peer(gpu_i, gpu_j)
                p2p_access[i][j] = can_access
                p2p_access[j][i] = can_access
                
                # Determine link type
                link_type, bandwidth = self._get_link_info(gpu_i, gpu_j)
                
                matrix[i][j] = link_type
                matrix[j][i] = link_type
                
                links.append(GPULink(
                    gpu_0=i,
                    gpu_1=j,
                    link_type=link_type,
                    bandwidth_gbps=bandwidth,
                ))
        
        self._topology = GPUTopology(
            num_gpus=num_gpus,
            gpus=gpus,
            links=links,
            matrix=matrix,
            p2p_access=p2p_access,
        )
        
        return self._topology
    
    def _get_gpu_info(self, device_id: int) -> GPUInfo:
        """Get information about a single GPU."""
        props = torch.cuda.get_device_properties(device_id)
        
        # Get memory info
        torch.cuda.set_device(device_id)
        memory_total = torch.cuda.get_device_properties(device_id).total_memory // (1024 * 1024)
        memory_free = torch.cuda.memory_reserved(device_id) // (1024 * 1024)
        
        # Try to get UUID and other info from nvidia-smi
        uuid = f"GPU-{device_id}"
        pcie_gen = 4  # Default
        pcie_width = 16  # Default
        nvlink_count = 0
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '-i', str(device_id), '--query-gpu=gpu_uuid,pcie.link.gen.current,pcie.link.width.current', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                if len(parts) >= 1:
                    uuid = parts[0]
                if len(parts) >= 2:
                    pcie_gen = int(parts[1])
                if len(parts) >= 3:
                    pcie_width = int(parts[2])
        except Exception:
            pass
        
        # Try to get NVLink count
        try:
            result = subprocess.run(
                ['nvidia-smi', 'nvlink', '-s', '-i', str(device_id)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                nvlink_count = result.stdout.count('Link')
        except Exception:
            pass
        
        return GPUInfo(
            index=device_id,
            name=props.name,
            uuid=uuid,
            compute_capability=(props.major, props.minor),
            memory_total_mb=memory_total,
            memory_free_mb=memory_total - memory_free,
            pcie_gen=pcie_gen,
            pcie_width=pcie_width,
            nvlink_count=nvlink_count,
        )
    
    def _get_link_info(self, gpu_0: int, gpu_1: int) -> Tuple[str, float]:
        """Get link information between two GPUs."""
        # Try nvidia-smi for topology info
        try:
            result = subprocess.run(
                ['nvidia-smi', 'topo', '-m'],
                capture_output=True,
                text=True,
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    # Look for the row corresponding to gpu_0
                    if line.startswith(f'GPU{gpu_0}') or line.startswith(f'\tGPU{gpu_0}'):
                        parts = line.split()
                        if len(parts) > gpu_1 + 1:
                            link_code = parts[gpu_1 + 1]
                            
                            if 'NV' in link_code:
                                # NVLink
                                return 'NVL', 600.0  # Approximate NVLink 4 bandwidth
                            elif 'PHB' in link_code:
                                return 'PHB', 32.0  # PCIe switch
                            elif 'PIX' in link_code:
                                return 'PIX', 16.0  # Single PCIe switch
                            elif 'PXB' in link_code:
                                return 'PXB', 16.0  # Multiple PCIe switches
                            elif 'SYS' in link_code:
                                return 'SYS', 8.0  # System/QPI
        except Exception:
            pass
        
        # Fallback: check P2P access
        if torch.cuda.can_device_access_peer(gpu_0, gpu_1):
            return 'P2P', 16.0
        
        return 'SYS', 8.0
    
    @property
    def topology(self) -> Optional[GPUTopology]:
        """Get analyzed topology."""
        return self._topology
    
    def get_optimal_ring_order(self) -> List[int]:
        """
        Get optimal GPU order for ring all-reduce.
        
        Returns:
            List of GPU indices in optimal ring order
        """
        if self._topology is None:
            self.analyze()
        
        if self._topology.num_gpus <= 2:
            return list(range(self._topology.num_gpus))
        
        # Greedy algorithm: start from GPU 0, always pick the best available next GPU
        order = [0]
        remaining = set(range(1, self._topology.num_gpus))
        
        while remaining:
            current = order[-1]
            best_next = None
            best_bandwidth = -1
            
            for gpu in remaining:
                link = self._topology.get_link(current, gpu)
                bandwidth = link.bandwidth_gbps if link else 0
                
                if bandwidth > best_bandwidth:
                    best_bandwidth = bandwidth
                    best_next = gpu
            
            if best_next is not None:
                order.append(best_next)
                remaining.remove(best_next)
        
        return order


def get_gpu_topology() -> GPUTopology:
    """
    Quick function to get GPU topology.
    
    Returns:
        GPUTopology
    """
    analyzer = TopologyAnalyzer()
    return analyzer.analyze()


def print_gpu_topology() -> None:
    """Print GPU topology information."""
    topology = get_gpu_topology()
    print(topology.summary())


def get_nccl_environment_variables(topology: Optional[GPUTopology] = None) -> Dict[str, str]:
    """
    Get recommended NCCL environment variables based on topology.
    
    Args:
        topology: Optional pre-analyzed topology
        
    Returns:
        Dictionary of environment variable recommendations
    """
    if topology is None:
        topology = get_gpu_topology()
    
    env_vars = {}
    
    # Check for NVLink
    has_nvlink = any(link.link_type == 'NVL' for link in topology.links)
    
    if has_nvlink:
        # Use NVLink for P2P
        env_vars['NCCL_P2P_LEVEL'] = 'NVL'
        env_vars['NCCL_NET_GDR_LEVEL'] = '5'
    else:
        # Use PCIe
        env_vars['NCCL_P2P_LEVEL'] = 'PHB'
    
    # Tree algorithm often works well for AllReduce
    env_vars['NCCL_ALGO'] = 'Tree,Ring'
    
    # Enable debug for troubleshooting
    env_vars['NCCL_DEBUG'] = 'WARN'
    
    # Buffer sizes
    if topology.num_gpus >= 8:
        env_vars['NCCL_BUFFSIZE'] = str(8 * 1024 * 1024)  # 8MB for large clusters
    else:
        env_vars['NCCL_BUFFSIZE'] = str(4 * 1024 * 1024)  # 4MB default
    
    return env_vars
