"""
Trace Analyzer - Analyze Chrome trace files from PyTorch Profiler.

This module provides tools for parsing and analyzing trace files to extract
insights about GPU performance.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TraceEvent:
    """Represents a single event in the trace."""
    
    name: str
    category: str
    phase: str  # 'B' = begin, 'E' = end, 'X' = complete
    timestamp_us: float
    duration_us: float
    thread_id: int
    process_id: int
    args: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration_us / 1000
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceEvent":
        """Create TraceEvent from trace dictionary."""
        return cls(
            name=data.get("name", "unknown"),
            category=data.get("cat", "unknown"),
            phase=data.get("ph", "X"),
            timestamp_us=data.get("ts", 0),
            duration_us=data.get("dur", 0),
            thread_id=data.get("tid", 0),
            process_id=data.get("pid", 0),
            args=data.get("args", {}),
        )


@dataclass
class KernelStats:
    """Statistics for a CUDA kernel."""
    
    name: str
    count: int
    total_time_us: float
    min_time_us: float
    max_time_us: float
    avg_time_us: float
    
    @property
    def total_time_ms(self) -> float:
        return self.total_time_us / 1000
    
    @property
    def avg_time_ms(self) -> float:
        return self.avg_time_us / 1000


@dataclass
class TraceAnalysis:
    """Results of trace analysis."""
    
    # Overview
    total_duration_ms: float
    cuda_time_ms: float
    cpu_time_ms: float
    
    # GPU utilization
    gpu_utilization: float  # 0-1
    
    # Top kernels
    top_cuda_kernels: List[KernelStats]
    top_cpu_ops: List[KernelStats]
    
    # Memory operations
    memory_ops: Dict[str, Any]
    
    # Communication
    nccl_ops: List[Dict[str, Any]]
    
    def summary(self) -> str:
        """Generate analysis summary."""
        lines = [
            "=" * 70,
            "TRACE ANALYSIS SUMMARY",
            "=" * 70,
            "",
            f"Total Duration: {self.total_duration_ms:.2f} ms",
            f"CUDA Time: {self.cuda_time_ms:.2f} ms",
            f"CPU Time: {self.cpu_time_ms:.2f} ms",
            f"GPU Utilization: {self.gpu_utilization * 100:.1f}%",
            "",
            "Top CUDA Kernels:",
        ]
        
        for i, kernel in enumerate(self.top_cuda_kernels[:10], 1):
            pct = (kernel.total_time_us / (self.cuda_time_ms * 1000) * 100) if self.cuda_time_ms > 0 else 0
            lines.append(
                f"  {i:2}. {kernel.name[:50]:50} "
                f"{kernel.total_time_ms:8.3f} ms ({pct:5.1f}%) "
                f"[{kernel.count:5} calls]"
            )
        
        if self.nccl_ops:
            lines.extend(["", "NCCL Operations:"])
            for op in self.nccl_ops:
                lines.append(f"  - {op['name']}: {op['time_ms']:.3f} ms")
        
        lines.append("=" * 70)
        return "\n".join(lines)


class TraceAnalyzer:
    """
    Analyze Chrome trace files from PyTorch Profiler.
    
    Example:
        >>> analyzer = TraceAnalyzer()
        >>> analysis = analyzer.analyze("trace.json")
        >>> print(analysis.summary())
    """
    
    def __init__(self):
        """Initialize trace analyzer."""
        self._events: List[TraceEvent] = []
        self._metadata: Dict[str, Any] = {}
    
    def load(self, trace_path: str) -> None:
        """
        Load trace file.
        
        Args:
            trace_path: Path to Chrome trace JSON file
        """
        path = Path(trace_path)
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_path}")
        
        with open(path, "r") as f:
            data = json.load(f)
        
        # Handle both array and object formats
        if isinstance(data, list):
            trace_events = data
        else:
            trace_events = data.get("traceEvents", [])
            self._metadata = {k: v for k, v in data.items() if k != "traceEvents"}
        
        # Parse events
        self._events = []
        for event_data in trace_events:
            if isinstance(event_data, dict) and "name" in event_data:
                self._events.append(TraceEvent.from_dict(event_data))
    
    def analyze(self, trace_path: Optional[str] = None) -> TraceAnalysis:
        """
        Analyze trace and return results.
        
        Args:
            trace_path: Optional path to load before analyzing
            
        Returns:
            TraceAnalysis with detailed results
        """
        if trace_path:
            self.load(trace_path)
        
        if not self._events:
            raise ValueError("No trace data loaded")
        
        # Calculate durations
        cuda_events = [e for e in self._events if self._is_cuda_event(e)]
        cpu_events = [e for e in self._events if self._is_cpu_event(e)]
        
        cuda_time_us = sum(e.duration_us for e in cuda_events)
        cpu_time_us = sum(e.duration_us for e in cpu_events)
        
        # Get timeline bounds
        all_events = [e for e in self._events if e.duration_us > 0]
        if all_events:
            start_time = min(e.timestamp_us for e in all_events)
            end_time = max(e.timestamp_us + e.duration_us for e in all_events)
            total_duration = (end_time - start_time) / 1000  # ms
        else:
            total_duration = 0
        
        # Calculate GPU utilization
        gpu_utilization = (cuda_time_us / 1000 / total_duration) if total_duration > 0 else 0
        gpu_utilization = min(gpu_utilization, 1.0)  # Cap at 100%
        
        # Aggregate kernel stats
        cuda_kernel_stats = self._aggregate_kernel_stats(cuda_events)
        cpu_op_stats = self._aggregate_kernel_stats(cpu_events)
        
        # Find NCCL operations
        nccl_ops = self._find_nccl_ops()
        
        # Memory operations
        memory_ops = self._analyze_memory_ops()
        
        return TraceAnalysis(
            total_duration_ms=total_duration,
            cuda_time_ms=cuda_time_us / 1000,
            cpu_time_ms=cpu_time_us / 1000,
            gpu_utilization=gpu_utilization,
            top_cuda_kernels=cuda_kernel_stats,
            top_cpu_ops=cpu_op_stats,
            memory_ops=memory_ops,
            nccl_ops=nccl_ops,
        )
    
    def _is_cuda_event(self, event: TraceEvent) -> bool:
        """Check if event is a CUDA event."""
        cuda_indicators = ["cuda", "kernel", "gpu", "nccl"]
        return any(ind in event.category.lower() for ind in cuda_indicators) or \
               any(ind in event.name.lower() for ind in cuda_indicators)
    
    def _is_cpu_event(self, event: TraceEvent) -> bool:
        """Check if event is a CPU event."""
        return "cpu" in event.category.lower() or \
               event.category in ["cpu_op", "Operator"]
    
    def _aggregate_kernel_stats(
        self,
        events: List[TraceEvent]
    ) -> List[KernelStats]:
        """Aggregate events into kernel statistics."""
        kernel_times: Dict[str, List[float]] = defaultdict(list)
        
        for event in events:
            if event.duration_us > 0:
                kernel_times[event.name].append(event.duration_us)
        
        stats = []
        for name, times in kernel_times.items():
            stats.append(KernelStats(
                name=name,
                count=len(times),
                total_time_us=sum(times),
                min_time_us=min(times),
                max_time_us=max(times),
                avg_time_us=sum(times) / len(times),
            ))
        
        # Sort by total time
        stats.sort(key=lambda x: x.total_time_us, reverse=True)
        return stats
    
    def _find_nccl_ops(self) -> List[Dict[str, Any]]:
        """Find NCCL communication operations."""
        nccl_events = [
            e for e in self._events
            if "nccl" in e.name.lower() or "nccl" in e.category.lower()
        ]
        
        nccl_ops = []
        for event in nccl_events:
            nccl_ops.append({
                "name": event.name,
                "time_ms": event.duration_ms,
                "args": event.args,
            })
        
        return nccl_ops
    
    def _analyze_memory_ops(self) -> Dict[str, Any]:
        """Analyze memory operations."""
        memory_events = [
            e for e in self._events
            if "memory" in e.name.lower() or "alloc" in e.name.lower()
        ]
        
        total_alloc_time = sum(e.duration_us for e in memory_events)
        
        return {
            "count": len(memory_events),
            "total_time_ms": total_alloc_time / 1000,
        }
    
    def find_bottlenecks(
        self,
        threshold_pct: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Find performance bottlenecks in the trace.
        
        Args:
            threshold_pct: Minimum percentage of total time to consider
            
        Returns:
            List of bottleneck descriptions
        """
        analysis = self.analyze()
        bottlenecks = []
        
        # Check for low GPU utilization
        if analysis.gpu_utilization < 0.7:
            bottlenecks.append({
                "type": "low_gpu_utilization",
                "severity": "high" if analysis.gpu_utilization < 0.5 else "medium",
                "description": f"GPU utilization is {analysis.gpu_utilization * 100:.1f}%",
                "recommendation": "Consider increasing batch size or overlapping data loading with computation",
            })
        
        # Check for expensive kernels
        total_time = analysis.cuda_time_ms
        for kernel in analysis.top_cuda_kernels[:5]:
            pct = (kernel.total_time_ms / total_time * 100) if total_time > 0 else 0
            if pct > threshold_pct:
                bottlenecks.append({
                    "type": "expensive_kernel",
                    "severity": "high" if pct > 20 else "medium",
                    "description": f"Kernel '{kernel.name}' takes {pct:.1f}% of CUDA time",
                    "recommendation": "Consider optimizing this kernel or using a more efficient implementation",
                })
        
        # Check for NCCL overhead
        if analysis.nccl_ops:
            nccl_time = sum(op["time_ms"] for op in analysis.nccl_ops)
            if nccl_time > total_time * 0.1:
                bottlenecks.append({
                    "type": "high_nccl_overhead",
                    "severity": "medium",
                    "description": f"NCCL operations take {nccl_time:.2f} ms ({nccl_time / total_time * 100:.1f}%)",
                    "recommendation": "Consider gradient accumulation or overlapping communication with computation",
                })
        
        return bottlenecks
    
    def compare_traces(
        self,
        other_path: str
    ) -> Dict[str, Any]:
        """
        Compare current trace with another trace.
        
        Args:
            other_path: Path to trace file to compare with
            
        Returns:
            Dictionary with comparison results
        """
        other_analyzer = TraceAnalyzer()
        other_analysis = other_analyzer.analyze(other_path)
        current_analysis = self.analyze()
        
        return {
            "duration_change_pct": (
                (current_analysis.total_duration_ms - other_analysis.total_duration_ms)
                / other_analysis.total_duration_ms * 100
            ) if other_analysis.total_duration_ms > 0 else 0,
            "gpu_utilization_change": current_analysis.gpu_utilization - other_analysis.gpu_utilization,
            "cuda_time_change_pct": (
                (current_analysis.cuda_time_ms - other_analysis.cuda_time_ms)
                / other_analysis.cuda_time_ms * 100
            ) if other_analysis.cuda_time_ms > 0 else 0,
        }
