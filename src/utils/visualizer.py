"""
Visualization tools for profiling results.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class ProfileVisualizer:
    """
    Visualize profiling results.
    
    Example:
        >>> viz = ProfileVisualizer()
        >>> viz.plot_kernel_breakdown(profile_result)
        >>> viz.save("kernel_breakdown.png")
    """
    
    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")
        
        self._fig: Optional[plt.Figure] = None
        self._axes: Optional[plt.Axes] = None
    
    def plot_kernel_breakdown(
        self,
        kernel_times: Dict[str, float],
        title: str = "CUDA Kernel Time Breakdown",
        top_n: int = 10,
    ) -> plt.Figure:
        """
        Plot breakdown of CUDA kernel times.
        
        Args:
            kernel_times: Dict mapping kernel name to time in ms
            title: Plot title
            top_n: Number of top kernels to show
            
        Returns:
            Matplotlib figure
        """
        # Sort and get top N
        sorted_kernels = sorted(kernel_times.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        names = [k[:30] + "..." if len(k) > 30 else k for k, v in sorted_kernels]
        times = [v for k, v in sorted_kernels]
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, times, color='steelblue')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel('Time (ms)')
        ax.set_title(title)
        
        # Add value labels
        for bar, time in zip(bars, times):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{time:.2f}ms', va='center', fontsize=9)
        
        plt.tight_layout()
        self._fig = fig
        self._axes = ax
        
        return fig
    
    def plot_memory_timeline(
        self,
        timestamps: List[float],
        memory_values: List[float],
        title: str = "GPU Memory Usage Over Time",
    ) -> plt.Figure:
        """
        Plot memory usage over time.
        
        Args:
            timestamps: List of timestamps (seconds)
            memory_values: List of memory values (MB)
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ax.fill_between(timestamps, memory_values, alpha=0.3, color='steelblue')
        ax.plot(timestamps, memory_values, color='steelblue', linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Memory (MB)')
        ax.set_title(title)
        
        # Add peak annotation
        peak_idx = np.argmax(memory_values)
        peak_memory = memory_values[peak_idx]
        peak_time = timestamps[peak_idx]
        
        ax.annotate(
            f'Peak: {peak_memory:.1f} MB',
            xy=(peak_time, peak_memory),
            xytext=(peak_time + 0.5, peak_memory + 100),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10,
            color='red',
        )
        
        plt.tight_layout()
        self._fig = fig
        self._axes = ax
        
        return fig
    
    def plot_bandwidth_comparison(
        self,
        operations: List[str],
        bandwidths: List[float],
        title: str = "NCCL Bandwidth Comparison",
    ) -> plt.Figure:
        """
        Plot bandwidth comparison across operations.
        
        Args:
            operations: Operation names
            bandwidths: Bandwidth values (Gbps)
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(operations))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(operations)))
        
        bars = ax.bar(x_pos, bandwidths, color=colors)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(operations, rotation=45, ha='right')
        ax.set_ylabel('Bandwidth (Gbps)')
        ax.set_title(title)
        
        # Add value labels
        for bar, bw in zip(bars, bandwidths):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{bw:.1f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        self._fig = fig
        self._axes = ax
        
        return fig
    
    def plot_dataloader_comparison(
        self,
        configs: List[str],
        throughputs: List[float],
        title: str = "DataLoader Throughput Comparison",
    ) -> plt.Figure:
        """
        Compare DataLoader configurations.
        
        Args:
            configs: Configuration descriptions
            throughputs: Throughput values (samples/s)
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(configs))
        colors = ['#2ecc71' if i == np.argmax(throughputs) else '#3498db' 
                  for i in range(len(configs))]
        
        bars = ax.bar(x_pos, throughputs, color=colors)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.set_ylabel('Throughput (samples/s)')
        ax.set_title(title)
        
        # Add value labels
        for bar, tp in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{tp:.0f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        self._fig = fig
        self._axes = ax
        
        return fig
    
    def save(self, path: str, dpi: int = 150) -> None:
        """
        Save current figure.
        
        Args:
            path: Output path
            dpi: Resolution
        """
        if self._fig is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._fig.savefig(path, dpi=dpi, bbox_inches='tight')
    
    def show(self) -> None:
        """Display current figure."""
        if self._fig is not None:
            plt.show()


def plot_timeline(
    events: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Execution Timeline",
) -> plt.Figure:
    """
    Plot execution timeline from events.
    
    Args:
        events: List of event dictionaries with 'name', 'start', 'duration'
        output_path: Optional path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Group events by category
    categories = {}
    for event in events:
        cat = event.get('category', 'default')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(event)
    
    # Plot each category
    y_position = 0
    category_colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
    
    for (cat, cat_events), color in zip(categories.items(), category_colors):
        for event in cat_events:
            start = event.get('start', 0)
            duration = event.get('duration', 1)
            name = event.get('name', 'unknown')
            
            ax.barh(y_position, duration, left=start, height=0.8,
                    label=cat if event == cat_events[0] else "", color=color)
            
            # Add label if space permits
            if duration > 0.5:
                ax.text(start + duration/2, y_position, name[:20],
                        ha='center', va='center', fontsize=8)
            
            y_position += 1
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Events')
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig
