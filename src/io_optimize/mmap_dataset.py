"""
Memory-Mapped Datasets - Efficient handling of large datasets.

This module provides dataset implementations that use memory mapping
to handle datasets too large to fit in RAM.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class MMapDataset(Dataset):
    """
    Memory-mapped numpy array dataset.
    
    Uses numpy memmap to efficiently access large arrays stored on disk
    without loading the entire array into memory.
    
    Example:
        >>> # Create dataset from numpy files
        >>> dataset = MMapDataset(
        ...     data_path="data.npy",
        ...     labels_path="labels.npy"
        ... )
        >>> 
        >>> # Use with DataLoader
        >>> loader = DataLoader(dataset, batch_size=32)
    """
    
    def __init__(
        self,
        data_path: str,
        labels_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        dtype: np.dtype = np.float32,
        mode: str = 'r',
    ):
        """
        Initialize memory-mapped dataset.
        
        Args:
            data_path: Path to numpy array file (.npy)
            labels_path: Optional path to labels file
            transform: Optional data transform
            target_transform: Optional label transform
            dtype: Data type for memmap
            mode: Memmap mode ('r', 'r+', 'w+', 'c')
        """
        self.data_path = Path(data_path)
        self.labels_path = Path(labels_path) if labels_path else None
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        
        # Load data shape without loading data
        if self.data_path.suffix == '.npy':
            # Load as memmap for .npy files
            self.data = np.load(str(self.data_path), mmap_mode=mode)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        # Load labels
        self.labels = None
        if self.labels_path is not None:
            if self.labels_path.suffix == '.npy':
                self.labels = np.load(str(self.labels_path), mmap_mode=mode)
            else:
                # Try loading as regular array for small label files
                self.labels = np.load(str(self.labels_path))
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Get item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Data tensor or (data, label) tuple
        """
        # Load data
        data = self.data[idx]
        
        # Convert to tensor
        data = torch.from_numpy(data.copy())  # copy() for memmap
        
        # Apply transform
        if self.transform is not None:
            data = self.transform(data)
        
        # Return with or without labels
        if self.labels is not None:
            label = self.labels[idx]
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label.copy())
            else:
                label = torch.tensor(label)
            
            if self.target_transform is not None:
                label = self.target_transform(label)
            
            return data, label
        
        return data
    
    @classmethod
    def create_from_arrays(
        cls,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        output_dir: str = "./data",
        name: str = "dataset",
    ) -> "MMapDataset":
        """
        Create MMapDataset from numpy arrays.
        
        Args:
            data: Data array
            labels: Optional labels array
            output_dir: Directory to save files
            name: Base name for files
            
        Returns:
            MMapDataset instance
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        data_path = output_dir / f"{name}_data.npy"
        np.save(str(data_path), data)
        
        labels_path = None
        if labels is not None:
            labels_path = output_dir / f"{name}_labels.npy"
            np.save(str(labels_path), labels)
        
        return cls(
            data_path=str(data_path),
            labels_path=str(labels_path) if labels_path else None,
        )


class HDF5Dataset(Dataset):
    """
    HDF5-based dataset for handling very large datasets.
    
    HDF5 provides efficient storage and retrieval of large arrays
    with built-in compression support.
    
    Example:
        >>> dataset = HDF5Dataset(
        ...     filepath="data.h5",
        ...     data_key="images",
        ...     labels_key="labels"
        ... )
    """
    
    def __init__(
        self,
        filepath: str,
        data_key: str = "data",
        labels_key: Optional[str] = "labels",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache_size: int = 100,
    ):
        """
        Initialize HDF5 dataset.
        
        Args:
            filepath: Path to HDF5 file
            data_key: Key for data dataset in HDF5
            labels_key: Optional key for labels
            transform: Optional data transform
            target_transform: Optional label transform
            cache_size: LRU cache size for samples
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5Dataset. Install with: pip install h5py")
        
        self.filepath = filepath
        self.data_key = data_key
        self.labels_key = labels_key
        self.transform = transform
        self.target_transform = target_transform
        
        # Open file and get dataset info
        self._file: Optional[Any] = None
        self._data: Optional[Any] = None
        self._labels: Optional[Any] = None
        
        # Initialize file handle
        self._open_file()
        
        # Get dataset length
        self._length = len(self._data)
        
        # Simple LRU cache
        self._cache: Dict[int, Any] = {}
        self._cache_order: List[int] = []
        self._cache_size = cache_size
    
    def _open_file(self) -> None:
        """Open HDF5 file."""
        import h5py
        
        if self._file is None:
            self._file = h5py.File(self.filepath, 'r')
            self._data = self._file[self.data_key]
            
            if self.labels_key and self.labels_key in self._file:
                self._labels = self._file[self.labels_key]
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self._length
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Get item by index."""
        # Check cache
        if idx in self._cache:
            return self._cache[idx]
        
        # Ensure file is open
        self._open_file()
        
        # Load data
        data = np.array(self._data[idx])
        data = torch.from_numpy(data)
        
        if self.transform is not None:
            data = self.transform(data)
        
        # Load labels
        result: Any
        if self._labels is not None:
            label = self._labels[idx]
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            else:
                label = torch.tensor(label)
            
            if self.target_transform is not None:
                label = self.target_transform(label)
            
            result = (data, label)
        else:
            result = data
        
        # Update cache
        self._update_cache(idx, result)
        
        return result
    
    def _update_cache(self, idx: int, value: Any) -> None:
        """Update LRU cache."""
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        
        self._cache[idx] = value
        self._cache_order.append(idx)
    
    def __del__(self):
        """Clean up file handle."""
        if self._file is not None:
            self._file.close()
    
    @classmethod
    def create_from_arrays(
        cls,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        filepath: str = "dataset.h5",
        data_key: str = "data",
        labels_key: str = "labels",
        compression: str = "gzip",
        compression_opts: int = 4,
    ) -> "HDF5Dataset":
        """
        Create HDF5Dataset from numpy arrays.
        
        Args:
            data: Data array
            labels: Optional labels array
            filepath: Output HDF5 file path
            data_key: Key for data dataset
            labels_key: Key for labels dataset
            compression: Compression algorithm
            compression_opts: Compression level
            
        Returns:
            HDF5Dataset instance
        """
        import h5py
        
        with h5py.File(filepath, 'w') as f:
            f.create_dataset(
                data_key,
                data=data,
                compression=compression,
                compression_opts=compression_opts,
                chunks=True,
            )
            
            if labels is not None:
                f.create_dataset(
                    labels_key,
                    data=labels,
                    compression=compression,
                    compression_opts=compression_opts,
                )
        
        return cls(filepath=filepath, data_key=data_key, labels_key=labels_key)


class ChunkedDataset(Dataset):
    """
    Dataset that loads data in chunks for efficient streaming.
    
    Useful when data is too large to load entirely and is stored
    in multiple files.
    """
    
    def __init__(
        self,
        chunk_dir: str,
        chunk_pattern: str = "chunk_*.npy",
        transform: Optional[Callable] = None,
        preload_chunks: int = 2,
    ):
        """
        Initialize chunked dataset.
        
        Args:
            chunk_dir: Directory containing chunk files
            chunk_pattern: Glob pattern for chunk files
            transform: Optional data transform
            preload_chunks: Number of chunks to preload
        """
        self.chunk_dir = Path(chunk_dir)
        self.transform = transform
        self.preload_chunks = preload_chunks
        
        # Find all chunk files
        self.chunk_files = sorted(self.chunk_dir.glob(chunk_pattern))
        if not self.chunk_files:
            raise ValueError(f"No chunk files found matching {chunk_pattern}")
        
        # Get chunk sizes
        self.chunk_sizes: List[int] = []
        self.cumulative_sizes: List[int] = []
        
        cumsum = 0
        for chunk_file in self.chunk_files:
            # Load just the header to get shape
            data = np.load(str(chunk_file), mmap_mode='r')
            size = len(data)
            self.chunk_sizes.append(size)
            cumsum += size
            self.cumulative_sizes.append(cumsum)
        
        self._total_size = cumsum
        
        # Preloaded chunks cache
        self._chunk_cache: Dict[int, np.ndarray] = {}
    
    def __len__(self) -> int:
        """Return total dataset size."""
        return self._total_size
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get item by global index."""
        # Find which chunk contains this index
        chunk_idx = self._find_chunk(idx)
        
        # Calculate local index within chunk
        local_idx = idx
        if chunk_idx > 0:
            local_idx = idx - self.cumulative_sizes[chunk_idx - 1]
        
        # Get chunk data
        chunk_data = self._get_chunk(chunk_idx)
        
        # Get item
        data = torch.from_numpy(chunk_data[local_idx].copy())
        
        if self.transform is not None:
            data = self.transform(data)
        
        return data
    
    def _find_chunk(self, idx: int) -> int:
        """Find chunk index for global index."""
        for i, cumsize in enumerate(self.cumulative_sizes):
            if idx < cumsize:
                return i
        raise IndexError(f"Index {idx} out of range")
    
    def _get_chunk(self, chunk_idx: int) -> np.ndarray:
        """Get chunk data, loading if necessary."""
        if chunk_idx in self._chunk_cache:
            return self._chunk_cache[chunk_idx]
        
        # Load chunk
        chunk_data = np.load(str(self.chunk_files[chunk_idx]))
        
        # Manage cache size
        if len(self._chunk_cache) >= self.preload_chunks:
            # Remove oldest chunk
            oldest_key = next(iter(self._chunk_cache))
            del self._chunk_cache[oldest_key]
        
        self._chunk_cache[chunk_idx] = chunk_data
        return chunk_data
