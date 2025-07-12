# DPSN_SSClustering
=====================================================

Project Overview
----------------
This is an implementation of a density peak-based clustering algorithm that combines nearest neighbor search, graph construction, and iterative optimization techniques. The algorithm can automatically discover clustering structures in data without requiring a pre-specified number of clusters.

Main Features
-------------

Core Algorithm Components:

1. Data Preprocessing (data_processing)
   - Support for data normalization (MinMaxScaler)
   - Automatic duplicate removal
   - Data shuffling option
   - Automatic handling of non-numeric data

2. Graph Structure Construction
   - Node class: Represents nodes in the graph with data, labels, adjacency relationships
   - Graph class: Manages node collections
   - Support for dynamic node and edge addition

3. Density Peak Detection (findDensityPeak)
   - Identifies cluster centers based on local density calculation
   - Uses cutoff distance to control density calculation range
   - Automatically constructs star-shaped cluster structures

4. Nearest Neighbor Search (findNNs, kdTree)
   - Efficient k-nearest neighbor search using sklearn's NearestNeighbors
   - Support for returning nearest and second-nearest neighbor information

5. Graph Construction Algorithm (construction)
   - Selects starting nodes based on degree
   - Constructs connected components
   - Automatically controls cluster count

Main Classes and Data Structures
---------------------------------

Node Class:
- id: Unique node identifier
- data: Node data vector
- label: True label
- adjacentNode: Adjacent node dictionary
- degree: Node degree
- iteration: Iteration number
- parent: Parent node reference
- children: Number of child nodes

ClusterResult Class:
Stores complete clustering result information:
- Dataset name
- Number of iterations
- Execution time
- Evaluation metrics (RI, ARI, NMI)

ClusterStoreWithHeap Class:
Manages multiple clustering results using heap structure, supports sorting by performance.

Algorithm Workflow
------------------
1. Data Preprocessing: Load data, perform normalization and cleaning
2. Nearest Neighbor Calculation: Find k nearest neighbors for each data point
3. Graph Construction: Build graph structure based on nearest neighbor relationships
4. Density Peak Detection: Find density peaks in each connected component as cluster centers
5. Structure Reconstruction: Optimize clustering structure, merge small clusters
6. Result Evaluation: Calculate clustering quality metrics

Evaluation Metrics
------------------
- RI (Rand Index): Rand Index
- ARI (Adjusted Rand Index): Adjusted Rand Index
- NMI (Normalized Mutual Information): Normalized Mutual Information

Usage Example
-------------
# Data preprocessing
data, labels, true_k = data_processing(
    filePath="your_data.csv",
    minMaxScaler=True,
    drop_duplicates=True,
    shuffle=False
)

# Create nodes
nodeList = []
for i, (point, label) in enumerate(zip(data, labels)):
    node = Node(i, point, label, str(label))
    nodeList.append(node)

# Execute clustering algorithm
# ... (specific clustering workflow)

Dependencies
------------
numpy
pandas
scikit-learn
memory-profiler
heapq (built-in)
functools (built-in)
enum (built-in)

Install Dependencies
--------------------
pip install numpy pandas scikit-learn memory-profiler

Features
--------
✓ Automatic cluster number determination
✓ Handle arbitrary cluster shapes
✓ Memory usage monitoring
✓ Multiple evaluation metrics
✓ Result storage and comparison
✓ Support for large-scale datasets

Notes
-----
1. Algorithm performance largely depends on k-nearest neighbor parameter selection
2. Cutoff distance parameter for density peak detection needs adjustment based on data characteristics
3. For high-dimensional data, dimensionality reduction is recommended
4. Algorithm time complexity is mainly determined by nearest neighbor search

Output Format
-------------
The algorithm outputs detailed clustering process information including:
- Clustering results for each iteration
- Execution time
- Memory usage
- Clustering quality evaluation

Results can be saved in CSV format for subsequent analysis and visualization.

File Structure
--------------
The main file ABC_3.27.py contains:
- Data preprocessing functions
- Node and Graph classes
- Clustering algorithm implementations
- Evaluation and result storage utilities
- Memory monitoring capabilities

Performance Considerations
--------------------------
- The algorithm scales well with data size due to efficient nearest neighbor search
- Memory usage is monitored throughout execution
- Results are stored using heap structures for efficient comparison
- Supports iterative refinement of clustering results
