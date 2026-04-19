#!/usr/bin/env python3
"""
Compatibility test for GPU-accelerated graph-based TD(lambda) agent.
"""

import sys
import os

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def test_imports():
    """Test if all imports work correctly."""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")

        # Test sparse tensor creation
        try:
            sparse_tensor = torch.sparse_coo_tensor(
                torch.tensor([[0, 1], [1, 0]]),
                torch.tensor([1.0, 1.0]),
                (2, 2)
            )
            print("✓ torch.sparse_coo_tensor available")
        except AttributeError:
            print("⚠ torch.sparse_coo_tensor not available, will use dense fallback")

        # Test the GPU agent import
        from agents.td_lambda_graph_gpu import TDLambdaGraphAgentGPU
        print("✓ TDLambdaGraphAgentGPU import successful")

        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality with CPU."""
    try:
        import torch
        import numpy as np
        from agents.td_lambda_graph_gpu import TDLambdaGraphAgentGPU

        # Create agent on CPU
        agent = TDLambdaGraphAgentGPU(
            n_actions=4,
            gamma=0.99,
            alpha=0.001,
            epsilon=0.1,
            lambda_value=0.9,
            hidden_dim=32,  # Smaller for testing
            num_layers=1,
            dropout=0.0,
            seed=42,
            device="cpu"
        )
        print("✓ Agent created successfully on CPU")

        # Test graph building with a simple maze
        obs = np.zeros((3, 3, 3), dtype=np.float32)
        obs[:, :, 0] = 1.0  # walls everywhere
        obs[1, 1, 0] = 0.0  # clear center
        obs[1, 1, 2] = 1.0  # agent at center
        obs[2, 2, 1] = 1.0  # goal at bottom-right

        node_features, adjacency, node_positions, agent_index, goal_index, index_map = agent._build_graph(obs)
        print(f"✓ Graph built successfully: {node_features.shape[0]} nodes")
        print(f"✓ Adjacency matrix shape: {adjacency.shape}, sparse: {adjacency.is_sparse}")

        # Test Q-value computation
        q_values = agent._compute_q_values(node_features, adjacency, node_positions, agent_index, index_map)
        print(f"✓ Q-values computed: {q_values}")

        return True
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing GPU agent compatibility...")
    print()

    import_success = test_imports()
    if not import_success:
        sys.exit(1)

    print()
    func_success = test_basic_functionality()

    if func_success:
        print("\n🎉 All compatibility tests passed!")
    else:
        print("\n❌ Compatibility tests failed!")
        sys.exit(1)