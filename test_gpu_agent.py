#!/usr/bin/env python3
"""
Simple test for GPU-accelerated graph-based TD(lambda) agent.
"""

import sys
import os

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    import torch
    import numpy as np
    from agents.td_lambda_graph_gpu import TDLambdaGraphAgentGPU
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic functionality of GPU agent."""
    print("\nTesting GPU-accelerated graph-based TD(lambda) agent...")

    # Test agent creation
    try:
        agent = TDLambdaGraphAgentGPU(
            n_actions=4,
            gamma=0.99,
            alpha=0.001,
            epsilon=0.1,
            lambda_value=0.9,
            hidden_dim=64,
            num_layers=2,
            dropout=0.0,
            seed=42,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"✓ Agent created successfully on device: {agent.device}")
        print(f"✓ Model parameters: {sum(p.numel() for p in agent.model.parameters())}")
    except Exception as e:
        print(f"✗ Agent creation failed: {e}")
        return False

    # Test graph building with a simple maze
    try:
        # Create a simple 3x3 maze observation
        obs = np.zeros((3, 3, 3), dtype=np.float32)
        obs[:, :, 0] = 1.0  # walls everywhere
        obs[1, 1, 0] = 0.0  # clear center
        obs[1, 1, 2] = 1.0  # agent at center
        obs[2, 2, 1] = 1.0  # goal at bottom-right

        node_features, adjacency, node_positions, agent_index, goal_index, index_map = agent._build_graph(obs)
        print(f"✓ Graph built successfully: {node_features.shape[0]} nodes")
        print(f"✓ Agent index: {agent_index}, Goal index: {goal_index}")
    except Exception as e:
        print(f"✗ Graph building failed: {e}")
        return False

    # Test Q-value computation
    try:
        q_values = agent._compute_q_values(node_features, adjacency, node_positions, agent_index, index_map)
        print(f"✓ Q-values computed: {q_values}")
    except Exception as e:
        print(f"✗ Q-value computation failed: {e}")
        return False

    print("\n✓ All basic functionality tests passed!")
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n🎉 GPU agent test completed successfully!")
    else:
        print("\n❌ GPU agent test failed!")
        sys.exit(1)