#!/usr/bin/env python3
"""
Debug script to identify the exact tensor shape mismatch issue.
"""

import sys
import os
sys.path.insert(0, '/home/felix/personal/zsa0/src')

import zsa0_rust

def debug_tensor_shapes():
    print("=== DEBUGGING TENSOR SHAPE ISSUE ===")
    
    # Check available constants
    print("\n1. Checking available constants...")
    print(f"   BUF_N_CHANNELS: {zsa0_rust.BUF_N_CHANNELS}")
    print(f"   N_COLS: {zsa0_rust.N_COLS}")
    print(f"   N_ROWS: {zsa0_rust.N_ROWS}")
    
    # Test with a simple game to see what Sample gets created
    print("\n2. Testing with minimal self-play...")
    try:
        # Run a minimal self-play game to generate a sample
        metadata = zsa0_rust.GameMetadata(game_id=0, player0_id=0, player1_id=0)
        print("   Creating game metadata...")
        print(f"   Game metadata: {metadata}")
        print(f"   Game metadata: {metadata.game_id}")
        
        # pub struct GameMetadata {
        # #[pyo3(get)]
        # pub game_id: u64,

        # #[pyo3(get)]
        # pub player0_id: ModelID,

        # #[pyo3(get)]
        # pub player1_id: ModelID,
        # This should generate samples during self-play
        # result = zsa0_rust.play_games(
        #     reqs,
        #     self_play_batch_size,
        #     n_mcts_iterations,
        #     c_exploration,
        #     c_ply_penalty,
        #     lambda player_id, pos: model.forward_numpy(pos),  # type: ignore
        # )

        # result = zsa0_rust.play_games(
        #     reqs=[metadata],
        #     max_nn_batch_size=1,
        #     n_mcts_iterations=2,  # Very minimal
        #     py_eval_pos_cb=None,
        #     c_exploration=1.
        #     c_ply_penalty=0.01
        # )
        
        # model = parent.get_model(base_dir)
        from zsa0.nn import BottinaNet, ModelConfig
        
        print("   Creating model...")
        model_config = ModelConfig(
            n_residual_blocks=1,
            conv_filter_size=32,
            n_policy_layers=4,
            n_value_layers=2,
            lr_schedule={
                0: 0.002,
                10: 0.0008
            },
            l2_reg=0.0004,
        )
        
        print("   Initializing model...")
        model = BottinaNet(model_config)
        print("   Initializing model2...")
        model.eval()
        print("   Initializing model3...")
        model.to('cpu')
        # model.to('cuda')
        
        print("   Running self-play...")
        result = zsa0_rust.play_games(
            [metadata],
            3,  # Batch size
            2000, # MCTS iterations
            1.0, # Exploration constant
            0.01, # Ply penalty
            lambda player_id, pos: model.forward_numpy(pos),  # type: ignore
        )
        
        print(f"   Generated {len(result.games)} games")
        
        if result.games:
            game = result.games[0]
            print(f"   Game has {len(game.samples)} samples")
            
            if game.samples:
                sample = game.samples[0]
                print(f"   First sample policy: {sample.policy}")
                print(f"   First sample q_penalty: {sample.q_penalty}")
                print(f"   First sample q_no_penalty: {sample.q_no_penalty}")
                
                # This is where the crash happens - let's see what the position looks like
                print(f"   Attempting to convert sample to numpy...")
                
                try:
                    pos_tensor, policy_tensor, q_penalty_tensor, q_no_penalty_tensor = sample.to_numpy()
                    print(f"   ✅ Sample to_numpy SUCCESS!")
                    print(f"   Position tensor shape: {pos_tensor.shape}")
                    print(f"   Policy tensor shape: {policy_tensor.shape}")
                except Exception as e:
                    print(f"   ❌ Sample to_numpy FAILED: {e}")
                    
                    # Try to get more info about the position
                    try:
                        pos_str = sample.pos_str()
                        print(f"   Position string preview: {pos_str[:200]}...")
                    except:
                        print(f"   Could not get position string")
            else:
                print(f"   No samples generated")
        else:
            print(f"   No games generated")
            
    except Exception as e:
        print(f"   ❌ Self-play test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_tensor_shapes()