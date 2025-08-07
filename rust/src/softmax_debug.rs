#[cfg(test)]
mod softmax_tests {
    use crate::mcts::softmax;
    
    #[test]
    fn test_softmax_with_neg_infinity() {
        // Test that masked moves get 0.0 probability
        let policy_logprobs = [0.0, f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY];
        let result = softmax(policy_logprobs);
        
        println!("Input: {:?}", policy_logprobs);
        println!("Output: {:?}", result);
        
        // Only moves 0 and 2 should have non-zero probability
        assert_eq!(result[1], 0.0);
        assert_eq!(result[3], 0.0);
        assert!(result[0] > 0.0);
        assert!(result[2] > 0.0);
        
        // Probabilities should sum to 1
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
