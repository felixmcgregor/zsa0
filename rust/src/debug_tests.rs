#[cfg(test)]
mod debug_tests {
    use crate::zootopia::*;

    #[test]
    fn debug_wall_avoidance() {
        // Test the wall avoidance case
        let json = r#"{
            "TimeStamp": "2025-08-07T00:00:00Z",
            "Tick": 1,
            "Cells": [
                {"Content": 0}, {"Content": 1}, {"Content": 0},
                {"Content": 0}, {"Content": 0}, {"Content": 0},
                {"Content": 0}, {"Content": 0}, {"Content": 0}
            ],
            "Animals": [{"x": 0, "y": 0, "id": 1}],
            "Zookeepers": []
        }"#;
        
        let game_state: GameState = serde_json::from_str(json).unwrap();
        let pos = Pos::from_game_state(&game_state);
        
        println!("Position: {:?}", pos);
        println!("Player position: {:?}", pos.player_position());
        println!("Grid dimensions: {:?}", pos.dimensions());
        
        // Check what's at cell (1, 0) - should be a wall
        println!("Cell (1, 0): {:?}", pos.get_cell_content(1, 0));
        
        // Test each move
        for (i, move_name) in ["Up", "Down", "Left", "Right"].iter().enumerate() {
            let mov = match i {
                0 => Move::Up,
                1 => Move::Down,
                2 => Move::Left,
                3 => Move::Right,
                _ => unreachable!(),
            };
            
            let can_move = pos.make_move(mov).is_some();
            println!("Can move {}: {}", move_name, can_move);
        }
        
        // Check legal moves array
        let legal_moves = pos.legal_moves();
        println!("Legal moves: {:?}", legal_moves);
        
        // Test right move specifically (should fail due to wall)
        assert!(pos.make_move(Move::Right).is_none(), "Right move should be blocked by wall");
        assert!(!legal_moves[3], "Right move should be illegal");
    }
}
