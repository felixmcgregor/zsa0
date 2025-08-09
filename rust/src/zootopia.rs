use core::fmt;
use std::fmt::Display;

use more_asserts::debug_assert_gt;
use serde::{Deserialize, Serialize};

use crate::types::{Policy, QValue};

/// Game state data structure matching the JSON format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    #[serde(rename = "TimeStamp")]
    pub timestamp: String,
    #[serde(rename = "Tick")]
    pub tick: u32,
    #[serde(rename = "Cells")]
    pub cells: Vec<Cell>,
    #[serde(rename = "Animals")]
    pub animals: Vec<Animal>,
    #[serde(rename = "Zookeepers")]
    pub zookeepers: Vec<Zookeeper>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cell {
    #[serde(rename = "Content")]
    pub content: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Animal {
    pub x: usize,
    pub y: usize,
    pub id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Zookeeper {
    pub x: usize,
    pub y: usize,
    pub id: u32,
}

/// Zootopia position representing the game state
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Pos {
    /// Grid dimensions - assuming square grid for now
    width: usize,
    height: usize,
    /// Flattened grid of cell contents
    cells: Vec<u8>,
    /// Player position
    player_x: usize,
    player_y: usize,
    /// Zookeeper positions
    zookeepers: Vec<(usize, usize)>,
    /// Current tick/turn number
    tick: u32,
    /// Score
    score: u32,
}

/// Cell content values matching the C# enum
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CellContent {
    Empty = 0,
    Wall = 1,
    Pellet = 2,
    ZookeeperSpawn = 3,
    AnimalSpawn = 4,
    PowerPellet = 5,
    ChameleonCloak = 6,
    Scavenger = 7,
    BigMooseJuice = 8,
}

/// Possible terminal states of the zootopia game
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum TerminalState {
    Success,    // All pellets collected
    Failure,    // Caught by zookeeper  
    Timeout,    // Game timeout/draw
    InProgress, // Game still going
}

/// Valid moves in the game
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Move {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

impl Default for Pos {
    fn default() -> Self {
        Pos {
            width: 20,  // Default grid size
            height: 20,
            cells: vec![0; 400], // 20x20 grid of empty cells
            player_x: 10,
            player_y: 10,
            zookeepers: vec![],
            tick: 0,
            score: 0,
        }
    }
}

impl Pos {
    /// Default grid dimensions (can be overridden when loading from JSON)
    pub const DEFAULT_WIDTH: usize = 20;
    pub const DEFAULT_HEIGHT: usize = 20;
    
    /// For compatibility with Connect Four interface
    pub const N_COLS: usize = Self::N_MOVES; // Map to number of moves for policy arrays
    pub const N_ROWS: usize = Self::DEFAULT_HEIGHT; // Map to height for buffer calculations
    
    /// Number of possible moves (up, down, left, right)
    pub const N_MOVES: usize = 4;

    /// The number of channels in the numpy buffer (grid state + player position + other features)
    pub const BUF_N_CHANNELS: usize = 3; // grid content, player position, additional features
    /// The required length (in # of f32s) of the numpy buffer
    pub const BUF_LEN: usize = Self::BUF_N_CHANNELS * Self::DEFAULT_WIDTH * Self::DEFAULT_HEIGHT;

    /// Creates a new position from a JSON game state
    pub fn from_game_state(game_state: &GameState) -> Self {
        // Infer grid dimensions from the cells array length
        let total_cells = game_state.cells.len();
        let width = (total_cells as f64).sqrt() as usize;
        let height = total_cells / width;
        
        let cells: Vec<u8> = game_state.cells.iter().map(|cell| cell.content).collect();
        
        // Find player position (assuming first animal is the player)
        let (player_x, player_y) = if let Some(animal) = game_state.animals.first() {
            (animal.x, animal.y)
        } else {
            (width / 2, height / 2) // Default to center if no animals
        };

        // Extract zookeeper positions
        let zookeepers: Vec<(usize, usize)> = game_state.zookeepers
            .iter()
            .map(|zk| (zk.x, zk.y))
            .collect();

        Pos {
            width,
            height,
            cells,
            player_x,
            player_y,
            zookeepers,
            tick: game_state.tick,
            score: 0, // Initialize score to 0, could be calculated from collected pellets
        }
    }

    /// Makes a move in the given direction
    /// Returns a new position if the move is valid, None otherwise
    pub fn make_move(&self, mov: Move) -> Option<Pos> {
        let (new_x, new_y) = match mov {
            Move::Up => (self.player_x, (self.player_y + self.height - 1) % self.height),
            Move::Down => (self.player_x, (self.player_y + 1) % self.height),
            Move::Left => ((self.player_x + self.width - 1) % self.width, self.player_y),
            Move::Right => ((self.player_x + 1) % self.width, self.player_y),
        };

        // Check if the move is valid (not into a wall or zookeeper spawn)
        match self.get_cell_content(new_x, new_y) {
            Some(CellContent::Wall) | Some(CellContent::ZookeeperSpawn) => return None,
            _ => {}
        }

        let mut new_pos = self.clone();
        new_pos.player_x = new_x;
        new_pos.player_y = new_y;
        new_pos.tick += 1;

        // Check if player is captured by any zookeeper
        if new_pos.zookeepers.iter().any(|&(zk_x, zk_y)| zk_x == new_x && zk_y == new_y) {
            // Player is captured, but we still return the position so the terminal state can be detected
            return Some(new_pos);
        }

        // Handle pellet collection
        if let Some(content) = self.get_cell_content(new_x, new_y) {
            match content {
                CellContent::Pellet => {
                    new_pos.score += 3;
                    new_pos.set_cell_content(new_x, new_y, CellContent::Empty);
                }
                CellContent::PowerPellet => {
                    new_pos.score += 30;
                    new_pos.set_cell_content(new_x, new_y, CellContent::Empty);
                }
                CellContent::ChameleonCloak => {
                    new_pos.score += 1;
                    new_pos.set_cell_content(new_x, new_y, CellContent::Empty);
                }
                CellContent::Scavenger => {
                    new_pos.score += 100;
                    new_pos.set_cell_content(new_x, new_y, CellContent::Empty);
                }
                CellContent::BigMooseJuice => {
                    new_pos.score += 60;
                    new_pos.set_cell_content(new_x, new_y, CellContent::Empty);
                }
                _ => {}
            }
        }

        Some(new_pos)
    }

    /// Gets the content of a cell at the given position
    pub fn get_cell_content(&self, x: usize, y: usize) -> Option<CellContent> {
        if x >= self.width || y >= self.height {
            return None;
        }
        
        let index = y * self.width + x;
        if index >= self.cells.len() {
            return None;
        }

        match self.cells[index] {
            0 => Some(CellContent::Empty),
            1 => Some(CellContent::Wall),
            2 => Some(CellContent::Pellet),
            3 => Some(CellContent::ZookeeperSpawn),
            4 => Some(CellContent::AnimalSpawn),
            5 => Some(CellContent::PowerPellet),
            6 => Some(CellContent::ChameleonCloak),
            7 => Some(CellContent::Scavenger),
            8 => Some(CellContent::BigMooseJuice),
            _ => Some(CellContent::Empty),
        }
    }

    /// Sets the content of a cell at the given position
    pub fn set_cell_content(&mut self, x: usize, y: usize, content: CellContent) {
        if x >= self.width || y >= self.height {
            return;
        }
        
        let index = y * self.width + x;
        if index >= self.cells.len() {
            return;
        }

        self.cells[index] = content as u8;
    }

    /// Returns the current tick/turn number
    pub fn ply(&self) -> usize {
        self.tick as usize
    }

    /// Returns the current score
    pub fn score(&self) -> u32 {
        self.score
    }

    /// For compatibility with Connect Four interface - returns a sequence of moves
    /// In Zootopia, this is not meaningful but we provide a stub implementation
    pub fn to_moves(&self) -> Vec<crate::zootopia::Move> {
        // For now, return an empty vector since Zootopia doesn't have a deterministic move sequence
        vec![]
    }

    /// For compatibility with Connect Four interface - horizontal flip
    /// In Zootopia this creates a horizontally mirrored version of the grid
    pub fn flip_h(&self) -> Pos {
        let mut flipped = self.clone();
        
        // Flip the grid horizontally
        for y in 0..self.height {
            for x in 0..self.width {
                let flipped_x = self.width - 1 - x;
                let original_content = self.get_cell_content(x, y);
                if let Some(content) = original_content {
                    flipped.set_cell_content(flipped_x, y, content);
                }
            }
        }
        
        // Flip player position
        flipped.player_x = self.width - 1 - self.player_x;
        
        flipped
    }

    /// Determines if the game is over
    pub fn is_terminal_state(&self) -> Option<TerminalState> {
        // Check if player is captured by any zookeeper
        if self.zookeepers.iter().any(|&(zk_x, zk_y)| zk_x == self.player_x && zk_y == self.player_y) {
            return Some(TerminalState::Failure);
        }

        // Check if all pellets are collected (win condition)
        let has_pellets = self.cells.iter().any(|&cell| {
            matches!(cell, 2 | 5) // Pellet or PowerPellet
        });

        if !has_pellets {
            return Some(TerminalState::Success);
        }

        // For now, assume game continues
        Some(TerminalState::InProgress)
    }

    /// Returns which moves are legal from the current position
    pub fn legal_moves(&self) -> [bool; Self::N_MOVES] {
        [
            self.make_move(Move::Up).is_some(),
            self.make_move(Move::Down).is_some(),
            self.make_move(Move::Left).is_some(),
            self.make_move(Move::Right).is_some(),
        ]
    }

    /// Mask the policy logprobs by setting illegal moves to f32::NEG_INFINITY
    pub fn mask_policy(&self, policy_logprobs: &mut Policy) {
        let legal_moves = self.legal_moves();
        debug_assert_gt!(
            legal_moves.iter().filter(|&&legal| legal).count(),
            0,
            "At least one move must be legal"
        );

        // Mask policy for illegal moves
        for mov in 0..Self::N_MOVES {
            if !legal_moves[mov] {
                policy_logprobs[mov] = f32::NEG_INFINITY;
            }
        }
    }

    /// Returns the terminal value with ply penalty, returns None if game is not over
    pub fn terminal_value_with_ply_penalty(&self, c_ply_penalty: f32) -> Option<(QValue, QValue)> {
        match self.is_terminal_state() {
            Some(TerminalState::Success) => {
                let q_no_penalty = 1.0;
                let q_penalty = q_no_penalty - (self.ply() as f32 * c_ply_penalty);
                Some((q_penalty.clamp(-1.0, 1.0), q_no_penalty))
            }
            Some(TerminalState::Failure) => Some((-1.0, -1.0)),
            Some(TerminalState::Timeout) => Some((0.0, 0.0)),
            Some(TerminalState::InProgress) => None,
            None => None,
        }
    }

    /// For compatibility with Connect Four interface - invert the position perspective
    /// In Zootopia this doesn't change the position since it's single-player
    pub fn invert(self) -> Pos {
        self // No-op for single player game
    }

    /// Returns the player position as a tuple (x, y)
    pub fn player_position(&self) -> (usize, usize) {
        (self.player_x, self.player_y)
    }

    /// Returns the positions of all zookeepers as a slice
    pub fn zookeeper_positions(&self) -> &[(usize, usize)] {
        &self.zookeepers
    }

    /// Returns the dimensions of the grid as (width, height)
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Writes the position to a buffer for neural network input
    pub fn write_numpy_buffer(&self, buf: &mut [f32]) {
        assert_eq!(buf.len(), Self::BUF_LEN);
        
        let channel_size = self.width * self.height;
        
        // Channel 0: Grid content (normalized)
        for i in 0..self.cells.len() {
            buf[i] = self.cells[i] as f32 / 8.0; // Normalize cell content values
        }
        
        // Channel 1: Player position
        for i in 0..channel_size {
            buf[channel_size + i] = 0.0;
        }
        let player_index = self.player_y * self.width + self.player_x;
        if player_index < channel_size {
            buf[channel_size + player_index] = 1.0;
        }
        
        // Channel 2: Additional features (score, tick, etc.)
        for i in 0..channel_size {
            buf[2 * channel_size + i] = self.score as f32 / 1000.0; // Normalized score
        }
    }

    /// For compatibility with Connect Four interface - creates position from move sequence
    /// In Zootopia this creates a simple test position
    pub fn from_moves(moves: &[crate::zootopia::Move]) -> Pos {
        let mut pos = Pos::default();
        for &mov in moves {
            if let Some(new_pos) = pos.make_move(mov) {
                pos = new_pos;
            }
        }
        pos
    }
}

impl Display for Pos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = String::new();
        
        for y in 0..self.height {
            for x in 0..self.width {
                let char = if x == self.player_x && y == self.player_y {
                    'P' // Player
                } else {
                    // ⚫🔵🔴
                    match self.get_cell_content(x, y) {
                        Some(CellContent::Empty) => ' ',
                        Some(CellContent::Wall) => '#',
                        Some(CellContent::Pellet) => '•',
                        Some(CellContent::ZookeeperSpawn) => 'Z',
                        Some(CellContent::AnimalSpawn) => '🔴',
                        Some(CellContent::PowerPellet) => 'P',
                        Some(CellContent::ChameleonCloak) => 'C',
                        Some(CellContent::Scavenger) => '🔵',
                        Some(CellContent::BigMooseJuice) => 'M',
                        None => '?',
                    }
                };
                result.push(char);
            }
            result.push('\n');
        }
        
        write!(f, "{}", result)
    }
}

impl fmt::Debug for Pos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pos {{ width: {}, height: {}, player: ({}, {}), tick: {}, score: {} }}",
            self.width, self.height, self.player_x, self.player_y, self.tick, self.score
        )
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_basic_movement() {
        let pos = Pos::default();
        
        // Test moving right
        let new_pos = pos.make_move(Move::Right).unwrap();
        assert_eq!(new_pos.player_position(), (11, 10));
        
        // Test moving up
        let new_pos = pos.make_move(Move::Up).unwrap();
        assert_eq!(new_pos.player_position(), (10, 9));
    }

    #[test]
    fn test_wall_collision() {
        let mut pos = Pos::default();
        // Set a wall to the right of the player
        pos.set_cell_content(11, 10, CellContent::Wall);
        
        // Moving right should fail
        assert!(pos.make_move(Move::Right).is_none());
    }

    #[test]
    fn test_pellet_collection() {
        let mut pos = Pos::default();
        // Place a pellet to the right of the player
        pos.set_cell_content(11, 10, CellContent::Pellet);
        
        let new_pos = pos.make_move(Move::Right).unwrap();
        assert_eq!(new_pos.score(), 3); // Pellet gives 3 points
        assert_eq!(new_pos.get_cell_content(11, 10), Some(CellContent::Empty));
    }

    #[test]
    fn test_legal_moves() {
        let pos = Pos::default();
        let legal_moves = pos.legal_moves();
        
        // All moves should be legal from the center of an empty grid
        assert!(legal_moves.iter().all(|&legal| legal));
    }

    #[test]
    fn test_display() {
        let pos = Pos::default();
        let display_string = format!("{}", pos);
        
        // Should contain the player character 'P'
        assert!(display_string.contains('P'));
    }

    #[test]
    fn test_from_game_state_json() {
        // Small 4x4 grid, 1 animal, 1 pellet, 1 wall, 1 zookeeper
        let json = r#"{
            "TimeStamp": "2025-08-07T00:00:00Z",
            "Tick": 1,
            "Cells": [
                {"Content": 0}, {"Content": 1}, {"Content": 2}, {"Content": 3},
                {"Content": 0}, {"Content": 0}, {"Content": 0}, {"Content": 0},
                {"Content": 0}, {"Content": 0}, {"Content": 0}, {"Content": 0},
                {"Content": 0}, {"Content": 0}, {"Content": 0}, {"Content": 0}
            ],
            "Animals": [{"x": 0, "y": 0, "id": 1}],
            "Zookeepers": [{"x": 3, "y": 0, "id": 1}]
        }"#;
        let game_state: GameState = serde_json::from_str(json).unwrap();
        let pos = Pos::from_game_state(&game_state);
        assert_eq!(pos.width, 4);
        assert_eq!(pos.height, 4);
        assert_eq!(pos.player_x, 0);
        assert_eq!(pos.player_y, 0);
        assert_eq!(pos.get_cell_content(1, 0), Some(CellContent::Wall));
        assert_eq!(pos.get_cell_content(2, 0), Some(CellContent::Pellet));
        assert_eq!(pos.get_cell_content(3, 0), Some(CellContent::ZookeeperSpawn));
    }

    #[test]
    fn test_from_game_state_json_small() {
        // Small 3x3 grid, 1 animal, 1 pellet, 1 wall, 1 zookeeper
        let json = r#"{
            "TimeStamp": "2025-08-07T00:00:00Z",
            "Tick": 1,
            "Cells": [
                {"Content": 0}, {"Content": 1}, {"Content": 2},
                {"Content": 3}, {"Content": 0}, {"Content": 0},
                {"Content": 0}, {"Content": 0}, {"Content": 0}
            ],
            "Animals": [{"x": 0, "y": 0, "id": 1}],
            "Zookeepers": [{"x": 1, "y": 0, "id": 1}]
        }"#;
        let game_state: crate::zootopia::GameState = serde_json::from_str(json).unwrap();
        let pos = crate::zootopia::Pos::from_game_state(&game_state);
        assert_eq!(pos.width, 3);
        assert_eq!(pos.height, 3);
        assert_eq!(pos.player_x, 0);
        assert_eq!(pos.player_y, 0);
        assert_eq!(pos.get_cell_content(1, 0), Some(crate::zootopia::CellContent::Wall));
        assert_eq!(pos.get_cell_content(2, 0), Some(crate::zootopia::CellContent::Pellet));
        assert_eq!(pos.get_cell_content(0, 1), Some(crate::zootopia::CellContent::ZookeeperSpawn));
    }
    
    #[test]
    fn debug_wall_avoidance() {
        // Test the wall avoidance case from failing MCTS test
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

    #[test]
    fn test_zookeeper_capture() {
        let mut pos = Pos::default();
        // Place a zookeeper to the right of the player
        pos.zookeepers = vec![(11, 10)];
        
        // Moving right should result in capture
        let new_pos = pos.make_move(Move::Right).unwrap();
        assert_eq!(new_pos.is_terminal_state(), Some(TerminalState::Failure));
    }

    #[test]
    fn test_zookeeper_avoidance() {
        let mut pos = Pos::default();
        // Place a zookeeper to the left of the player
        pos.zookeepers = vec![(9, 10)];
        
        // Moving right should be safe
        let new_pos = pos.make_move(Move::Right).unwrap();
        assert_eq!(new_pos.is_terminal_state(), Some(TerminalState::InProgress));
        
        // Moving left should result in capture
        let captured_pos = pos.make_move(Move::Left).unwrap();
        assert_eq!(captured_pos.is_terminal_state(), Some(TerminalState::Failure));
    }

    #[test]
    fn test_from_game_state_with_zookeepers() {
        // Small 3x3 grid with zookeeper
        let json = r#"{
            "TimeStamp": "2025-08-07T00:00:00Z",
            "Tick": 1,
            "Cells": [
                {"Content": 0}, {"Content": 0}, {"Content": 2},
                {"Content": 0}, {"Content": 0}, {"Content": 0},
                {"Content": 0}, {"Content": 0}, {"Content": 0}
            ],
            "Animals": [{"x": 0, "y": 0, "id": 1}],
            "Zookeepers": [{"x": 1, "y": 0, "id": 1}]
        }"#;
        let game_state: GameState = serde_json::from_str(json).unwrap();
        let pos = Pos::from_game_state(&game_state);
        
        assert_eq!(pos.zookeeper_positions(), &[(1, 0)]);
        
        // Moving right should result in capture
        let new_pos = pos.make_move(Move::Right).unwrap();
        assert_eq!(new_pos.is_terminal_state(), Some(TerminalState::Failure));
    }
}
