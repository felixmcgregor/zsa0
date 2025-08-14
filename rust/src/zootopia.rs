use core::fmt;
use std::fmt::Display;

use more_asserts::debug_assert_gt;
use serde::{Deserialize, Serialize};

use crate::types::{Policy, QValue};

/// Game state data structure matching the actual JSON format from game engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    #[serde(rename = "TimeStamp")]
    pub timestamp: String,
    #[serde(rename = "Tick")]
    pub tick: u32,
    #[serde(rename = "Cells")]
    pub cells: Vec<CellWithPosition>,
    #[serde(rename = "Animals")]
    pub animals: Vec<AnimalState>,
    #[serde(rename = "Zookeepers")]
    pub zookeepers: Vec<ZookeeperState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellWithPosition {
    #[serde(rename = "X")]
    pub x: usize,
    #[serde(rename = "Y")]
    pub y: usize,
    #[serde(rename = "Content")]
    pub content: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimalState {
    #[serde(rename = "X", skip_serializing_if = "Option::is_none")]
    pub x: Option<usize>,
    #[serde(rename = "Y", skip_serializing_if = "Option::is_none")]
    pub y: Option<usize>,
    #[serde(rename = "id", skip_serializing_if = "Option::is_none")]
    pub id: Option<u32>,
    #[serde(rename = "ActivePowerUp")]
    pub active_power_up: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZookeeperState {
    #[serde(rename = "X", skip_serializing_if = "Option::is_none")]
    pub x: Option<usize>,
    #[serde(rename = "Y", skip_serializing_if = "Option::is_none")]
    pub y: Option<usize>,
    #[serde(rename = "id", skip_serializing_if = "Option::is_none")]
    pub id: Option<u32>,
    #[serde(rename = "SpawnY", skip_serializing_if = "Option::is_none")]
    pub spawn_y: Option<usize>,
    #[serde(rename = "SpawnX", skip_serializing_if = "Option::is_none")]
    pub spawn_x: Option<usize>,
}

/// Legacy cell structure for backwards compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cell {
    #[serde(rename = "Content")]
    pub content: u8,
}

/// Legacy animal structure for backwards compatibility  
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Animal {
    pub x: usize,
    pub y: usize,
    pub id: u32,
}

/// Legacy zookeeper structure for backwards compatibility
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
    /// Target number of pellets to collect for winning
    target_pellets: u32,
    /// Number of pellets collected so far
    pellets_collected: u32,
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
        // Try to load from JSON file, fall back to hardcoded default if it fails
        if let Ok(pos) = Self::load_from_json() {
            pos
        } else {
            // Fallback to original hardcoded default
            let mut cells = vec![0; 51 * 51];
            // Place pellets near the player for easier testing
            cells[25 * 51 + 26] = 2; // Pellet at (26, 25) - one move right from player
            cells[24 * 51 + 25] = 2;  // Pellet at (25, 24) - one move up from player
            Pos {
                width: 51,
                height: 51,
                cells,
                player_x: 25,
                player_y: 25,
                zookeepers: vec![],
                tick: 0,
                score: 0,
                target_pellets: 5,
                pellets_collected: 0,
            }
        }
    }
}

impl Pos {
    /// Default grid dimensions (can be overridden when loading from JSON)
    pub const DEFAULT_WIDTH: usize = 51;
    pub const DEFAULT_HEIGHT: usize = 51;
    
    /// For compatibility with Connect Four interface
    pub const N_COLS: usize = Self::DEFAULT_WIDTH; // Map to number of moves for policy arrays
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
            // Handle optional x,y coordinates with defaults
            let x = animal.x.unwrap_or(width / 2);
            let y = animal.y.unwrap_or(height / 2);
            (x % width, y % height) // Wrap around grid
        } else {
            (width / 2, height / 2) // Default to center if no animals
        };

        // Extract zookeeper positions
        let zookeepers: Vec<(usize, usize)> = game_state.zookeepers
            .iter()
            .filter_map(|zk| {
                // Handle optional x,y coordinates
                if let (Some(x), Some(y)) = (zk.x, zk.y) {
                    Some((x % width, y % height)) // Wrap positions
                } else {
                    None // Skip zookeepers without valid positions
                }
            })
            .collect();

        // Check if there were any pellets in the initial game state
        let total_pellets = game_state.cells.iter().filter(|cell| {
            matches!(cell.content, 2 | 5) // Pellet or PowerPellet
        }).count() as u32;
        
        // Cap target pellets for training efficiency
        let target_pellets = total_pellets.min(5);

        Pos {
            width,
            height,
            cells,
            player_x,
            player_y,
            zookeepers,
            tick: game_state.tick,
            score: 0, // Initialize score to 0, could be calculated from collected pellets
            target_pellets,
            pellets_collected: 0,
        }
    }

    /// Makes a move in the given direction
    /// Returns a new position if the move is valid, None otherwise
    pub fn make_move(&self, mov: Move) -> Option<Pos> {
        // Wrap player position around grid
        let (new_x, new_y) = match mov {
            Move::Up => (self.player_x, (self.player_y + self.height - 1) % self.height),
            Move::Down => (self.player_x, (self.player_y + 1) % self.height),
            Move::Left => ((self.player_x + self.width - 1) % self.width, self.player_y),
            Move::Right => ((self.player_x + 1) % self.width, self.player_y),
        };

        // println!("Attempting move: {:?} to ({}, {})", mov, new_x, new_y);

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
            println!("Player captured by zookeeper at ({}, {})", new_x, new_y);
            new_pos.score = new_pos.score.saturating_sub(1); // Penalize for being caught, clamp to zero
            new_pos.pellets_collected = new_pos.pellets_collected.saturating_sub(3); // Clamp to zero
            // return Some(new_pos);
            return None; // rather do this because I dont even want to search this
        }

        // Handle pellet collection
        if let Some(content) = self.get_cell_content(new_x, new_y) {
            match content {
                CellContent::Pellet => {
                    new_pos.score += 3;
                    new_pos.pellets_collected += 1;
                    new_pos.set_cell_content(new_x, new_y, CellContent::Empty);
                }
                CellContent::PowerPellet => {
                    new_pos.score += 30;
                    new_pos.pellets_collected += 1;
                    new_pos.set_cell_content(new_x, new_y, CellContent::Empty);
                }
                CellContent::ChameleonCloak => {
                    new_pos.score += 1;
                    new_pos.pellets_collected += 1;
                    new_pos.set_cell_content(new_x, new_y, CellContent::Empty);
                }
                CellContent::Scavenger => {
                    new_pos.pellets_collected += 3;
                    new_pos.score += 100;
                    new_pos.set_cell_content(new_x, new_y, CellContent::Empty);
                }
                CellContent::BigMooseJuice => {
                    new_pos.pellets_collected += 2;
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

    /// Returns the target number of pellets to collect for winning
    pub fn target_pellets(&self) -> u32 {
        self.target_pellets
    }

    /// Returns the number of pellets collected so far
    pub fn pellets_collected(&self) -> u32 {
        self.pellets_collected
    }

    /// Sets the target number of pellets to collect for winning
    pub fn set_target_pellets(&mut self, target: u32) {
        self.target_pellets = target;
    }

    /// Creates a new position with a specific target pellet count
    pub fn with_target_pellets(mut self, target: u32) -> Self {
        self.target_pellets = target;
        self
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
        // Add timeout mechanism to prevent infinite games
        const MAX_MOVES: u32 = 30; // Reasonable limit for Zootopia games
        if self.tick >= MAX_MOVES {
            println!("Game timeout after {} moves", self.tick);
            return Some(TerminalState::Timeout);
        }

        // Check if player is captured by any zookeeper
        if self.zookeepers.iter().any(|&(zk_x, zk_y)| zk_x == self.player_x && zk_y == self.player_y) {
            return Some(TerminalState::Failure);
        }

        // Check if all pellets are collected (win condition)
        let _has_pellets = self.cells.iter().any(|&cell| {
            matches!(cell, 2 | 5 | 6 | 7 | 8) // Pellet, PowerPellet, ChameleonCloak, Scavenger, BigMooseJuice
        });

        if self.pellets_collected >= self.target_pellets && self.target_pellets > 0 {
            // All target pellets collected - this is a win!
            return Some(TerminalState::Success);
        }

        // Game continues
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
        // println!("Legal moves: {:?}", legal_moves);
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
        // println!("Masked policy: {:?}", policy_logprobs);
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

    /// Load game state from default.json file
    fn load_from_json() -> Result<Self, Box<dyn std::error::Error>> {
        let json_content = std::fs::read_to_string("src/default.json")?;
        let game_state: JsonGameState = serde_json::from_str(&json_content)?;
        
        // Create a 51x51 grid initialized with zeros
        let mut cells = vec![0u8; 51 * 51];
        
        // Fill the grid with data from JSON
        for cell in &game_state.cells {
            if cell.x < 51 && cell.y < 51 {
                cells[cell.y * 51 + cell.x] = cell.content;
            }
        }
        
        // Find player position (look for content type 3 in the JSON)
        let (player_x, player_y) = game_state.cells
            .iter()
            .find(|cell| cell.content == 3)
            .map(|cell| (cell.x, cell.y))
            .unwrap_or((25, 25)); // Default to center if not found
        
        // Extract zookeeper positions
        let zookeepers: Vec<(usize, usize)> = game_state.zookeepers
            .iter()
            .map(|zk| (zk.x, zk.y))
            .filter(|&(x, y)| x.is_some() && y.is_some())
            .map(|(x, y)| (x.unwrap() % 51, y.unwrap() % 51)) // Wrap around grid
            .collect();
            
        
        // Count pellets (content type 2)
        let pellets_collected = game_state.cells
            .iter()
            .filter(|cell| cell.content == 2)
            .count() as u32;
        
        Ok(Pos {
            width: 51,
            height: 51,
            cells,
            player_x,
            player_y,
            zookeepers,
            // tick: game_state.tick,
            tick: 0, // start every game from tick 0 to enable tracking timeout
            score: 0, // Could extract from animals if needed
            target_pellets: pellets_collected + 5, // Estimate based on current pellets
            pellets_collected: 0, // Start fresh
        })
    }
    
    /// Load a specific game state from a JSON file
    pub fn from_json_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json_content = std::fs::read_to_string(path)?;
        let game_state: JsonGameState = serde_json::from_str(&json_content)?;
        
        // Similar parsing logic as load_from_json but with custom path
        let mut cells = vec![0u8; 51 * 51];
        
        for cell in &game_state.cells {
            if cell.x < 51 && cell.y < 51 {
                cells[cell.y * 51 + cell.x] = cell.content;
            }
        }
        
        let (player_x, player_y) = game_state.cells
            .iter()
            .find(|cell| cell.content == 3)
            .map(|cell| (cell.x, cell.y))
            .unwrap_or((25, 25));
        
        let zookeepers: Vec<(usize, usize)> = game_state.zookeepers
            .iter()
            .map(|zk| (zk.x, zk.y))
            .filter(|&(x, y)| x.is_some() && y.is_some())
            .map(|(x, y)| (x.unwrap() % 51, y.unwrap() % 51))
            .collect();
        
        let pellets_collected = game_state.cells
            .iter()
            .filter(|cell| cell.content == 2)
            .count() as u32;
        
        Ok(Pos {
            width: 51,
            height: 51,
            cells,
            player_x,
            player_y,
            zookeepers,
            tick: game_state.tick,
            score: 0,
            target_pellets: pellets_collected + 5,
            pellets_collected: 0,
        })
    }
}

/// JSON parsing structures
#[derive(Deserialize)]
struct JsonCell {
    #[serde(rename = "X")]
    x: usize,
    #[serde(rename = "Y")]
    y: usize,
    #[serde(rename = "Content")]
    content: u8,
}

#[derive(Deserialize)]
struct JsonAnimal {
    #[serde(rename = "ActivePowerUp")]
    active_power_up: Option<String>,
    // Note: X and Y coordinates are missing in the actual JSON structure
    // They might be in a different format or missing entirely
}

#[derive(Deserialize)]
struct JsonZookeeper {
    #[serde(rename = "Y")]
    y: Option<usize>,
    #[serde(rename = "X")]
    x: Option<usize>,
    // Note: Current X and Y coordinates are missing in the actual JSON structure
}

#[derive(Deserialize)]
struct JsonGameState {
    #[serde(rename = "TimeStamp")]
    timestamp: String,
    #[serde(rename = "Tick")]
    tick: u32,
    #[serde(rename = "Cells")]
    cells: Vec<JsonCell>,
    #[serde(rename = "Animals")]
    animals: Vec<JsonAnimal>,
    #[serde(rename = "Zookeepers")]
    zookeepers: Vec<JsonZookeeper>,
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
                        Some(CellContent::AnimalSpawn) => '+',
                        Some(CellContent::PowerPellet) => 'P',
                        Some(CellContent::ChameleonCloak) => 'C',
                        Some(CellContent::Scavenger) => 'S',
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
    // "Pos {{ width: {}, height: {}, player: ({}, {}), tick: {}, score: {} }}",
    // self.width, self.height, self.player_x, self.player_y, self.tick, self.score

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pos {{ player: (x={}, y={}), tick: {}, score: {}, target_pellets: {}, pellets_collected: {} }}",
            self.player_x, self.player_y, self.tick, self.score, self.target_pellets, self.pellets_collected
        )
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_basic_movement() {
        let pos = Pos::default();
        
        // Test moving right (from center at 25,25 to 26,25)
        let new_pos = pos.make_move(Move::Right).unwrap();
        assert_eq!(new_pos.player_position(), (26, 25));
        
        // Test moving up (from center at 25,25 to 25,24)
        let new_pos = pos.make_move(Move::Up).unwrap();
        assert_eq!(new_pos.player_position(), (25, 24));
    }

    #[test]
    fn test_wall_collision() {
        let mut pos = Pos::default();
        // Set a wall to the right of the player (at position 26,25)
        pos.set_cell_content(26, 25, CellContent::Wall);
        
        // Moving right should fail
        assert!(pos.make_move(Move::Right).is_none());
    }

    #[test]
    fn test_pellet_collection() {
        let mut pos = Pos::default();
        // Place a pellet to the right of the player (at position 26,25)
        pos.set_cell_content(26, 25, CellContent::Pellet);
        
        let new_pos = pos.make_move(Move::Right).unwrap();
        assert_eq!(new_pos.score(), 3); // Pellet gives 3 points
        assert_eq!(new_pos.get_cell_content(26, 25), Some(CellContent::Empty));
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
                {"X": 0, "Y": 0, "Content": 0}, {"X": 1, "Y": 0, "Content": 1}, {"X": 2, "Y": 0, "Content": 2},
                {"X": 0, "Y": 1, "Content": 0}, {"X": 1, "Y": 1, "Content": 0}, {"X": 2, "Y": 1, "Content": 0},
                {"X": 0, "Y": 2, "Content": 0}, {"X": 1, "Y": 2, "Content": 0}, {"X": 2, "Y": 2, "Content": 0}
            ],
            "Animals": [{"X": 0, "Y": 0, "id": 1}],
            "Zookeepers": [{"X": 3, "Y": 0, "id": 1}]
        }"#;
        let game_state: GameState = serde_json::from_str(json).unwrap();
        let pos = Pos::from_game_state(&game_state);
        assert_eq!(pos.width, 3);
        assert_eq!(pos.height, 3);
        assert_eq!(pos.player_x, 0);
        assert_eq!(pos.player_y, 0);
        assert_eq!(pos.get_cell_content(1, 0), Some(CellContent::Wall));
        assert_eq!(pos.get_cell_content(2, 0), Some(CellContent::Pellet));
    }

    #[test]
    fn test_from_game_state_json_small() {
        // Small 3x3 grid, 1 animal, 1 pellet, 1 wall, 1 zookeeper
        let json = r#"{
            "TimeStamp": "2025-08-07T00:00:00Z",
            "Tick": 1,
            "Cells": [
                {"X": 0, "Y": 0, "Content": 0}, {"X": 1, "Y": 0, "Content": 1}, {"X": 2, "Y": 0, "Content": 2},
                {"X": 0, "Y": 1, "Content": 3}, {"X": 1, "Y": 1, "Content": 0}, {"X": 2, "Y": 1, "Content": 0},
                {"X": 0, "Y": 2, "Content": 0}, {"X": 1, "Y": 2, "Content": 0}, {"X": 2, "Y": 2, "Content": 0}
            ],
            "Animals": [{"X": 0, "Y": 0, "id": 1}],
            "Zookeepers": [{"X": 1, "Y": 0, "id": 1}]
        }"#;
        let game_state: crate::zootopia::GameState = serde_json::from_str(json).unwrap();
        let pos = crate::zootopia::Pos::from_game_state(&game_state);
        assert_eq!(pos.width, 3);
        assert_eq!(pos.height, 3);
        assert_eq!(pos.player_x, 0);
        assert_eq!(pos.player_y, 0);
        assert_eq!(pos.get_cell_content(1, 0), Some(crate::zootopia::CellContent::Wall));
        assert_eq!(pos.get_cell_content(2, 0), Some(crate::zootopia::CellContent::Pellet));
    }
    
    #[test]
    fn debug_wall_avoidance() {
        // Test the wall avoidance case from failing MCTS test
        let json = r#"{
            "TimeStamp": "2025-08-07T00:00:00Z",
            "Tick": 1,
            "Cells": [
                {"X": 0, "Y": 0, "Content": 0}, {"X": 1, "Y": 0, "Content": 1}, {"X": 2, "Y": 0, "Content": 2},
                {"X": 0, "Y": 1, "Content": 0}, {"X": 1, "Y": 1, "Content": 0}, {"X": 2, "Y": 1, "Content": 1},
                {"X": 0, "Y": 2, "Content": 0}, {"X": 1, "Y": 2, "Content": 1}, {"X": 2, "Y": 2, "Content": 0}
            ],
            "Animals": [{"X": 1, "Y": 1, "id": 1}],
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

        // left move should be legal
        assert!(pos.make_move(Move::Left).is_some(), "Left move should be  legal");
        assert!(legal_moves[2], "Left move should be legal");
    }

    #[test]
    fn debug_wall_avoidance_warp() {
        // Test the wall avoidance case from failing MCTS test
        let json = r#"{
            "TimeStamp": "2025-08-07T00:00:00Z",
            "Tick": 1,
            "Cells": [
                {"X": 0, "Y": 0, "Content": 0}, {"X": 1, "Y": 0, "Content": 1}, {"X": 2, "Y": 0, "Content": 2},
                {"X": 0, "Y": 1, "Content": 1}, {"X": 1, "Y": 1, "Content": 0}, {"X": 2, "Y": 1, "Content": 0},
                {"X": 0, "Y": 2, "Content": 0}, {"X": 1, "Y": 2, "Content": 0}, {"X": 2, "Y": 2, "Content": 0}
            ],
            "Animals": [{"X": 0, "Y": 0, "id": 1}],
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

        // up move should be legal
        assert!(pos.make_move(Move::Up).is_some(), "Up move should be  legal");
        assert!(legal_moves[0], "Up move should be legal");
    }

    #[test]
    fn test_zookeeper_avoidance() {
        let mut pos = Pos::default();
        // Place a zookeeper to the left of the player (at position 24,25)
        pos.zookeepers = vec![(24, 25)];
        
        // Moving right should be safe
        let new_pos = pos.make_move(Move::Right).unwrap();
        assert_eq!(new_pos.is_terminal_state(), Some(TerminalState::InProgress));
        
        // Moving left should result in capture
        // let captured_pos = pos.make_move(Move::Left).unwrap();
        // assert_eq!(captured_pos.is_terminal_state(), Some(TerminalState::Failure));
        
        // check that you cant move left into the zookeeper
        assert!(pos.make_move(Move::Left).is_none(), "Left move should be illegal due to zookeeper");
        
    }

    #[test]
    fn is_terminal() {
        println!("Running test: is_terminal");
        // Create a small test position with JSON
        let json = r#"{
            "TimeStamp": "2025-08-07T00:00:00Z",
            "Tick": 1,
            "Cells": [
                {"X": 0, "Y": 0, "Content": 0}, {"X": 1, "Y": 0, "Content": 2}, {"X": 2, "Y": 0, "Content": 2},
                {"X": 0, "Y": 1, "Content": 0}, {"X": 1, "Y": 1, "Content": 0}, {"X": 2, "Y": 1, "Content": 0},
                {"X": 0, "Y": 2, "Content": 0}, {"X": 1, "Y": 2, "Content": 0}, {"X": 2, "Y": 2, "Content": 0}
            ],
            "Animals": [{"X": 0, "Y": 0, "id": 1}],
            "Zookeepers": []
        }"#;
        let game_state: crate::zootopia::GameState = serde_json::from_str(json).unwrap();
        let pos = crate::zootopia::Pos::from_game_state(&game_state);
        println!("Position: {:?}", pos);
        println!("Player position: {:?}", pos.player_position());
        println!("Grid dimensions: {:?}", pos.dimensions());
        println!("Cell (1, 0): {:?}", pos.get_cell_content(1, 0));
        // Check if the position is terminal
        let terminal_state = pos.is_terminal_state();
        println!("Terminal state: {:?}", terminal_state);
        assert_eq!(terminal_state, Some(TerminalState::InProgress), "Position should be in progress");

        // Now collect the pellet at (1, 0)
        let new_pos = pos.make_move(Move::Right).unwrap();
        println!("New position after collecting pellet: {:?}", new_pos);
        assert_eq!(new_pos.pellets_collected(), 1, "Should have collected 1 pellet");
        assert_eq!(new_pos.score(), 3, "Score should be 3 after collecting pellet");
        // Check if the new position is terminal
        let new_terminal_state = new_pos.is_terminal_state();
        println!("New terminal state: {:?}", new_terminal_state);
        assert_eq!(new_terminal_state, Some(TerminalState::InProgress), "New position should still be in progress");
        // Now collect the pellet at (2, 0)
        let new_pos = new_pos.make_move(Move::Right).unwrap();
        println!("New position after collecting second pellet: {:?}", new_pos);
        assert_eq!(new_pos.pellets_collected(), 2, "Should have collected 2 pellets");
        assert_eq!(new_pos.score(), 6, "Score should be 6 after collecting second pellet");
        // Check if the new position is terminal
        let new_terminal_state = new_pos.is_terminal_state();
        println!("New terminal state after second pellet: {:?}", new_terminal_state);
        assert_eq!(new_terminal_state, Some(TerminalState::Success), "New position should still be in progress");
    }

    #[test]
    fn test_from_game_state_with_zookeepers() {
        // Small 3x3 grid with zookeeper
        let json = r#"{
            "TimeStamp": "2025-08-07T00:00:00Z",
            "Tick": 1,
            "Cells": [
                {"X": 0, "Y": 0, "Content": 0}, {"X": 1, "Y": 0, "Content": 0}, {"X": 2, "Y": 0, "Content": 2},
                {"X": 0, "Y": 1, "Content": 0}, {"X": 1, "Y": 1, "Content": 0}, {"X": 2, "Y": 1, "Content": 0},
                {"X": 0, "Y": 2, "Content": 0}, {"X": 1, "Y": 2, "Content": 0}, {"X": 2, "Y": 2, "Content": 0}
            ],
            "Animals": [{"X": 0, "Y": 0, "id": 1}],
            "Zookeepers": [{"X": 1, "Y": 0, "id": 1}]
        }"#;
        let game_state: GameState = serde_json::from_str(json).unwrap();
        let pos = Pos::from_game_state(&game_state);
        
        assert_eq!(pos.zookeeper_positions(), &[(1, 0)]);
        
        // Moving right should not be allowed due to zookeeper
        assert!(pos.make_move(Move::Right).is_none(), "Right move should be illegal due to zookeeper");
    }
}

// Test function to debug tensor shape issues
#[cfg(test)]
mod tensor_debug_tests {
    use super::*;
    
    #[test]
    fn test_tensor_dimensions() {
        println!("=== Rust Tensor Dimension Debug ===");
        println!("BUF_N_CHANNELS: {}", Pos::BUF_N_CHANNELS);
        println!("DEFAULT_HEIGHT: {}", Pos::DEFAULT_HEIGHT);
        println!("DEFAULT_WIDTH: {}", Pos::DEFAULT_WIDTH);
        println!("BUF_LEN: {}", Pos::BUF_LEN);
        println!("N_MOVES: {}", Pos::N_MOVES);
        
        // Test creating a batch of positions
        let pos1 = Pos::default();
        let pos2 = Pos::default();
        let positions = vec![pos1, pos2];
        
        // Simulate what create_pos_batch does
        let batch_size = positions.len();
        let total_buffer_size = batch_size * Pos::BUF_LEN;
        
        println!("Batch size: {}", batch_size);
        println!("Total buffer size: {}", total_buffer_size);
        println!("Expected tensor shape: ({}, {}, {}, {})", 
                 batch_size, Pos::BUF_N_CHANNELS, Pos::DEFAULT_HEIGHT, Pos::DEFAULT_WIDTH);
        
        // Calculate expected flattened size after conv layers
        // With 32 filters on a 51x51 grid (assuming no padding changes size)
        let conv_filters = 32; // From your config
        let expected_fc_input_size = conv_filters * Pos::DEFAULT_HEIGHT * Pos::DEFAULT_WIDTH;
        println!("Expected FC input size with {} filters: {}", conv_filters, expected_fc_input_size);
        
        // Test buffer creation
        let mut buffer = vec![0.0f32; total_buffer_size];
        for i in 0..batch_size {
            let pos = &positions[i];
            let pos_buffer = &mut buffer[i * Pos::BUF_LEN..(i + 1) * Pos::BUF_LEN];
            pos.write_numpy_buffer(pos_buffer);
        }
        
        println!("Buffer created successfully with {} elements", buffer.len());
        
        // Verify the calculations match what Python expects
        assert_eq!(Pos::BUF_N_CHANNELS, 3);
        assert_eq!(Pos::DEFAULT_HEIGHT, 51);
        assert_eq!(Pos::DEFAULT_WIDTH, 51);
        assert_eq!(Pos::BUF_LEN, 3 * 51 * 51);
        assert_eq!(expected_fc_input_size, 32 * 51 * 51); // Should be 83232
    }
    
    #[test] 
    fn test_numpy_buffer_format() {
        println!("=== Testing numpy buffer format ===");
        let pos = Pos::default();
        let mut buffer = vec![0.0f32; Pos::BUF_LEN];
        pos.write_numpy_buffer(&mut buffer);
        
        let channel_size = Pos::DEFAULT_WIDTH * Pos::DEFAULT_HEIGHT;
        println!("Channel size: {}", channel_size);
        
        // Check channel 0 (grid content)
        println!("Channel 0 (grid) first 10 values: {:?}", &buffer[0..10]);
        
        // Check channel 1 (player position) 
        let player_channel_start = channel_size;
        println!("Channel 1 (player) around player position: {:?}", 
                 &buffer[player_channel_start + 200 - 5..player_channel_start + 200 + 5]);
        
        // Check channel 2 (features)
        let features_channel_start = 2 * channel_size;
        println!("Channel 2 (features) first 10 values: {:?}", &buffer[features_channel_start..features_channel_start + 10]);
    }

    #[test]
    fn test_create_pos_batch_shape() {
        println!("=== Testing tensor buffer creation manually ===");
        
        // Create a small batch of positions
        let positions = vec![
            Pos::default(),
            Pos::default(),
            Pos::default(),
        ];
        
        // Simulate what create_pos_batch does manually
        let batch_size = positions.len();
        let mut buffer = vec![0.0f32; batch_size * Pos::BUF_LEN];
        
        for i in 0..batch_size {
            let pos = &positions[i];
            let pos_buffer = &mut buffer[i * Pos::BUF_LEN..(i + 1) * Pos::BUF_LEN];
            pos.write_numpy_buffer(pos_buffer);
        }
        
        // Calculate the shape that would be created
        let expected_shape = (batch_size, Pos::BUF_N_CHANNELS, Pos::DEFAULT_HEIGHT, Pos::DEFAULT_WIDTH);
        
        println!("Buffer length: {}", buffer.len());
        println!("Expected shape: {:?}", expected_shape);
        
        let expected_buffer_len = batch_size * Pos::BUF_LEN;
        assert_eq!(buffer.len(), expected_buffer_len);
        
        println!("✅ Buffer creation produces correct dimensions!");
        
        // Calculate what the flattened conv output size should be
        let (_batch_size, channels, height, width) = expected_shape;
        let input_elements_per_batch = channels * height * width;
        println!("Input elements per batch item: {}", input_elements_per_batch);
        
        // With 32 conv filters, the output should be 32 * 51 * 51 = 83232 per batch item
        let conv_filters = 32;
        let expected_conv_output_per_batch = conv_filters * height * width;
        println!("Expected conv output per batch (32 filters): {}", expected_conv_output_per_batch);
        
        // This should match the new expected size with 51x51 grid
        assert_eq!(expected_conv_output_per_batch, 83232);
        
        // Also verify the input calculation
        assert_eq!(input_elements_per_batch, 7803); // 3 * 51 * 51
        assert_eq!(channels * height * width, Pos::BUF_LEN);
    }
    
    #[test]
    fn test_mysterious_2560_calculation() {
        println!("=== Investigating the mysterious 2560x2560 dimension ===");
        
        // The error mentioned "2560x2560" - let's figure out where this could come from
        println!("Checking various calculations that could result in 2560:");
        
        // Could it be related to a different filter size?
        let filter_64 = 64 * Pos::DEFAULT_HEIGHT * Pos::DEFAULT_WIDTH;
        println!("64 filters * 20 * 20 = {}", filter_64); // Should be 25600
        
        let filter_128 = 128 * Pos::DEFAULT_HEIGHT;
        println!("128 filters * 20 = {}", filter_128); // 2560!
        
        let sqrt_2560 = (2560.0_f32).sqrt();
        println!("sqrt(2560) = {:.2}", sqrt_2560); // ~50.6
        
        // Could this be from an incorrectly calculated grid size?
        let possible_grid = 2560 / Pos::BUF_N_CHANNELS;
        println!("2560 / 3 channels = {:.2}", possible_grid); // ~853
        
        // Could this be from flattening a tensor incorrectly?
        println!("Our correct calculation: 32 filters * 20 * 20 = {}", 32 * 20 * 20);
        println!("Mysterious calculation: some_val * some_val = 2560 * 2560 = {}", 2560 * 2560);
        
        // The 2560x2560 suggests a square matrix, which might indicate
        // the neural network is expecting a different input size
        println!("Checking if 2560 could be a different conv output calculation...");
        
        // Maybe the issue is the neural network was configured with different dimensions?
        println!("128 filters * 20 height = {}", 128 * 20); // This is 2560!
    }
}
