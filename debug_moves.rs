use serde_json;

// Simplified structs for testing
#[derive(Debug, serde::Deserialize)]
struct GameState {
    #[serde(rename = "TimeStamp")]
    pub time_stamp: String,
    #[serde(rename = "Tick")]
    pub tick: u32,
    #[serde(rename = "Cells")]
    pub cells: Vec<Cell>,
    #[serde(rename = "Animals")]
    pub animals: Vec<Animal>,
    #[serde(rename = "Zookeepers")]
    pub zookeepers: Vec<Zookeeper>,
}

#[derive(Debug, serde::Deserialize)]
struct Cell {
    #[serde(rename = "Content")]
    pub content: u8,
}

#[derive(Debug, serde::Deserialize)]
struct Animal {
    pub x: usize,
    pub y: usize,
    pub id: u32,
}

#[derive(Debug, serde::Deserialize)]
struct Zookeeper {
    pub x: usize,
    pub y: usize,
    pub id: u32,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum CellContent {
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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Move {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

fn main() {
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
    println!("Game state: {:?}", game_state);
    
    // Check what's at each position
    println!("Grid (3x3):");
    for y in 0..3 {
        for x in 0..3 {
            let index = y * 3 + x;
            let content = game_state.cells[index].content;
            print!("{} ", content);
        }
        println!();
    }
    
    println!("Player at: ({}, {})", game_state.animals[0].x, game_state.animals[0].y);
    
    // Simulate what happens when moving right
    let player_x = game_state.animals[0].x;
    let player_y = game_state.animals[0].y;
    let new_x = (player_x + 1) % 3;
    let new_y = player_y;
    
    println!("Moving right from ({}, {}) to ({}, {})", player_x, player_y, new_x, new_y);
    
    let new_cell_index = new_y * 3 + new_x;
    let new_cell_content = game_state.cells[new_cell_index].content;
    println!("New cell content: {}", new_cell_content);
    
    if new_cell_content == 1 {
        println!("This is a wall! Move should be blocked.");
    } else {
        println!("This is not a wall. Move should be allowed.");
    }
}
