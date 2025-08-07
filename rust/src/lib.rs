#![allow(dead_code)]
mod zootopia;
mod interactive_play;
mod mcts;
mod pybridge;
mod self_play;
mod solver;
mod tui;
mod types;
mod utils;

use zootopia::Pos;
use env_logger::Env;
use pybridge::PlayGamesResult;
use pyo3::prelude::*;
use types::{GameMetadata, GameResult, Sample};

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn zsa0_rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    m.add("N_COLS", Pos::N_COLS)?;
    m.add("N_ROWS", Pos::N_ROWS)?;
    m.add("BUF_N_CHANNELS", Pos::BUF_N_CHANNELS)?;

    m.add_class::<GameMetadata>()?;
    m.add_class::<GameResult>()?;
    m.add_class::<Sample>()?;
    m.add_class::<PlayGamesResult>()?;

    m.add_function(wrap_pyfunction!(pybridge::play_games, m)?)?;
    m.add_function(wrap_pyfunction!(pybridge::run_tui, m)?)?;

    Ok(())
}

pub mod debug_tests;
pub mod softmax_debug;
