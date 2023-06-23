use std::io;
use std::io::BufRead;
use crate::board::{Board, CellValue};
use crate::input::Direction;

mod board;
mod input;

fn main() -> io::Result<()> {
    let mut board = Board::new();

    // Spawn a couple tiles
    board.spawn_tile_in_random_location();
    board.spawn_tile_in_random_location();

    // Show the initial state of the board
    println!("{board}");

    let stdin = io::stdin();
    for maybe_next_line_of_input in stdin.lock().lines() {
        if let Err(e) = maybe_next_line_of_input {
            return Err(e);
        }

        let next_line_of_input = maybe_next_line_of_input.unwrap();
        let direction = match Direction::try_from(next_line_of_input.as_ref()) {
            Ok(d) => d,
            Err(_) => {
                println!("Unrecognized input!");
                continue;
            },
        };

        // Only after setting up the `press`/input handling
        board.press(direction);
        board.spawn_tile_in_random_location();
        // Show the new state of the board
        println!("{board}");
    }

    Ok(())
}
