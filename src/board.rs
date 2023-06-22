use std::fmt::{Display, Formatter};

use rand::prelude::*;

const BOARD_WIDTH: usize = 4;
const BOARD_HEIGHT: usize = 4;

#[derive(Debug, Copy, Clone)]
pub(crate) struct BoardCoordinate(usize, usize);

#[derive(Debug, PartialEq)]
pub(crate) struct CellValue(pub(crate) usize);

#[derive(Debug, PartialEq)]
pub(crate) enum CellContents {
    Empty,
    Occupied(CellValue),
}

#[derive(Debug)]
pub(crate) struct Cell {
    coords: BoardCoordinate,
    pub(crate) contents: CellContents,
}

impl Cell {
    fn new(coords: BoardCoordinate) -> Self {
        Self {
            coords,
            contents: CellContents::Empty,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Board {
    pub(crate) cells: [Cell; BOARD_WIDTH * BOARD_HEIGHT],
}

impl Board {
    pub(crate) fn new() -> Self {
        let mut cells = vec![];
        for i in 0..(BOARD_WIDTH * BOARD_HEIGHT) {
            let row_idx = i / BOARD_WIDTH;
            let col_idx = i % BOARD_WIDTH;
            cells.push(Cell::new(BoardCoordinate(row_idx, col_idx)))
        }
        Self {
            cells: cells.try_into().unwrap()
        }
    }

    pub(crate) fn spawn_tile_in_random_location(&mut self) {
        // Pick a random free cell
        let free_cells = self.cells.iter_mut().filter(|elem|{
            elem.contents == CellContents::Empty
        });
        let chosen_cell = free_cells.choose(&mut rand::thread_rng()).unwrap();
        let value = [2, 4].choose(&mut rand::thread_rng()).unwrap();
        chosen_cell.contents = CellContents::Occupied(CellValue(*value));
    }

    pub(crate) fn cell_with_coords(&self, coords: BoardCoordinate) -> &Cell {
        &self.cells[(coords.0 * BOARD_WIDTH) + coords.1]
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Maximum of 4-character width, plus a space on either side for padding
        let cell_width = 4 + 6;
        let cell_width_including_inter_cell_border = cell_width + 1;

        let horizontal_trim = "-".repeat(cell_width_including_inter_cell_border * BOARD_WIDTH);
        write!(f, "\n{}-\n", horizontal_trim)?;
        let mut cells_by_row = vec![];
        for row_idx in 0..BOARD_HEIGHT {
            let mut cells_in_row = vec![];
            for col_idx in 0..BOARD_WIDTH {
                cells_in_row.push(self.cell_with_coords(BoardCoordinate(row_idx, col_idx)))
            }
            cells_by_row.push(cells_in_row);
        }

        for row in cells_by_row.iter() {
            // Each tile should occupy a few lines vertically, to bulk out the presentation
            for line_idx in 0..4 {
                let empty_cell_line = format!("|{}", " ".repeat(cell_width));
                match line_idx {
                    1 => {
                        // TODO(PT): This can be filled in
                        for cell in row.iter() {
                            let cell_text = match &cell.contents {
                                CellContents::Empty => "    ".to_string(),
                                CellContents::Occupied(value) => format!("{: ^4}", value.0),
                            };
                            write!(f, "|   {cell_text}   ")?;
                        }
                        write!(f, "|\n")?
                    }
                    3 => write!(f, "{}-\n", horizontal_trim)?,
                    _ => write!(f, "{}|\n", empty_cell_line.repeat(BOARD_WIDTH))?
                }
            }
        }

        Ok(())
    }
}
