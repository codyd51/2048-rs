use std::fmt::{Display, Formatter};
use std::iter::Rev;
use std::slice::Iter;

// Only need itertools once we start iterating multiple board rows at a time, for merging
use itertools::{Either, Itertools};
use rand::prelude::*;
use crate::input::Direction;

const BOARD_WIDTH: usize = 4;
const BOARD_HEIGHT: usize = 4;

#[derive(Debug, PartialEq, Copy, Clone)]
pub(crate) struct BoardCoordinate(usize, usize);

#[derive(Debug, PartialEq, Copy, Clone)]
pub(crate) struct CellValue(pub(crate) usize);

#[derive(Debug, PartialEq, Copy, Clone)]
pub(crate) enum CellContents {
    Empty,
    Occupied(CellValue),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Cell {
    coords: BoardCoordinate,
    pub(crate) contents: CellContents,
}

impl Cell {
    fn new(coords: BoardCoordinate, contents: CellContents) -> Self {
        Self {
            coords,
            contents,
        }
    }

    fn contents_as_padded_str(&self) -> String {
        match &self.contents {
            CellContents::Empty => "    ".to_string(),
            CellContents::Occupied(value) => format!("{: ^4}", value.0),
        }
    }

    // Only needed when pushing pieces around
    fn is_empty(&self) -> bool {
        matches!(self.contents, CellContents::Empty)
    }
}

impl Display for Cell {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[ {} ]", self.contents_as_padded_str())
    }
}

#[derive(Debug)]
pub(crate) struct Board {
    pub(crate) cells: [Cell; BOARD_WIDTH * BOARD_HEIGHT],
}

impl Board {
    pub(crate) fn new() -> Self {
        let mut cells = vec![];
        for col_idx in 0..BOARD_WIDTH {
            for row_idx in 0..BOARD_HEIGHT {
                cells.push(Cell::new(BoardCoordinate(row_idx, col_idx), CellContents::Empty));
            }
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
        let chosen_cell = free_cells.choose(&mut thread_rng()).unwrap();
        let value = [2, 4].choose(&mut thread_rng()).unwrap();
        chosen_cell.contents = CellContents::Occupied(CellValue(*value));
    }

    pub(crate) fn place_cell(&mut self, coordinates: BoardCoordinate, contents: CellValue) {
        self.cell_with_coords_mut(coordinates).contents = CellContents::Occupied(contents)
    }

    pub(crate) fn cell_with_coords(&self, coords: BoardCoordinate) -> &Cell {
        &self.cells[(coords.0 * BOARD_WIDTH) + coords.1]
    }

    pub(crate) fn cell_with_coords_mut(&mut self, coords: BoardCoordinate) -> &mut Cell {
        &mut self.cells[(coords.0 * BOARD_WIDTH) + coords.1]
    }

    // TODO(PT): Replace with cells_by_row_idx
    fn cells_by_row(&self) -> Vec<Vec<&Cell>> {
        let mut cells_by_row = vec![];
        for row_idx in 0..BOARD_HEIGHT {
            let mut cells_in_row = vec![];
            for col_idx in 0..BOARD_WIDTH {
                cells_in_row.push(self.cell_with_coords(BoardCoordinate(row_idx, col_idx)))
            }
            cells_by_row.push(cells_in_row);
        }
        cells_by_row
    }

    fn move_cell_into_cell(&mut self, source_cell_idx: usize, dest_cell_idx: usize) {
        self.cells[dest_cell_idx].contents = self.cells[source_cell_idx].contents;
        // And empty the source cell, since it's been moved
        self.cells[source_cell_idx].contents = CellContents::Empty;
    }

    // Only needed during press
    fn cell_indexes_by_row(&self) -> Vec<Vec<usize>> {
        let mut cell_indexes_by_row = vec![];
        for row_idx in 0..BOARD_HEIGHT {
            let mut cell_indexes_in_row = vec![];
            for col_idx in 0..BOARD_WIDTH {
                cell_indexes_in_row.push((row_idx * BOARD_WIDTH) + col_idx)
            }
            cell_indexes_by_row.push(cell_indexes_in_row)
        }
        cell_indexes_by_row
    }

    fn cell_indexes_by_col(&self) -> Vec<Vec<usize>> {
        let mut cell_indexes_by_col = vec![];
        for col_idx in 0..BOARD_WIDTH {
            let mut cell_indexes_in_col = vec![];
            for row_idx in 0..BOARD_HEIGHT {
                cell_indexes_in_col.push((row_idx * BOARD_WIDTH) + col_idx)
            }
            cell_indexes_by_col.push(cell_indexes_in_col)
        }
        cell_indexes_by_col
    }

    fn iter_axis_in_direction<'a>(direction: Direction, cell_indexes_by_col: &'a Vec<Vec<usize>>, cell_indexes_by_row: &'a Vec<Vec<usize>>) -> Either<Rev<Iter<'a, Vec<usize>>>, Iter<'a, Vec<usize>>> {
        match direction {
            Direction::Left => Either::Left(cell_indexes_by_col.iter().rev()),
            Direction::Right => Either::Right(cell_indexes_by_col.iter()),
            Direction::Up => Either::Left(cell_indexes_by_row.iter().rev()),
            Direction::Down => Either::Right(cell_indexes_by_row.iter()),
        }
    }

    fn push_cells_to_close_empty_gaps_with_perpendicular_rows(&mut self, direction: Direction) {
        let cell_indexes_by_col = self.cell_indexes_by_col();
        let cell_indexes_by_row = self.cell_indexes_by_row();
        println!("direction {direction:?}");
        let row_iter = Self::iter_axis_in_direction(direction, &cell_indexes_by_col, &cell_indexes_by_row);
        for (higher_row, lower_row) in row_iter.tuple_windows::<(&Vec<usize>, &Vec<usize>)>() {
            println!("\tl={lower_row:?}, h={higher_row:?}");
            for (dest_cell_idx, source_cell_idx) in lower_row.iter().zip(higher_row.iter()) {
                println!("\t\t{dest_cell_idx}, {source_cell_idx}");
                let dest_cell = &self.cells[*dest_cell_idx];
                let source_cell = &self.cells[*source_cell_idx];
                if source_cell.is_empty() {
                    continue;
                }
                if dest_cell.is_empty() {
                    println!("\t\t\tmove {source_cell:?} to {dest_cell:?}");
                    self.move_cell_into_cell(*source_cell_idx, *dest_cell_idx);
                }
            }
        }
    }

    fn merge_contiguous_cells_in_direction(&mut self, direction: Direction) {
        let cell_indexes_by_col = self.cell_indexes_by_col();
        let cell_indexes_by_row = self.cell_indexes_by_row();
        let row_iter = Self::iter_axis_in_direction(direction, &cell_indexes_by_col, &cell_indexes_by_row);
        for (higher_row, lower_row) in row_iter.tuple_windows::<(&Vec<usize>, &Vec<usize>)>() {}
    }

    pub(crate) fn press(&mut self, direction: Direction) {
        // First, push all the elements towards the edge, until they meet resistance
        self.push_cells_to_close_empty_gaps_with_perpendicular_rows(direction);
        // Now iterate again and try to merge contiguous tiles that share the same value
        //self.merge_contiguous_cells_in_direction(direction);

        // Starting from the bottom row, merge tiles downwards
        /*
        for (lower_row, higher_row) in cell_indexes_by_row.iter().rev().tuple_windows::<(&Vec<usize>, &Vec<usize>)>() {
            for (dest_cell_idx, source_cell_idx) in lower_row.iter().zip(higher_row.iter()) {
                let dest_cell = &self.cells[*dest_cell_idx];
                let source_cell = &self.cells[*source_cell_idx];
                match &source_cell.contents {
                    CellContents::Empty => {
                        // If the source cell is empty, we have nothing to do
                        continue;
                    }
                    CellContents::Occupied(source_value) => {
                        match &dest_cell.contents {
                            CellContents::Empty => {
                                // If the destination cell is empty, copy the source cell
                                //dest_cell.contents = source_cell.contents;
                                self.cells[*dest_cell_idx].contents = source_cell.contents;
                            }
                            CellContents::Occupied(dest_value) => {
                                // Check whether we can combine the source and dest
                                if source_value == dest_value {
                                    // Combine into the destination cell
                                    self.cells[*dest_cell_idx].contents = CellContents::Occupied(CellValue(dest_value.0 * 2));
                                    // Clear the contents of the source cell, because it's been pushed
                                    self.cells[*source_cell_idx].contents = CellContents::Empty;
                                }
                            }
                        }
                    }
                }
            }
        }
         */
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Maximum of 4-character width, plus a space on either side for padding
        let cell_width = 4 + 6;
        let cell_width_including_inter_cell_border = cell_width + 1;

        let horizontal_trim = "-".repeat(cell_width_including_inter_cell_border * BOARD_WIDTH);
        write!(f, "\n{}-\n", horizontal_trim)?;

        for row in self.cells_by_row().iter() {
            // Each tile should occupy a few lines vertically, to bulk out the presentation
            for line_idx in 0..4 {
                let empty_cell_line = format!("|{}", " ".repeat(cell_width));
                match line_idx {
                    1 => {
                        // TODO(PT): This can be filled in
                        for cell in row.iter() {
                            let cell_text = cell.contents_as_padded_str();
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

#[cfg(test)]
mod test {
    use crate::board::{Board, BoardCoordinate, Cell, CellContents, CellValue};
    use crate::input::Direction;

    fn get_occupied_cells(board: &Board) -> Vec<Cell> {
        let mut out = vec![];
        for cell in board.cells.iter() {
            if !cell.is_empty() {
                out.push(cell.clone());
            }
        }
        out
    }

    struct PushTileToEdgeTestVector {
        input_cells: Vec<Cell>,
        direction: Direction,
        expected_output_cells: Vec<Cell>,
    }

    #[test]
    fn push_tile_to_edge() {
        let input_cells_and_direction_to_expected_output = vec![
            /*
            PushTileToEdgeTestVector {
                input_cells: vec![Cell::new(BoardCoordinate(0, 0), CellContents::Occupied(CellValue(2)))],
                direction: Direction::Down,
                expected_output_cells: vec![
                    Cell::new(BoardCoordinate(0, 3), CellContents::Occupied(CellValue(2)))
                ],
            },
            PushTileToEdgeTestVector {
                input_cells: vec![Cell::new(BoardCoordinate(0, 0), CellContents::Occupied(CellValue(2)))],
                direction: Direction::Right,
                expected_output_cells: vec![
                    Cell::new(BoardCoordinate(3, 0), CellContents::Occupied(CellValue(2)))
                ],
            },
            PushTileToEdgeTestVector {
                input_cells: vec![Cell::new(BoardCoordinate(0, 3), CellContents::Occupied(CellValue(2)))],
                direction: Direction::Left,
                expected_output_cells: vec![
                    Cell::new(BoardCoordinate(0, 0), CellContents::Occupied(CellValue(2)))
                ],
            },
            PushTileToEdgeTestVector {
                input_cells: vec![Cell::new(BoardCoordinate(3, 3), CellContents::Occupied(CellValue(2)))],
                direction: Direction::Up,
                expected_output_cells: vec![
                    Cell::new(BoardCoordinate(3, 0), CellContents::Occupied(CellValue(2)))
                ],
            },
            */
            PushTileToEdgeTestVector {
                input_cells: vec![
                    Cell::new(BoardCoordinate(1, 0), CellContents::Occupied(CellValue(2))),
                    Cell::new(BoardCoordinate(2, 0), CellContents::Occupied(CellValue(2))),
                ],
                direction: Direction::Left,
                expected_output_cells: vec![
                    Cell::new(BoardCoordinate(0, 0), CellContents::Occupied(CellValue(2))),
                    Cell::new(BoardCoordinate(1, 0), CellContents::Occupied(CellValue(2)))
                ],
            },
        ];
        for vector in input_cells_and_direction_to_expected_output.iter() {
            println!("doing vector");
            let mut board = Board::new();
            for input_cell in vector.input_cells.iter() {
                match input_cell.contents {
                    CellContents::Empty => {}
                    CellContents::Occupied(val) => {
                        board.place_cell(input_cell.coords, val);
                    }
                }
            }
            board.press(vector.direction);
            assert_eq!(
                get_occupied_cells(&board),
                vector.expected_output_cells
            );
        }
    }
}
