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
pub struct BoardCoordinate(usize, usize);

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct CellValue(pub(crate) usize);

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Cell {
    Empty,
    Occupied(CellValue),
}

impl Cell {
    fn as_padded_str(&self) -> String {
        match &self {
            Cell::Empty => "    ".to_string(),
            Cell::Occupied(value) => format!("{: ^4}", value.0),
        }
    }

    // Only needed when pushing pieces around
    fn is_empty(&self) -> bool {
        matches!(self, Cell::Empty)
    }

    fn with_val(v: usize) -> Self {
        Self::Occupied(CellValue(v))
    }

    fn unwrap(&self) -> CellValue {
        match self {
            Cell::Empty => panic!("Expected a non-empty cell"),
            Cell::Occupied(val) => *val
        }
    }
}

impl Display for Cell {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[ {} ]", self.as_padded_str())
    }
}

#[derive(Debug)]
pub(crate) struct CellGrid([Cell; BOARD_WIDTH * BOARD_HEIGHT]);

impl CellGrid {
    fn new() -> Self {
        let mut cells = vec![];
        for _row_idx in 0..BOARD_HEIGHT {
            for _col_idx in 0..BOARD_WIDTH {
                cells.push(Cell::Empty);
            }
        }
        Self(cells.try_into().unwrap())
    }

    fn iter(&self) -> CellGridIterator {
        CellGridIterator {
            cells: &self.0,
            index: 0,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Board {
    pub(crate) cells: CellGrid,
}

impl Board {
    pub(crate) fn new() -> Self {
        Self {
            cells: CellGrid::new(),
        }
    }

    pub(crate) fn spawn_tile_in_random_location(&mut self) {
        // Pick a random free cell
        let free_cells = self.cells.0.iter_mut().filter(|elem|{
            **elem == Cell::Empty
        });
        let chosen_cell = free_cells.choose(&mut thread_rng()).unwrap();
        let value = [2, 4].choose(&mut thread_rng()).unwrap();
        *chosen_cell = Cell::Occupied(CellValue(*value));
    }

    pub(crate) fn place_cell(&mut self, coordinates: BoardCoordinate, contents: CellValue) {
        *self.cell_with_coords_mut(coordinates) = Cell::Occupied(contents)
    }

    pub(crate) fn cell_with_coords(&self, coords: BoardCoordinate) -> &Cell {
        //println!("cell_with_coords {coords:?}");
        &self.cells.0[coords.0 + (coords.1 * BOARD_WIDTH)]
    }

    pub(crate) fn cell_with_coords_mut(&mut self, coords: BoardCoordinate) -> &mut Cell {
        //println!("cell_with_coords_mut {coords:?}");
        &mut self.cells.0[coords.0 + (coords.1 * BOARD_WIDTH)]
    }

    // TODO(PT): Replace with cells_by_row_idx
    fn cells_by_row(&self) -> Vec<Vec<&Cell>> {
        let mut cells_by_row = vec![];
        for col_idx in 0..BOARD_WIDTH {
            let mut cells_in_row = vec![];
            for row_idx in 0..BOARD_HEIGHT {
                cells_in_row.push(self.cell_with_coords(BoardCoordinate(row_idx, col_idx)))
            }
            cells_by_row.push(cells_in_row);
        }
        cells_by_row
    }

    fn move_cell_into_cell(&mut self, source_cell_idx: usize, dest_cell_idx: usize) {
        self.cells.0[dest_cell_idx] = self.cells.0[source_cell_idx];
        // And empty the source cell, since it's been moved
        self.cells.0[source_cell_idx] = Cell::Empty;
    }

    // Only needed during press
    fn cell_indexes_by_row(&self) -> Vec<Vec<usize>> {
        let mut cell_indexes_by_row = vec![];
        for col_idx in 0..BOARD_WIDTH {
            let mut cell_indexes_in_row = vec![];
            for row_idx in 0..BOARD_HEIGHT {
                cell_indexes_in_row.push(row_idx + (col_idx * BOARD_WIDTH))
            }
            cell_indexes_by_row.push(cell_indexes_in_row)
        }
        cell_indexes_by_row
    }

    fn cell_indexes_by_col(&self) -> Vec<Vec<usize>> {
        let mut cell_indexes_by_col = vec![];
        for row_idx in 0..BOARD_HEIGHT {
            let mut cell_indexes_in_col = vec![];
            for col_idx in 0..BOARD_WIDTH {
                cell_indexes_in_col.push(row_idx + (col_idx * BOARD_WIDTH))
            }
            cell_indexes_by_col.push(cell_indexes_in_col)
        }
        //println!("Cell indexes by col {:?}", cell_indexes_by_col);
        cell_indexes_by_col
    }

    fn iter_axis_in_direction<'a>(direction: Direction, cell_indexes_by_col: &'a Vec<Vec<usize>>, cell_indexes_by_row: &'a Vec<Vec<usize>>) -> Either<Iter<'a, Vec<usize>>, Rev<Iter<'a, Vec<usize>>>> {
        match direction {
            Direction::Left => Either::Left(cell_indexes_by_col.iter()),
            Direction::Right => Either::Right(cell_indexes_by_col.iter().rev()),
            Direction::Up => Either::Left(cell_indexes_by_row.iter()),
            Direction::Down => Either::Right(cell_indexes_by_row.iter().rev()),
        }
    }

    fn push_cells_to_close_empty_gaps_with_perpendicular_rows(&mut self, direction: Direction) {
        let cell_indexes_by_col = self.cell_indexes_by_col();
        let cell_indexes_by_row = self.cell_indexes_by_row();
        loop {
            let mut did_modify_cells = false;
            let row_iter = Self::iter_axis_in_direction(direction, &cell_indexes_by_col, &cell_indexes_by_row);
            for (dest_row, source_row) in row_iter.tuple_windows::<(&Vec<usize>, &Vec<usize>)>() {
                for (dest_cell_idx, source_cell_idx) in dest_row.iter().zip(source_row.iter()) {
                    let dest_cell = &self.cells.0[*dest_cell_idx];
                    let source_cell = &self.cells.0[*source_cell_idx];
                    if source_cell.is_empty() {
                        // If the source cell is empty, we have nothing to do
                        continue;
                    }
                    if dest_cell.is_empty() {
                        // If the destination cell is empty, copy the source cell
                        self.move_cell_into_cell(*source_cell_idx, *dest_cell_idx);
                        did_modify_cells = true;
                        break;
                    }
                }
            }
            if !did_modify_cells {
                break;
            }
        }
    }

    fn merge_contiguous_cells_in_direction(&mut self, direction: Direction) {
        let cell_indexes_by_col = self.cell_indexes_by_col();
        let cell_indexes_by_row = self.cell_indexes_by_row();
        let row_iter = Self::iter_axis_in_direction(direction, &cell_indexes_by_col, &cell_indexes_by_row);
        for (dest_row, source_row) in row_iter.tuple_windows::<(&Vec<usize>, &Vec<usize>)>() {
            for (dest_cell_idx, source_cell_idx) in dest_row.iter().zip(source_row.iter()) {
                let dest_cell = &self.cells.0[*dest_cell_idx];
                let source_cell = &self.cells.0[*source_cell_idx];
                if source_cell.is_empty() || dest_cell.is_empty() {
                    // If one of the cells is empty, we can't merge them
                    continue;
                }

                let source_value = source_cell.unwrap();
                let dest_value = dest_cell.unwrap();
                if source_value != dest_value {
                    // The cells didn't contain the same value, so we can't merge them
                    continue;
                }

                // Combine into the destination cell
                self.cells.0[*dest_cell_idx] = Cell::Occupied(CellValue(dest_value.0 * 2));
                // Clear the contents of the source cell, because it's been merged
                self.cells.0[*source_cell_idx] = Cell::Empty;
            }
        }
    }

    pub(crate) fn press(&mut self, direction: Direction) {
        // First, push all the elements towards the edge, until they meet resistance
        self.push_cells_to_close_empty_gaps_with_perpendicular_rows(direction);
        // Now iterate again and try to merge contiguous tiles that share the same value
        // We need to do this in a separate iteration because the behavior is subtly different:
        // When pushing cells around, we want to recursively push cells until there's no remaining free
        // space.
        // However, when merging cells, we want to stop processing a row as soon as we merge a pair of cells,
        // even if more merges are possible. The user needs to do another turn to perform the next merge.
        self.merge_contiguous_cells_in_direction(direction);
        // The above step may have produced some gaps, so push cells again
        // For example,
        // | 16 | 16 | 16 |  4 |
        // | 32 |    | 16 |  4 |
        self.push_cells_to_close_empty_gaps_with_perpendicular_rows(direction);
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
                            let cell_text = cell.as_padded_str();
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

impl<'a> IntoIterator for &'a CellGrid {
    type Item = (BoardCoordinate, Cell);
    type IntoIter = CellGridIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        CellGridIterator {
            cells: &self.0,
            index: 0,
        }
    }
}

pub struct CellGridIterator<'a> {
    cells: &'a [Cell; BOARD_WIDTH * BOARD_HEIGHT],
    index: usize,
}

impl<'a> Iterator for CellGridIterator<'a> {
    type Item = (BoardCoordinate, Cell);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.cells.len() {
            return None;
        }

        let row_idx = self.index % BOARD_WIDTH;
        let col_idx = self.index / BOARD_WIDTH;

        let out = Some(
            (
                BoardCoordinate(row_idx, col_idx),
                self.cells[self.index]
            )
        );
        self.index += 1;
        out
    }
}

#[cfg(test)]
mod test {
    use crate::board::{Board, BOARD_HEIGHT, BOARD_WIDTH, BoardCoordinate, Cell, CellGridIterator, CellValue};
    use crate::input::Direction;

    fn get_occupied_cells(board: &Board) -> Vec<(BoardCoordinate, Cell)> {
        let mut out = vec![];
        for (coord, cell) in board.cells.iter() {
            //println!("coord {coord:?}");
            if !cell.is_empty() {
                out.push((coord, cell));
            }
        }
        out
    }

    struct PushTileToEdgeTestVector {
        input_cells: Vec<(BoardCoordinate, Cell)>,
        direction: Direction,
        expected_output_cells: Vec<(BoardCoordinate, Cell)>,
    }

    #[test]
    fn push_tile_to_edge() {
        let input_cells_and_direction_to_expected_output = vec![
            PushTileToEdgeTestVector {
                input_cells: vec![
                    (BoardCoordinate(0, 0), Cell::with_val(2)),
                ],
                direction: Direction::Down,
                expected_output_cells: vec![
                    (BoardCoordinate(0, 3), Cell::Occupied(CellValue(2))),
                ],
            },
            /*
            PushTileToEdgeTestVector {
                input_cells: vec![Cell::new(BoardCoordinate(0, 0), Cell::Occupied(CellValue(2)))],
                direction: Direction::Down,
                expected_output_cells: vec![
                    Cell::new(BoardCoordinate(0, 3), Cell::Occupied(CellValue(2)))
                ],
            },
            PushTileToEdgeTestVector {
                input_cells: vec![Cell::new(BoardCoordinate(0, 0), Cell::Occupied(CellValue(2)))],
                direction: Direction::Right,
                expected_output_cells: vec![
                    Cell::new(BoardCoordinate(3, 0), Cell::Occupied(CellValue(2)))
                ],
            },
            PushTileToEdgeTestVector {
                input_cells: vec![Cell::new(BoardCoordinate(0, 3), Cell::Occupied(CellValue(2)))],
                direction: Direction::Left,
                expected_output_cells: vec![
                    Cell::new(BoardCoordinate(0, 0), Cell::Occupied(CellValue(2)))
                ],
            },
            PushTileToEdgeTestVector {
                input_cells: vec![Cell::new(BoardCoordinate(3, 3), Cell::Occupied(CellValue(2)))],
                direction: Direction::Up,
                expected_output_cells: vec![
                    Cell::new(BoardCoordinate(3, 0), Cell::Occupied(CellValue(2)))
                ],
            },
            */
            /*
            PushTileToEdgeTestVector {
                input_cells: vec![
                    (BoardCoordinate(1, 0), Cell::with_val(2)),
                    (BoardCoordinate(2, 0), Cell::with_val(4)),
                ],
                direction: Direction::Left,
                expected_output_cells: vec![
                    (BoardCoordinate(0, 0), Cell::Occupied(CellValue(2))),
                    (BoardCoordinate(1, 0), Cell::Occupied(CellValue(4)))
                ],
            },

             */
        ];
        for vector in input_cells_and_direction_to_expected_output.iter() {
            //println!("doing vector");
            let mut board = Board::new();
            for (cell_coords, input_cell) in vector.input_cells.iter() {
                match input_cell {
                    Cell::Empty => {}
                    Cell::Occupied(val) => {
                        board.place_cell(*cell_coords, *val);
                    }
                }
            }
            //println!("board {board}");
            for (i, c) in board.cells.0.iter().enumerate() {
                //println!("{i}: {c}");
            }
            board.press(vector.direction);
            //println!("board {board}");
            assert_eq!(
                get_occupied_cells(&board),
                vector.expected_output_cells
            );
        }
    }

    #[test]
    fn test_cells_by_axis() {
        // TODO(PT): This, with visual output, is a good checkpoint
        let mut board = Board::new();
        for i in 0..BOARD_WIDTH * BOARD_HEIGHT {
            board.cells.0[i] = Cell::Occupied(CellValue(i))
        }

        assert_eq!(
            board.cell_indexes_by_row(),
            vec![
                vec![0, 1, 2, 3],
                vec![4, 5, 6, 7],
                vec![8, 9, 10, 11],
                vec![12, 13, 14, 15],
            ],
        );

        assert_eq!(
            board.cell_indexes_by_col(),
            vec![
                vec![0, 4, 8, 12],
                vec![1, 5, 9, 13],
                vec![2, 6, 10, 14],
                vec![3, 7, 11, 15],
            ],
        );
    }

    #[test]
    fn test_iter_axis_in_direction() {
        let mut board = Board::new();
        let cell_indexes_by_col = board.cell_indexes_by_col();
        let cell_indexes_by_row = board.cell_indexes_by_row();
        let direction_and_expected_iter = vec![
            (
                Direction::Left,
                vec![
                    vec![0, 4, 8, 12], vec![1, 5, 9, 13],
                    vec![2, 6, 10, 14], vec![3, 7, 11, 15],
                ],
            ),
            (
                Direction::Right,
                vec![
                    vec![3, 7, 11, 15],  vec![2, 6, 10, 14],
                    vec![1, 5, 9, 13], vec![0, 4, 8, 12]
                ],
            ),
            (
                Direction::Up,
                vec![
                    vec![0, 1, 2, 3], vec![4, 5, 6, 7],
                    vec![8, 9, 10, 11], vec![12, 13, 14, 15],
                ],
            ),
            (
                Direction::Down,
                vec![
                    vec![12, 13, 14, 15], vec![8, 9, 10, 11],
                    vec![4, 5, 6, 7], vec![0, 1, 2, 3],
                ],
            ),
        ];
        for (direction, expected_iter) in direction_and_expected_iter.iter() {
            let expected_iter_with_ref = expected_iter.iter().map(|v| v).collect::<Vec<_>>();
            assert_eq!(
                Board::iter_axis_in_direction(
                    *direction,
                    &cell_indexes_by_col,
                    &cell_indexes_by_row
                ).collect::<Vec<_>>(),
                expected_iter_with_ref
            );
        }
    }

    #[test]
    fn test_merge_tiles() {
        let mut board = Board::new();
        board.place_cell(BoardCoordinate(0, 0), CellValue(2));
        board.place_cell(BoardCoordinate(2, 0), CellValue(2));
        board.place_cell(BoardCoordinate(0, 3), CellValue(16));
        board.place_cell(BoardCoordinate(1, 3), CellValue(16));
        board.place_cell(BoardCoordinate(2, 3), CellValue(16));
        board.place_cell(BoardCoordinate(3, 3), CellValue(4));

        board.press(Direction::Left);
        assert_eq!(
            get_occupied_cells(&board),
            vec![
                (BoardCoordinate(0, 0), Cell::Occupied(CellValue(4))),
                (BoardCoordinate(0, 3), Cell::Occupied(CellValue(32))),
                (BoardCoordinate(1, 3), Cell::Occupied(CellValue(16))),
                (BoardCoordinate(2, 3), Cell::Occupied(CellValue(4))),
            ],
        );
    }
}
