use std::fmt::{Display, Formatter};
use std::iter::Rev;
use std::slice::Iter;

// Only need itertools once we start iterating multiple board rows at a time, for merging
use crate::input::Direction;
use itertools::{Either, Itertools};
use rand::prelude::*;

const BOARD_WIDTH: usize = 4;
const BOARD_HEIGHT: usize = 4;

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct BoardCoordinate(usize, usize);

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum CellContents {
    Empty,
    Occupied(usize),
}

impl CellContents {
    fn as_padded_str(&self) -> String {
        match &self {
            Self::Empty => "    ".to_string(),
            Self::Occupied(value) => format!("{: ^4}", value),
        }
    }

    // Only needed when pushing pieces around
    fn is_empty(&self) -> bool {
        matches!(self, CellContents::Empty)
    }

    fn with_val(v: usize) -> Self {
        Self::Occupied(v)
    }

    fn unwrap(&self) -> usize {
        match self {
            Self::Empty => panic!("Expected a non-empty cell"),
            Self::Occupied(val) => *val,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Cell {
    coords: BoardCoordinate,
    contents: CellContents,
}

impl Cell {
    fn with_coords(coords: BoardCoordinate) -> Self {
        Self {
            coords,
            contents: CellContents::Empty,
        }
    }

    fn new(coords: BoardCoordinate, contents: CellContents) -> Self {
        Self {
            coords,
            contents
        }
    }

    fn is_empty(&self) -> bool {
        matches!(self.contents, CellContents::Empty)
    }
}

#[derive(Debug)]
pub(crate) struct Board {
    pub(crate) cells: [Cell; BOARD_WIDTH * BOARD_HEIGHT],
}

impl Board {
    pub(crate) fn new() -> Self {
        let mut cells = vec![];
        for row_idx in 0..BOARD_HEIGHT {
            for col_idx in 0..BOARD_WIDTH {
                cells.push(Cell::with_coords(BoardCoordinate(col_idx, row_idx)));
            }
        }
        Self {
            cells: cells.try_into().unwrap(),
        }
    }

    pub(crate) fn spawn_tile_in_random_location(&mut self) {
        // Pick a random free cell
        let free_cells = self.cells.iter_mut().filter(|elem| elem.is_empty());
        let chosen_cell = free_cells.choose(&mut thread_rng()).unwrap();
        let value = [2, 4].choose(&mut thread_rng()).unwrap();
        chosen_cell.contents = CellContents::Occupied(*value);
    }

    pub(crate) fn place_cell(&mut self, coordinates: BoardCoordinate, contents: usize) {
        self.cell_with_coords_mut(coordinates).contents = CellContents::Occupied(contents)
    }

    pub(crate) fn cell_with_coords_mut(&mut self, coords: BoardCoordinate) -> &mut Cell {
        &mut self.cells[coords.0 + (coords.1 * BOARD_WIDTH)]
    }

    fn move_cell_into_cell(&mut self, source_cell_idx: usize, dest_cell_idx: usize) {
        self.cells[dest_cell_idx].contents = self.cells[source_cell_idx].contents;
        // And empty the source cell, since it's been moved
        self.cells[source_cell_idx].contents = CellContents::Empty;
    }

    // Only needed during press
    fn cell_indexes_by_row(&self) -> Vec<Vec<usize>> {
        (0..BOARD_WIDTH)
            .map(|col_idx| {
                (0..BOARD_HEIGHT)
                    .map(|row_idx| (row_idx + (col_idx * BOARD_WIDTH)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    fn cell_indexes_by_col(&self) -> Vec<Vec<usize>> {
        (0..BOARD_HEIGHT)
            .map(|row_idx| {
                (0..BOARD_WIDTH)
                    .map(|col_idx| (row_idx + (col_idx * BOARD_WIDTH)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    fn iter_axis_in_direction<'a>(
        direction: Direction,
        cell_indexes_by_col: &'a Vec<Vec<usize>>,
        cell_indexes_by_row: &'a Vec<Vec<usize>>,
    ) -> Either<Iter<'a, Vec<usize>>, Rev<Iter<'a, Vec<usize>>>> {
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
            let row_iter =
                Self::iter_axis_in_direction(direction, &cell_indexes_by_col, &cell_indexes_by_row);
            for (dest_row, source_row) in row_iter.tuple_windows::<(&Vec<usize>, &Vec<usize>)>() {
                for (dest_cell_idx, source_cell_idx) in dest_row.iter().zip(source_row.iter()) {
                    let dest_cell = &self.cells[*dest_cell_idx];
                    let source_cell = &self.cells[*source_cell_idx];
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
        let row_iter =
            Self::iter_axis_in_direction(direction, &cell_indexes_by_col, &cell_indexes_by_row);
        for (dest_row, source_row) in row_iter.tuple_windows::<(&Vec<usize>, &Vec<usize>)>() {
            for (dest_cell_idx, source_cell_idx) in dest_row.iter().zip(source_row.iter()) {
                let dest_cell = &self.cells[*dest_cell_idx];
                let source_cell = &self.cells[*source_cell_idx];
                if source_cell.is_empty() || dest_cell.is_empty() {
                    // If one of the cells is empty, we can't merge them
                    continue;
                }

                let source_value = source_cell.contents.unwrap();
                let dest_value = dest_cell.contents.unwrap();
                if source_value != dest_value {
                    // The cells didn't contain the same value, so we can't merge them
                    continue;
                }

                // Combine into the destination cell
                self.cells[*dest_cell_idx].contents = CellContents::Occupied(dest_value * 2);
                // Clear the contents of the source cell, because it's been merged
                self.cells[*source_cell_idx].contents = CellContents::Empty;
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

    pub(crate) fn is_full(&self) -> bool {
        for cell in self.cells.iter() {
            if cell.contents == CellContents::Empty {
                return false;
            }
        }
        true
    }

    pub(crate) fn empty(&mut self) {
        for cell in self.cells.iter_mut() {
            cell.contents = CellContents::Empty
        }
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Maximum of 4-character width, plus a space on either side for padding
        let cell_width = 4 + 6;
        let cell_width_including_inter_cell_border = cell_width + 1;

        let horizontal_trim = "-".repeat(cell_width_including_inter_cell_border * BOARD_WIDTH);
        write!(f, "\n{}-\n", horizontal_trim)?;

        for cell_indexes_in_row in self.cell_indexes_by_row().iter() {
            // Each tile should occupy a few lines vertically, to bulk out the presentation
            for line_idx in 0..4 {
                let empty_cell_line = format!("|{}", " ".repeat(cell_width));
                match line_idx {
                    1 => {
                        // TODO(PT): This can be filled in
                        for cell_idx in cell_indexes_in_row.iter() {
                            let cell = &self.cells[*cell_idx];
                            let cell_text = cell.contents.as_padded_str();
                            write!(f, "|   {cell_text}   ")?;
                        }
                        write!(f, "|\n")?
                    }
                    3 => write!(f, "{}-\n", horizontal_trim)?,
                    _ => write!(f, "{}|\n", empty_cell_line.repeat(BOARD_WIDTH))?,
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::board::{Board, BoardCoordinate, Cell, CellContents, BOARD_HEIGHT, BOARD_WIDTH};
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
            PushTileToEdgeTestVector {
                input_cells: vec![Cell::new(BoardCoordinate(0, 0), CellContents::with_val(2))],
                direction: Direction::Down,
                expected_output_cells: vec![Cell::new(
                    BoardCoordinate(0, 3),
                    CellContents::Occupied(2),
                )],
            },
            PushTileToEdgeTestVector {
                input_cells: vec![Cell::new(BoardCoordinate(0, 0), CellContents::with_val(2))],
                direction: Direction::Right,
                expected_output_cells: vec![Cell::new(
                    BoardCoordinate(3, 0),
                    CellContents::Occupied(2),
                )],
            },
            PushTileToEdgeTestVector {
                input_cells: vec![Cell::new(BoardCoordinate(3, 0), CellContents::with_val(2))],
                direction: Direction::Left,
                expected_output_cells: vec![Cell::new(
                    BoardCoordinate(0, 0),
                    CellContents::Occupied(2),
                )],
            },
            PushTileToEdgeTestVector {
                input_cells: vec![Cell::new(BoardCoordinate(3, 3), CellContents::with_val(2))],
                direction: Direction::Up,
                expected_output_cells: vec![Cell::new(
                    BoardCoordinate(3, 0),
                    CellContents::Occupied(2),
                )],
            },
            PushTileToEdgeTestVector {
                input_cells: vec![
                    Cell::new(BoardCoordinate(1, 0), CellContents::with_val(2)),
                    Cell::new(BoardCoordinate(2, 0), CellContents::with_val(4)),
                ],
                direction: Direction::Left,
                expected_output_cells: vec![
                    Cell::new(BoardCoordinate(0, 0), CellContents::Occupied(2)),
                    Cell::new(BoardCoordinate(1, 0), CellContents::Occupied(4)),
                ],
            },
        ];
        for vector in input_cells_and_direction_to_expected_output.iter() {
            let mut board = Board::new();
            for cell in vector.input_cells.iter() {
                board.place_cell(cell.coords, cell.contents.unwrap());
            }
            board.press(vector.direction);
            assert_eq!(get_occupied_cells(&board), vector.expected_output_cells);
        }
    }

    #[test]
    fn test_cells_by_axis() {
        // TODO(PT): This setup, with visual output, is a good checkpoint
        let mut board = Board::new();
        for i in 0..BOARD_WIDTH * BOARD_HEIGHT {
            board.cells[i].contents = CellContents::Occupied(i)
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
        let board = Board::new();
        let cell_indexes_by_col = board.cell_indexes_by_col();
        let cell_indexes_by_row = board.cell_indexes_by_row();
        let direction_and_expected_iter = vec![
            (
                Direction::Left,
                vec![
                    vec![0, 4, 8, 12],
                    vec![1, 5, 9, 13],
                    vec![2, 6, 10, 14],
                    vec![3, 7, 11, 15],
                ],
            ),
            (
                Direction::Right,
                vec![
                    vec![3, 7, 11, 15],
                    vec![2, 6, 10, 14],
                    vec![1, 5, 9, 13],
                    vec![0, 4, 8, 12],
                ],
            ),
            (
                Direction::Up,
                vec![
                    vec![0, 1, 2, 3],
                    vec![4, 5, 6, 7],
                    vec![8, 9, 10, 11],
                    vec![12, 13, 14, 15],
                ],
            ),
            (
                Direction::Down,
                vec![
                    vec![12, 13, 14, 15],
                    vec![8, 9, 10, 11],
                    vec![4, 5, 6, 7],
                    vec![0, 1, 2, 3],
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
                )
                .collect::<Vec<_>>(),
                expected_iter_with_ref
            );
        }
    }

    #[test]
    fn test_merge_tiles() {
        let mut board = Board::new();
        board.place_cell(BoardCoordinate(0, 0), 2);
        board.place_cell(BoardCoordinate(2, 0), 2);
        board.place_cell(BoardCoordinate(0, 3), 16);
        board.place_cell(BoardCoordinate(1, 3), 16);
        board.place_cell(BoardCoordinate(2, 3), 16);
        board.place_cell(BoardCoordinate(3, 3), 4);

        board.press(Direction::Left);
        assert_eq!(
            get_occupied_cells(&board),
            vec![
                Cell::new(BoardCoordinate(0, 0), CellContents::Occupied(4)),
                Cell::new(BoardCoordinate(0, 3), CellContents::Occupied(32)),
                Cell::new(BoardCoordinate(1, 3), CellContents::Occupied(16)),
                Cell::new(BoardCoordinate(2, 3), CellContents::Occupied(4)),
            ],
        );
    }
}
