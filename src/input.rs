
#[derive(Debug, Copy, Clone)]
pub(crate) enum Direction {
    Left,
    Right,
    Up,
    Down
}

impl TryFrom<&str> for Direction {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "h" => Ok(Direction::Left),
            "j" => Ok(Direction::Down),
            "k" => Ok(Direction::Up),
            "l" => Ok(Direction::Right),
            // Unhandled input
            _ => Err(()),
        }
    }
}