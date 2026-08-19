#[derive(Debug)]
pub enum TransformError {
    UnsupportedDtype,
    Overflow,
    InvalidValue,
}
