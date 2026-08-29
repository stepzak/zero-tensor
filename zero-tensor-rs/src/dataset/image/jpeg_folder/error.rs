use std::path::PathBuf;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum JpegFolderDatasetNewError {
    #[error("Not a dir: {0}")]
    NotDirectory(PathBuf),
    
    #[error("No JPEG files found in {0}")]
    Empty(PathBuf)
}