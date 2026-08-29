pub mod error;

use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::decoder::{ImageInfo, default::JpegDecoder};

pub struct JpegFolderDataset {
    samples: Vec<(PathBuf, i64)>,
    mmaps: Vec<Mmap>,
    infos: Vec<ImageInfo>,
    decoder: JpegDecoder
}

impl JpegFolderDataset {
    pub fn new<F>(path: &Path, label_fn: F)
    where 
    F: Fn(&Path) -> Option<i64> {
        if !path.is_dir() {
            todo!()
        }
        let mut samples: Vec<(PathBuf, i64)> = Vec::new();
        for entry in walkdir::WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_file() {
                if let Some(extos) = path.extension() {
                    let ext = extos.to_string_lossy();
                    if ext == "jpg" || ext == "jpeg" {
                        if let Some(label) = label_fn(path) {
                            samples.push((path.into(), label));
                        }
                    }
                }
            }
        }

        if samples.is_empty() {
            todo!()
        }
    }
}