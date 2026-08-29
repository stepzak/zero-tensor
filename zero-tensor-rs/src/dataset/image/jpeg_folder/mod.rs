pub mod error;
pub use error::JpegFolderDatasetNewError;

use std::{
    fs::File,
    path::{Path, PathBuf},
};

use memmap2::Mmap;

use crate::decoder::{ImageDecoder, ImageInfo, default::JpegDecoder};

pub struct JpegFolderDataset {
    samples: Vec<(PathBuf, i64)>,
    mmaps: Vec<Mmap>,
    infos: Vec<ImageInfo>,
    decoder: JpegDecoder,
}

impl JpegFolderDataset {
    pub fn new<F>(
        root: &Path,
        label_fn: F,
    ) -> Result<Self, JpegFolderDatasetNewError<turbojpeg::Error>>
    where
        F: Fn(&Path) -> Option<i64>,
    {
        if !root.is_dir() {
            return Err(JpegFolderDatasetNewError::NotDirectory(root.into()));
        }
        let mut samples: Vec<(PathBuf, i64)> = Vec::new();
        for entry in walkdir::WalkDir::new(root)
            .into_iter()
            .filter_map(|e| e.ok())
        {
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
            return Err(JpegFolderDatasetNewError::Empty(root.into()));
        }

        samples.sort_by(|a, b| a.0.cmp(&b.0));

        let mut mmaps = Vec::with_capacity(samples.len());
        let mut infos = Vec::with_capacity(samples.len());
        let decoder = JpegDecoder::new();

        for (path, _) in samples.iter() {
            let file = File::open(path)?;
            let mmap = unsafe { Mmap::map(&file) }?;

            let info = decoder
                .info(&mmap)
                .map_err(|e| JpegFolderDatasetNewError::DecodeError(path.into(), e))?;
            mmaps.push(mmap);
            infos.push(info);
        }

        Ok(Self {
            samples,
            mmaps,
            infos,
            decoder,
        })
    }
}
