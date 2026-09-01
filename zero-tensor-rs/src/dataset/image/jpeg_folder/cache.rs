use std::{
    collections::{HashMap, VecDeque},
    fs::File,
    path::Path,
};

use memmap2::Mmap;

pub(super) struct MmapCache {
    cache: HashMap<usize, (Mmap, File)>,
    order: VecDeque<usize>,
    max_size: usize,
}

impl MmapCache {
    pub(super) fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            order: VecDeque::new(),
            max_size,
        }
    }

    pub(super) fn get(&mut self, idx: usize, path: &Path) -> Result<&Mmap, std::io::Error> {
        if self.cache.get(&idx).is_some() {
            if let Some(pos) = self.order.iter().position(|&x| x == idx) {
                self.order.remove(pos);
                self.order.push_back(idx);
            }
            Ok(&self.cache.get(&idx).unwrap().0)
        } else {
            let file = File::open(path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            mmap.advise(memmap2::Advice::Sequential)?;

            if self.cache.len() >= self.max_size {
                if let Some(oldest_idx) = self.order.pop_front() {
                    self.cache.remove(&oldest_idx);
                }
            }

            self.cache.insert(idx, (mmap, file));
            self.order.push_back(idx);
            Ok(&self.cache.get(&idx).unwrap().0)
        }
    }
}
