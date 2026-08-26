use indexmap::IndexMap;

type Offset = usize;
type Size = usize;

pub type LinStorage<'a> = smallvec::SmallVec<[(&'a str, Offset, Size); 8]>;
pub type WrittenFlags = smallvec::SmallVec<[bool; 8]>;

pub enum TensorWriterStorage<'a> {
    Lin(LinStorage<'a>),
    Map(IndexMap<&'a str, (Offset, Size)>),
}

impl<'a> TensorWriterStorage<'a> {
    #[inline]
    pub fn get_offset_size(&self, key: &str) -> Option<(Offset, Size)> {
        match self {
            TensorWriterStorage::Lin(v) => v
                .iter()
                .find(|&&x| x.0 == key)
                .map(|&(_, offs, s)| (offs, s)),
            TensorWriterStorage::Map(m) => m.get(key).copied(),
        }
    }

    #[inline]
    pub fn insert(&mut self, key: &'a str, offset: Offset, size: Size) -> usize {
        match self {
            TensorWriterStorage::Lin(v) => {
                let idx = v.len();
                v.push((key, offset, size));
                idx
            }
            TensorWriterStorage::Map(m) => {
                let (idx, _) = m.insert_full(key, (offset, size));
                idx
            }
        }
    }

    #[inline]
    pub fn contains(&self, key: &str) -> bool {
        match self {
            TensorWriterStorage::Lin(v) => v.iter().any(|&x| x.0 == key),
            TensorWriterStorage::Map(m) => m.contains_key(key),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            TensorWriterStorage::Lin(v) => v.len(),
            TensorWriterStorage::Map(m) => m.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn clear(&mut self) {
        match self {
            TensorWriterStorage::Lin(v) => v.clear(),
            TensorWriterStorage::Map(m) => m.clear(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        const THRESHOLD: usize = 16;
        if capacity > THRESHOLD {
            TensorWriterStorage::Map(IndexMap::with_capacity(capacity))
        } else {
            TensorWriterStorage::Lin(LinStorage::with_capacity(capacity))
        }
    }

    #[inline]
    pub fn get_key_pos(&self, key: &str) -> Option<usize> {
        match self {
            Self::Lin(v) => v.iter().position(|&x| x.0 == key),
            Self::Map(m) => m.get_index_of(key),
        }
    }

    pub fn keys(&self) -> Vec<&'a str> {
        match self {
            TensorWriterStorage::Lin(v) => v.iter().map(|&(k, _, _)| k).collect(),
            TensorWriterStorage::Map(m) => m.keys().copied().collect(),
        }
    }
}

pub struct TensorWriterCache<'a> {
    slot_buffers: TensorWriterStorage<'a>,
    written: WrittenFlags,
}

impl<'a> TensorWriterCache<'a> {
    pub fn with_capacity(tensors_per_sample: usize) -> Self {
        let slot_buffers = TensorWriterStorage::with_capacity(tensors_per_sample);
        let written = WrittenFlags::with_capacity(tensors_per_sample);
        Self {
            slot_buffers,
            written,
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.slot_buffers.clear();
        self.written.clear();
    }

    #[inline]
    pub fn insert(&mut self, key: &'a str, offset: usize, size: usize) {
        self.slot_buffers.insert(key, offset, size);
        self.written.push(false);
    }

    #[inline]
    pub fn slot_buffers_mut(&mut self) -> &mut TensorWriterStorage<'a> {
        &mut self.slot_buffers
    }

    #[inline]
    pub fn slot_buffers(&self) -> &TensorWriterStorage<'a> {
        &self.slot_buffers
    }

    #[inline]
    pub fn written_mut(&mut self) -> &mut WrittenFlags {
        &mut self.written
    }

    #[inline]
    pub fn written(&self) -> &[bool] {
        &self.written
    }

    #[inline]
    pub fn get_offset_size(&self, key: &str) -> Option<(Offset, Size)> {
        self.slot_buffers.get_offset_size(key)
    }

    #[inline]
    pub fn contains(&self, key: &str) -> bool {
        self.slot_buffers.contains(key)
    }

    pub fn mark_written(&mut self, key: &str) -> bool {
        if let Some(i) = self.slot_buffers.get_key_pos(key) {
            if let Some(w) = self.written.get_mut(i) {
                *w = true;
                return true;
            }
        }
        false
    }

    #[inline]
    pub fn is_fully_written(&self) -> bool {
        self.written.iter().all(|&w| w)
    }

    pub fn keys(&self) -> Vec<&'a str> {
        self.slot_buffers.keys()
    }

    pub fn get_missing_keys(&self) -> Vec<String> {
        self.slot_buffers
            .keys()
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !self.written[*i])
            .map(|(_, k)| k.to_string())
            .collect()
    }
}
