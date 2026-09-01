use crate::core::dataset::ZTDatasetError;

use super::*;

#[derive(Clone)]
pub struct ZeroTensorProducerBuilder {
    // Required
    pub slot_size: u64,
    pub shm_filename: String,
    pub socket_addr: PathBuf,

    // Optional
    pub num_slots: u64,
    pub read_timeout: Option<u64>,
    pub overwrite_socket: bool,
    pub shuffle: bool,
    pub seed: Option<u64>,
    pub max_steps: Option<usize>,
}

impl ZeroTensorProducerBuilder {
    pub fn new<P: AsRef<Path>>(slot_size: u64, shm_filename: &str, socket_addr: P) -> Self {
        let mut s_size = slot_size;
        let rec_slot_size_align = ZeroTensorControlBlock::recommended_slot_alignment() as u64;
        let min_slot_size_align = ZeroTensorControlBlock::min_slot_alignment() as u64;

        if !slot_size.is_multiple_of(min_slot_size_align) {
            //TODO: warning
            s_size = Self::round_slot_size(slot_size, rec_slot_size_align);
        }

        Self {
            slot_size: s_size,
            shm_filename: shm_filename.to_string(),
            socket_addr: socket_addr.as_ref().to_path_buf(),
            num_slots: DEFAULT_SLOTS,
            read_timeout: None,
            overwrite_socket: false,
            shuffle: false,
            seed: None,
            max_steps: None,
        }
    }

    fn round_slot_size(slot_size: u64, rec: u64) -> u64 {
        slot_size.div_ceil(rec) * rec
    }

    pub fn num_slots(mut self, slots: u64) -> Self {
        self.num_slots = slots;
        self
    }

    pub fn read_timeout(mut self, timeout_ms: u64) -> Self {
        self.read_timeout = Some(timeout_ms);
        self
    }

    pub fn overwrite_socket(mut self, overwrite: bool) -> Self {
        self.overwrite_socket = overwrite;
        self
    }

    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn max_steps<M: Into<Option<usize>>>(mut self, max_steps: M) -> Self {
        self.max_steps = max_steps.into();
        self
    }

    pub fn from_dataset<'a, D: ZeroTensorDataset<'a>, P: AsRef<Path>, S: Into<Option<usize>>>(
        dataset: &D,
        shm_filename: &str,
        socket_addr: P,
        batch_size: usize,
        probe_size: S,
    ) -> Result<Self, ZTProducerErr<D::Error>> {
        let slot_size = if let Some(static_layouts) = dataset.static_layouts() {
            ZeroTensorBuffer::calculate_slot_size(static_layouts, batch_size)
        } else {
            if dataset.len() == 0 {
                return Err(ZTProducerErr::EmptyDataset);
            }

            let actual_probe_size = probe_size.into().unwrap_or(dataset.len());
            let actual_probe_size = actual_probe_size.min(dataset.len());

            let idxs = if actual_probe_size < dataset.len() {
                use rand::seq::SliceRandom;
                let mut all_idxs: Vec<usize> = (0..dataset.len()).collect();
                let mut rng = rand::rng();
                all_idxs.shuffle(&mut rng);
                all_idxs.truncate(actual_probe_size);
                all_idxs
            } else {
                (0..dataset.len()).collect()
            };

            let layouts = dataset.dynamic_layouts(idxs.as_slice()).map_err(|e| {
                ZTProducerErr::DatasetError {
                    idx: e.index(),
                    source: e,
                }
            })?;

            ZeroTensorBuffer::calculate_slot_size(&layouts, batch_size)
        };

        Ok(Self {
            slot_size,
            shm_filename: shm_filename.to_string(),
            socket_addr: socket_addr.as_ref().into(),
            num_slots: DEFAULT_SLOTS,
            read_timeout: None,
            overwrite_socket: false,
            shuffle: false,
            seed: None,
            max_steps: None,
        })
    }

    pub fn build(self) -> Result<ZeroTensorProducer, ZTProducerNewErr> {
        let running = Arc::new(AtomicBool::new(true));
        let rclone = running.clone();

        let _ = ctrlc::set_handler(move || {
            rclone.store(false, Ordering::SeqCst);
        });

        let buffer = ZeroTensorBuffer::new(&self.shm_filename, self.slot_size, self.num_slots)?;

        if self.socket_addr.exists() {
            if self.overwrite_socket {
                fs::remove_file(&self.socket_addr)?;
            } else {
                return Err(ZTProducerNewErr::IoError(io::Error::from(
                    io::ErrorKind::AddrInUse,
                )));
            }
        }

        let listener = UnixListener::bind(&self.socket_addr)?;

        Ok(ZeroTensorProducer {
            buffer,
            listener,
            sock_path: self.socket_addr,
            read_timeout: self.read_timeout,
            running,
            shuffle: self.shuffle,
            seed: self.seed,
            max_steps: self.max_steps,
            connected: false,
        })
    }
}

impl TryFrom<ZeroTensorProducerBuilder> for ZeroTensorProducer {
    type Error = ZTProducerNewErr;

    fn try_from(value: ZeroTensorProducerBuilder) -> Result<Self, Self::Error> {
        value.build()
    }
}
