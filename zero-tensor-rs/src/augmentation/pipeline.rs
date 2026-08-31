use std::cell::RefCell;

use rand::{Rng};

use crate::augmentation::{Augmentation, AugmentationError, AugmentationItem};


pub type AugVec<I, O> = Vec<Box<dyn Augmentation<InputItem = I, OutputItem = O>>>;

const SCRATCH_INIT_CAP: usize = 3 * 224 * 224;
thread_local! {
    static SCRATCH_PAD: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(SCRATCH_INIT_CAP));
    static INTER_SCRATCH_PAD: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(SCRATCH_INIT_CAP));
}

#[derive(Debug, Default)]
pub struct AugmentationPipeline<I: AugmentationItem, O: AugmentationItem> {
    augmentations: AugVec<I, O>
}

impl<I: AugmentationItem, O: AugmentationItem> AugmentationPipeline<I, O> {
    pub fn new(augmentations: AugVec<I, O>) -> Self {
        Self { augmentations }
    }

    pub fn add<A: Augmentation<InputItem = I, OutputItem = O> + 'static>(mut self, augmentaion: A) -> Self {
        self.augmentations.push(Box::new(augmentaion));
        self
    }

    pub fn apply(&self, input: &[I], output: &mut [O], rng: Option<&mut dyn Rng>) -> Result<(), AugmentationError> {
        let mut default_rng = rand::rng();
        let rng = rng.unwrap_or(&mut default_rng);
        if self.augmentations.is_empty() {
            let o_input: &[O] = bytemuck::cast_slice(input);
            output.copy_from_slice(o_input);
            return Ok(());
        }
        
        if self.augmentations.len() == 1 {
            return self.augmentations[0].apply(input, output, Some(rng));
        }

        SCRATCH_PAD.with_borrow_mut(|pad| {
            let u8_input: &[u8] = bytemuck::cast_slice(input);
            if pad.len() < output.len() {
                pad.resize(u8_input.len(), 0);
            }

           self.augmentations[0].apply(input, output, Some(rng));

           for (i, aug) in self.augmentations.iter().skip(1).enumerate() {
                let is_last = i == self.augmentations.len() - 1;
                
           } 
        });

        Ok(())
    }
}