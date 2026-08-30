use indexmap::IndexMap;
use zero_tensor_lib::core::{
    dataset::{
        ZeroTensorDataset,
        item::{ShapeVec, StrideVec, TensorBatchLayout},
    },
    producer::ZeroTensorProducerBuilder,
};

struct BlankDS {
    layout: IndexMap<&'static str, TensorBatchLayout>,
}

impl BlankDS {
    pub fn new() -> Self {
        let layouts = TensorBatchLayout::new(
            ShapeVec::new(),
            StrideVec::new(),
            zero_tensor_lib::core::dataset::item::TensorDT::BF16,
        );
        let mut layout = IndexMap::new();
        layout.insert("data", layouts);
        Self { layout }
    }
}

impl<'a> ZeroTensorDataset<'a> for BlankDS {
    type Error = std::io::Error;

    fn len(&self) -> usize {
        64
    }

    fn static_layouts(&self) -> Option<&IndexMap<&'static str, TensorBatchLayout>> {
        Some(&self.layout)
    }

    fn write_item_into<'layout, 'b, 'c>(
        &self,
        _idx: usize,
        _writer: &mut zero_tensor_lib::core::writer::TensorWriter<'layout, 'b, 'c>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

fn main() {
    let dataset = BlankDS::new();
    let mut producer = ZeroTensorProducerBuilder::new(1024, "zt_blank", "zt_blank.sock")
        .build()
        .unwrap();

    producer.start_streaming(&dataset, 32).unwrap();
}
