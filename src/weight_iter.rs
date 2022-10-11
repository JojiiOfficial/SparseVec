use crate::{sp_vec::SpVector, sparse_iter::SpVecIter};
use num_traits::Float;

pub struct WeightIter<'a, W> {
    iter: SpVecIter<'a, W>,
}

impl<'a, W> WeightIter<'a, W> {
    #[inline]
    pub(crate) fn new(vec: &'a SpVector<W>) -> Self {
        let iter = SpVecIter::new(vec);
        Self { iter }
    }
}

impl<'a, W> Iterator for WeightIter<'a, W>
where
    W: Float + Default,
{
    type Item = W;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.iter.next()?.1)
    }
}
