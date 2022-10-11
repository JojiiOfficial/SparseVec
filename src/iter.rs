use num::Float;

use crate::{sp_vec::SpVector, VecExt};

pub struct SpVecIter<'a, W> {
    vec: &'a SpVector<W>,
    pos: usize,
}

impl<'a, W> SpVecIter<'a, W> {
    #[inline]
    pub(crate) fn new(vec: &'a SpVector<W>) -> Self {
        Self { vec, pos: 0 }
    }
}

impl<'a, W> Iterator for SpVecIter<'a, W>
where
    W: Float + Default,
{
    type Item = (usize, W);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let item = self.vec.as_vec().get(self.pos)?;
        self.pos += 1;
        Some((item.0 as usize, item.1))
    }
}
