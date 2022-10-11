use std::ops::AddAssign;

use crate::{sparse_iter::SpVecIter, VecExt};
use num_traits::Float;
use serde::{Deserialize, Serialize};

/// Sparse vector implementation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpVector<W> {
    /// Dimensions mapped to values
    inner: Vec<(u32, W)>,
    /// Length of the vector
    length: W,
}

impl<W> SpVector<W> {
    #[inline]
    pub(crate) fn dim_index(&self, dim: usize) -> Option<usize> {
        self.inner.binary_search_by(|a| a.0.cmp(&(dim as u32))).ok()
    }

    /// Sort the Vec<> by the dimensions
    #[inline]
    fn sort(&mut self) {
        self.inner.sort_by(|a, b| a.0.cmp(&b.0));
        self.inner.dedup_by(|a, b| a.0 == b.0);
    }
}

impl<W> SpVector<W>
where
    W: Float + Default,
{
    /// Calculate the vector length
    #[inline]
    fn calc_len(&self) -> W {
        //self.inner.iter().map(|(_, i)| i.powi(2)).sum::<W>().sqrt()
        self.inner
            .iter()
            .map(|(_, i)| i.powi(2))
            .fold(W::default(), |i, b| i.add(b))
    }
}

impl<W> VecExt for SpVector<W>
where
    W: Float + Default,
{
    type Wtype = W;

    #[inline]
    fn create_new_raw<I>(sparse: I) -> Self
    where
        I: IntoIterator<Item = (u32, Self::Wtype)>,
    {
        let mut vec = Self {
            inner: sparse.into_iter().collect(),
            length: W::default(),
        };
        vec.update();
        vec
    }

    #[inline]
    fn new_raw(sparse: Vec<(u32, Self::Wtype)>, length: Self::Wtype) -> Self {
        Self {
            inner: sparse,
            length,
        }
    }

    #[inline]
    fn empty() -> Self {
        Self::default()
    }

    #[inline]
    fn get_length(&self) -> Self::Wtype {
        self.length
    }

    #[inline]
    fn as_vec_mut(&mut self) -> &mut Vec<(u32, Self::Wtype)> {
        &mut self.inner
    }

    #[inline]
    fn as_vec(&self) -> &Vec<(u32, W)> {
        &self.inner
    }

    #[inline]
    fn dim_count(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    fn get_dim(&self, dim: usize) -> Option<Self::Wtype> {
        let index = self.dim_index(dim)?;
        self.inner.get(index).map(|i| i.1)
    }

    #[inline]
    fn set_dim(&mut self, dim: usize, val: Self::Wtype) {
        if !self.has_dim(dim) {
            self.inner.push((dim as u32, val));
        }

        *self.inner.iter_mut().find(|i| i.0 == dim as u32).unwrap() = (dim as u32, val);
    }

    #[inline]
    fn has_dim(&self, dim: usize) -> bool {
        self.dim_index(dim).is_some()
    }

    #[inline]
    fn update(&mut self) {
        self.sort();
        self.length = self.calc_len();
    }

    #[inline]
    fn iter(&self) -> SpVecIter<'_, Self::Wtype> {
        SpVecIter::new(self)
    }

    #[inline]
    fn last_dim(&self) -> Option<usize> {
        self.inner.last().map(|i| i.0 as usize)
    }

    #[inline]
    fn first_dim(&self) -> Option<usize> {
        self.inner.first().map(|i| i.0 as usize)
    }
}

impl<W: Default> Default for SpVector<W> {
    #[inline]
    fn default() -> Self {
        Self {
            inner: Default::default(),
            length: Default::default(),
        }
    }
}

impl<V: Float + Default> AddAssign<&Self> for SpVector<V> {
    #[inline]
    fn add_assign(&mut self, rhs: &SpVector<V>) {
        assert_eq!(rhs.dim_count(), self.dim_count());

        for (s, o) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            //s.1 += o.1;
            s.1 = s.1.add(o.1);
        }

        self.update();
    }
}

impl<V: Float + Default> PartialEq for SpVector<V> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<V: Float + Default> Eq for SpVector<V> {}
