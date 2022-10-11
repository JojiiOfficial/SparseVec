pub mod iter;
pub mod sp_vec;

use std::ops::{Add, Mul};

use crate::sp_vec::SpVector;
use intersect_iter::TupleIntersect;
use iter::SpVecIter;
use num::{Float, NumCast};

pub type Vector32 = SpVector<f32>;
pub type Vector64 = SpVector<f64>;

pub trait VecExt {
    type Wtype: Float + Default;

    fn get_length(&self) -> Self::Wtype;

    fn empty() -> Self;

    fn as_vec(&self) -> &Vec<(u32, Self::Wtype)>;

    fn dim_count(&self) -> usize;

    fn get_dim(&self, dim: usize) -> Option<(usize, Self::Wtype)>;

    fn set_dim(&mut self, dim: usize, val: Self::Wtype);

    fn has_dim(&self, dim: usize) -> bool;

    fn update(&mut self) {}

    fn iter(&self) -> iter::SpVecIter<'_, Self::Wtype>;

    #[inline]
    fn delete_dim(&mut self, dim: usize) {
        self.set_dim(dim, NumCast::from(0.0).unwrap());
    }

    fn last_indice(&self) -> Option<usize>;

    fn first_indice(&self) -> Option<usize>;

    /// Returns `true` if both vectors could potentionally have overlapping vectors.
    /// This is just an indication whether they could overlap and therefore faster than
    /// `overlaps_with` but less accurate
    #[inline]
    fn could_overlap<V: VecExt>(&self, other: &V) -> bool {
        let cant_overlap = self.is_empty()
            || other.is_empty()
            || self.first_indice() > other.last_indice()
            || self.last_indice() < other.first_indice();

        !cant_overlap
    }

    /// Returns `true` if both vectors have at least one dimension in common
    #[inline]
    fn overlaps_with<V: VecExt<Wtype = Self::Wtype>>(&self, other: &V) -> bool {
        if !self.could_overlap(other) {
            return false;
        }
        self.intersect_iter(other).next().is_some()
    }

    #[inline]
    fn intersect_iter<'a, 'b, V: VecExt<Wtype = Self::Wtype>>(
        &'a self,
        other: &'b V,
    ) -> TupleIntersect<SpVecIter<'a, Self::Wtype>, SpVecIter<'b, Self::Wtype>, usize, Self::Wtype>
    {
        let self_iter = self.iter();
        let other_iter = other.iter();
        TupleIntersect::new(self_iter, other_iter)
    }

    /// Returns the scalar product of self and `other`
    #[inline]
    fn scalar<V: VecExt<Wtype = Self::Wtype>>(&self, other: &V) -> Self::Wtype {
        self.intersect_iter(other)
            .map(|(_, a, b)| a.mul(b))
            .fold(Self::Wtype::default(), |a, b| a.add(b))
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.dim_count() == 0
    }
}
