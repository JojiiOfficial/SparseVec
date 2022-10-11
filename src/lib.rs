pub mod dim_iter;
pub mod sp_vec;
pub mod sparse_iter;
pub mod weight_iter;

use dim_iter::DimIter;
use intersect_iter::TupleIntersect;
use num_traits::Float;
use sparse_iter::SpVecIter;
use std::{
    ops::{Add, Mul},
    slice::IterMut,
};
use weight_iter::WeightIter;

pub use sp_vec::SpVector;

pub type SpVec32 = SpVector<f32>;
pub type SpVec64 = SpVector<f64>;

pub trait VecExt {
    type Wtype: Float + Default;

    /// Create a new Vec from raw values. Values don't have to be ordered
    fn create_new_raw<I>(sparse: I) -> Self
    where
        I: IntoIterator<Item = (u32, Self::Wtype)>;

    /// Create a new Vec from raw values. `sparse` must be sorted by dimensions and `length` must
    /// be the length of the vector `sparse` represents
    fn new_raw(sparse: Vec<(u32, Self::Wtype)>, length: Self::Wtype) -> Self;

    /// Creates a new empty vector
    fn empty() -> Self;

    /// Returns the vectors length in the given vector space
    fn get_length(&self) -> Self::Wtype;

    /// Returns `true` if there is at least one dimension with a value > 0
    #[inline]
    fn is_empty(&self) -> bool {
        self.dim_count() == 0
    }

    /// Returns a mutable reference to a stdlib Vector with the sparse indices and values
    fn as_vec_mut(&mut self) -> &mut Vec<(u32, Self::Wtype)>;

    /// Returns a reference to a stdlib Vector with the sparse indices and values
    fn as_vec(&self) -> &Vec<(u32, Self::Wtype)>;

    /// Returns the amount of sparse pairs
    fn dim_count(&self) -> usize;

    /// Returns the vectors value in the given dimension
    fn get_dim(&self, dim: usize) -> Option<Self::Wtype>;

    /// Sets the vectors value in the given dimension
    fn set_dim(&mut self, dim: usize, val: Self::Wtype);

    /// Returns `true` if the vector has a value > 0 in the given dimension
    fn has_dim(&self, dim: usize) -> bool;

    /// Updates the vector after change
    #[inline]
    fn update(&mut self) {}

    /// Returns an iterator over all dimensions with a value and skips those that are 0
    fn iter(&self) -> SpVecIter<'_, Self::Wtype>;

    /// Returns an iterator over all dimensions with a value and skips those that are 0
    #[inline]
    fn iter_mut(&mut self) -> IterMut<'_, (u32, Self::Wtype)> {
        self.as_vec_mut().iter_mut()
    }

    /// Returns an iterator over all dimensions with a value > 0
    fn dimensions(&self) -> DimIter<'_, Self::Wtype>;

    /// Returns an iterator over all weight > 0
    fn weights(&self) -> WeightIter<'_, Self::Wtype>;

    /// Sets the dimensions value to 0
    #[inline]
    fn delete_dim(&mut self, dim: usize) {
        self.set_dim(dim, Self::Wtype::default());
    }

    /// Returns the last (greatest) dimension of the vector
    fn last_dim(&self) -> Option<usize>;

    /// Returns the first (smallest) dimension of the vector
    fn first_dim(&self) -> Option<usize>;

    /// Returns `true` if both vectors could potentionally have overlapping vectors.
    /// This is just an indication whether they could overlap and therefore faster than
    /// `overlaps_with` but less accurate
    #[inline]
    fn could_overlap<V: VecExt>(&self, other: &V) -> bool {
        let cant_overlap = self.is_empty()
            || other.is_empty()
            || self.first_dim() > other.last_dim()
            || self.last_dim() < other.first_dim();

        !cant_overlap
    }

    /// Returns `true` if both vectors have at least one dimension in common. If this value is
    /// true, the standart scalar product is not zero
    #[inline]
    fn overlaps_with<V: VecExt<Wtype = Self::Wtype>>(&self, other: &V) -> bool {
        if !self.could_overlap(other) {
            return false;
        }
        self.intersect_iter(other).next().is_some()
    }

    /// Returns an iterator over all dimensions with both vectors having a value > 0
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

    /// Calculates the cosine similarity between two vectors
    #[inline]
    fn cosine<V: VecExt<Wtype = Self::Wtype>>(&self, other: &V) -> Self::Wtype {
        if !self.could_overlap(other) {
            return Self::Wtype::default();
        }

        let sc = self.scalar(other);
        sc / (self.get_length() * other.get_length())
    }
}

#[cfg(test)]
mod test {
    // it works, trust me
}
