//! Phoebe does reverse-mode [automatic differentiation][] (often abbreviated "autodiff" or simply
//! "AD") of Rust code.
//!
//! # Usage
//!
//! First add Phoebe as a dependency in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! phoebe = "0.1"
//! ```
//!
//! You can make a function differentiable by adding the [`differentiable`] attribute to it. Because
//! [`fn_traits`][] is unstable, Phoebe creates a module with the same name as a function, and puts
//! a type called `Repr` in that module which represents the differentiable semantics of that
//! function. If you have a differentiable function that returns a scalar, you can call [`grad`] on
//! it to take its gradient:
//!
//! ```
//! use phoebe::{differentiable, grad};
//!
//! #[differentiable]
//! fn square(x: f64) -> f64 {
//!    x * x
//! }
//!
//! fn main() {
//!     assert_eq!(square(3.), 9.);
//!     assert_eq!(grad(square::Repr)(3.), 6.);
//! }
//! ```
//!
//! For more advanced usage, see the [details](#details) section.
//!
//! # Background
//!
//! Of course, with the rising popularity of machine learning, there are already several crates
//! which do this. Why another one? The answer: **performance**.
//!
//! # Details
//!
//! [`fn_traits`]: https://doc.rust-lang.org/stable/unstable-book/library-features/fn-traits.html
//! [automatic differentiation]: https://en.wikipedia.org/wiki/Automatic_differentiation

// the proc macro emits a lot of absolute paths starting with `::phoebe` that otherwise wouldn't
// work in this crate, which would expect those to instead start with `crate`; this fixes that
// problem by making `::phoebe` valid even inside of this crate
extern crate self as phoebe;

/// Generate code for the derivative of a function.
pub use phoebe_macro::differentiable;

/// A [differentiable manifold][].
///
/// [differentiable manifold]: https://en.wikipedia.org/wiki/Differentiable_manifold
pub trait Manifold {
    /// This manifold's [cotangent bundle].
    ///
    /// [cotangent bundle]: https://en.wikipedia.org/wiki/Cotangent_bundle
    type Cotangent;
}

impl Manifold for f64 {
    type Cotangent = Self;
}

impl<T: Manifold> Manifold for (T,) {
    type Cotangent = (T::Cotangent,);
}

/// A [function][].
///
/// [function]: https://en.wikipedia.org/wiki/Function_(mathematics)
pub trait Function {
    /// This function's [domain][].
    ///
    /// [domain]: https://en.wikipedia.org/wiki/Domain_of_a_function
    type Domain;

    /// This function's [codomain][].
    ///
    /// [codomain]: https://en.wikipedia.org/wiki/Codomain
    type Codomain;

    /// Return the result of applying this function to a given element of the domain.
    fn apply(self, x: Self::Domain) -> Self::Codomain;
}

pub trait Differentiable: Function<Domain: Manifold, Codomain: Manifold> {
    fn vjp(
        self,
        x: Self::Domain,
        dx: <Self::Domain as Manifold>::Cotangent,
    ) -> (
        Self::Codomain,
        <Self::Codomain as Manifold>::Cotangent,
        impl FnOnce(<Self::Codomain as Manifold>::Cotangent) -> <Self::Domain as Manifold>::Cotangent,
    );
}

/// A [vector space][].
///
/// [vector space]: https://en.wikipedia.org/wiki/Vector_space
pub trait VectorSpace {
    /// The [zero vector][].
    ///
    /// [zero vector]: https://en.wikipedia.org/wiki/Zero_element
    fn zero() -> Self;
}

impl VectorSpace for f64 {
    fn zero() -> Self {
        0.
    }
}

/// A [scalar][].
///
/// [scalar]: https://en.wikipedia.org/wiki/Scalar_(mathematics)
pub trait Scalar {
    /// The [unit][].
    ///
    /// [unit]: https://en.wikipedia.org/wiki/Unit_(ring_theory)
    fn one() -> Self;
}

impl Scalar for f64 {
    fn one() -> Self {
        1.
    }
}

fn to_singleton<T: Manifold>(dx: T::Cotangent) -> <(T,) as Manifold>::Cotangent {
    (dx,)
}

fn from_singleton<T: Manifold>(dx: <(T,) as Manifold>::Cotangent) -> T::Cotangent {
    dx.0
}

/// Take the [gradient][] of a scalar-valued function.
///
/// [gradient]: https://en.wikipedia.org/wiki/Gradient
pub fn grad<T: Manifold, F: Differentiable<Domain = (T,)>>(f: F) -> impl FnOnce(T) -> T::Cotangent
where
    T::Cotangent: VectorSpace + Clone,
    <F::Codomain as Manifold>::Cotangent: Scalar,
{
    |x| {
        let (_, _, df) = f.vjp((x,), to_singleton(T::Cotangent::zero()));
        from_singleton(df(<F::Codomain as Manifold>::Cotangent::one()))
    }
}
