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
//! function. If you have a differentiable function that returns a scalar, you can call
//! [`Context::grad`] on it to take its gradient:
//!
//! ```
//! use phoebe::{differentiable, Context};
//!
//! #[differentiable]
//! fn square(x: f64) -> f64 {
//!    x * x
//! }
//!
//! fn main() {
//!     assert_eq!(square(3.), 9.);
//!     let ctx = Context::new();
//!     assert_eq!(ctx.grad(square::Repr)(3.), 6.);
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

use bumpalo::Bump;

use paste::paste;

/// Generate code for the derivative of a function.
pub use phoebe_macro::differentiable;

/// A context for memory allocation.
pub struct Context {
    bump: Bump,
}

impl Context {
    /// Create and return a new context.
    pub fn new() -> Self {
        Self { bump: Bump::new() }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

/// A [differentiable manifold][].
///
/// [differentiable manifold]: https://en.wikipedia.org/wiki/Differentiable_manifold
pub trait Manifold<'a> {
    /// This manifold's [cotangent bundle].
    ///
    /// [cotangent bundle]: https://en.wikipedia.org/wiki/Cotangent_bundle
    type Cotangent;

    /// Return the zero cotangent at a given point.
    fn zero(&self, ctx: &'a Context) -> Self::Cotangent;
}

impl Manifold<'_> for f64 {
    type Cotangent = Self;

    fn zero(&self, _: &Context) -> Self::Cotangent {
        0.
    }
}

macro_rules! impl_manifold_tuple {
    ($($T:tt),*) => {
        paste!{
            impl<'a, $($T),*> Manifold<'a> for ($($T,)*) where $($T: Manifold<'a>),*{
                type Cotangent = ($($T::Cotangent,)+);
                fn zero(&self, ctx: &'a Context) -> Self::Cotangent {
                        let ($([<$T:lower>],)*) = self;
                        ($([<$T:lower>].zero(ctx),)*)
                }
            }
        }
    }
}

impl_manifold_tuple!(A);
impl_manifold_tuple!(A, B);
impl_manifold_tuple!(A, B, C);
impl_manifold_tuple!(A, B, C, D);

impl<'a, T: Manifold<'a>> Manifold<'a> for &'a [T] {
    type Cotangent = &'a [T::Cotangent];

    fn zero(&self, ctx: &'a Context) -> Self::Cotangent {
        ctx.bump
            .alloc_slice_fill_iter(self.iter().map(|x| x.zero(ctx)))
    }
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

pub trait Differentiable<'a>: Function<Domain: Manifold<'a>, Codomain: Manifold<'a>> {
    fn vjp(
        self,
        ctx: &'a Context,
        x: Self::Domain,
        dx: <Self::Domain as Manifold<'a>>::Cotangent,
    ) -> (
        Self::Codomain,
        <Self::Codomain as Manifold<'a>>::Cotangent,
        impl FnOnce(
            <Self::Codomain as Manifold<'a>>::Cotangent,
        ) -> <Self::Domain as Manifold<'a>>::Cotangent,
    );
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

fn to_singleton<'a, T: Manifold<'a>>(dx: T::Cotangent) -> <(T,) as Manifold<'a>>::Cotangent {
    (dx,)
}

fn from_singleton<'a, T: Manifold<'a>>(dx: <(T,) as Manifold<'a>>::Cotangent) -> T::Cotangent {
    dx.0
}

impl Context {
    /// Take the [gradient][] of a scalar-valued function.
    ///
    /// [gradient]: https://en.wikipedia.org/wiki/Gradient
    pub fn grad<'a, T: Manifold<'a>, F: Differentiable<'a, Domain = (T,)> + 'a>(
        &'a self,
        f: F,
    ) -> impl FnOnce(T) -> T + 'a
    where
        T::Cotangent: Into<T>,
        <F::Codomain as Manifold<'a>>::Cotangent: Scalar,
    {
        move |x| {
            let dx = x.zero(self);
            let (_, _, df) = f.vjp(self, (x,), to_singleton(dx));
            from_singleton(df(<F::Codomain as Manifold>::Cotangent::one())).into()
        }
    }
}
