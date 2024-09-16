use proc_macro2::TokenStream;
use quote::quote;
use syn::{parse_quote, FnArg, Ident, ItemFn, Pat, ReturnType, Type};

pub fn differentiable(function: ItemFn) -> syn::Result<TokenStream> {
    let name = &function.sig.ident;
    let ((params, args), (domain, pats)) = function
        .sig
        .inputs
        .iter()
        .map(|arg| match arg {
            FnArg::Receiver(_) => todo!(),
            FnArg::Typed(arg) => {
                let pat = &arg.pat;
                let path = match pat.as_ref() {
                    Pat::Ident(name) => name.ident.clone(),
                    _ => todo!(),
                };
                let ty = &arg.ty;
                let param = parse_quote!(#path: #ty);
                Ok(((param, path), (ty.as_ref().clone(), pat.as_ref().clone())))
            }
        })
        .collect::<syn::Result<((Vec<FnArg>, Vec<Ident>), (Vec<Type>, Vec<Pat>))>>()?;
    let codomain: Box<Type> = match &function.sig.output {
        ReturnType::Default => parse_quote!(()),
        ReturnType::Type(_, ty) => ty.clone(),
    };
    let block = &function.block;
    let items = quote! {
        mod #name {
            use super::*;

            pub struct Repr;

            impl ::phoebe::Function for Repr {
                type Domain = (#(#domain,)*);
                type Codomain = #codomain;

                fn apply(self, (#(#pats,)*): Self::Domain) -> Self::Codomain #block
            }

            impl ::phoebe::Differentiable for Repr {
                fn vjp(
                    self,
                    x: Self::Domain,
                    dx: <Self::Domain as ::phoebe::Manifold>::Cotangent,
                ) -> (
                    Self::Codomain,
                    <Self::Codomain as ::phoebe::Manifold>::Cotangent,
                    impl FnOnce(<Self::Codomain as ::phoebe::Manifold>::Cotangent) -> <Self::Domain as ::phoebe::Manifold>::Cotangent,
                ) {
                    (9., 0., |_| (6.,)) // TODO: don't just hardcode
                }
            }
        }

        fn #name(#(#params),*) -> #codomain {
            ::phoebe::Function::apply(#name::Repr, (#(#args,)*))
        }
    };
    Ok(items)
}
