mod internal;

use proc_macro::TokenStream;
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn differentiable(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let function = parse_macro_input!(item as ItemFn);
    internal::differentiable(function)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}
