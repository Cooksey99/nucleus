fn main() {
    #[cfg(all(target_os = "macos", feature = "coreml"))]
    {
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Foundation");
        
        cc::Build::new()
            .file("src/provider/coreml_wrapper.m")
            .flag("-fobjc-arc")
            .compile("coreml_wrapper");
        
        println!("cargo:rerun-if-changed=src/provider/coreml_wrapper.m");
    }
}
