# Ultimate-Realism-Renderer
Physically based ray tracing / path tracing with CUDA. Real time rendering with Vulkan.

### Physically based rendering
- Global illumination using path tracing
- Lambertian diffuse
- Fresnel effect
- Trowbridge–Reitz microfacet model
- Rayleigh-Mie sky model
- Local light source

### Variance reducing
- Quasi Monte Carlo random
- BRDF important sampling
- Multiple important sampling

### Acceleration
- BVH (todo)
- SIMD re-grouping (todo)

### Post-processing & denoising
- Eye adaption, histogram based auto exposure
- Tone-mapping
- Atrous denoising with spike removal
- TAA (temporal anti-aliasing)