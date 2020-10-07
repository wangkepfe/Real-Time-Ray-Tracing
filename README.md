# Real-time Ray Tracing
- Side project. No much time to maintain. Maybe ugly looking code and full of bugs.
- Physically based ray tracing / path tracing.
- GPU Compute workload implemented in CUDA.
- Real time rendering framework implemented with Vulkan.
- Achieved real-time interactive frame rate without utilizing ray tracing hardware features (e.g. Nvidia RTX). GPU work done in GPGPU computing only. The traditional offline rendering algorithms are modified here to aggressively achieve better performance with as small quality degradation as possible.

### Physically based rendering
- Global illumination using Monte Carlo path tracing
- Surface BSDF models. Lambertian diffuse, Fresnel effect, Trowbridge–Reitz microfacet. etc.
- Volumetric rendering: Rayleigh-Mie sky model
- Local light source

### Variance reducing
- Low-Discrepancy Sampler https://perso.liris.cnrs.fr/david.coeurjolly/publications/heitz19.html
- Multiple importance sampling (MIS)
    - Importance sampling BSDF
    - Importance smapling lights
    - Keep one sample per intersection! Use probability to choose from samples.

### Acceleration
- LBVH http://graphics.snu.ac.kr/class/graphics2011/references/2007_lauterbach.pdf

### Denoising
- Edge-aware `Atrous denoising https://jo.dreggn.org/home/2010_atrous.pdf
- Temporal filter
- Bilateral filter

### Post-processing
- Eye adaptive histogram based auto exposure
- Tone mapping
- Bloom effect
- Lens flare

### Some interesting ideas and studies I might want to implement in the future
- Separate diffuse and glossy for denoising https://pdfs.semanticscholar.org/a474/60591702574ccd38f8613839df15f31ef1af.pdf
- HLBVH https://research.nvidia.com/sites/default/files/pubs/2010-06_HLBVH-Hierarchical-LBVH/HLBVH-final.pdf
- AAC http://graphics.cs.cmu.edu/projects/aac/
- PLOC https://meistdan.github.io/publications/ploc/paper.pdf

### References
- All links listed above.
- PBRT book and pbrt-v3 code. Invaluable
- https://vulkan-tutorial.com/ and https://github.com/SaschaWillems/Vulkan for my simple vulkan implementation
- https://github.com/NVIDIA/cuda-samples for common cuda functions
- http://raytracey.blogspot.com/2015/10/gpu-path-tracing-tutorial-1-drawing.html The blog series are great
- https://www.shadertoy.com/ great place to learn cool ideas about rendering
- https://www.scratchapixel.com/ basic stuffs. well explained
- https://www.iquilezles.org/www/index.htm All solid fun stuffs
- https://developer.nvidia.com/blog/thinking-parallel-part-i-collision-detection-gpu/ LBVH explained
- https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda