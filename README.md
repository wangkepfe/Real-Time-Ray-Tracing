# Real-time Ray Tracing
- Physically based ray tracing / path tracing.
- GPU Compute workload implemented in CUDA.
- Real time rendering framework implemented with Vulkan.

## Screenshot
<table> <tr>
    <td> <img src="screenshot/screenShot1-11-21-2020.png" alt="Drawing" style="width: 600px;"/> </td>
</tr> <tr>
    <td> <img src="screenshot/screenShot2-11-21-2020.png" alt="Drawing" style="width: 600px;"/> </td>
</tr> <tr>
    <td> <img src="screenshot/screenShot3-11-21-2020.png" alt="Drawing" style="width: 600px;"/> </td>
</tr> </table>

## Physically based rendering
- Monte Carlo path tracing
- Surface BSDF models
- Rayleigh-Mie sky model
- Local light source

## Acceleration
### Variance reducing
- Low-discrepancy blue noise sampler
- Multiple importance sampling. Importance sampling BSDF and lights. Keep one sample per intersection.

### Traversal
- LBVH
- (todo) PLOC

### Denoising
- Edge-aware `Atrous
- Temporal spatial filter

### Other
- Adaptive sampling
- Dynamic resolution

## Post-processing
- Eye adaptive histogram based auto exposure
- Bloom effect
- Lens flare
- Tone mapping
- Sharpening

## References
- Low-Discrepancy Sampler https://perso.liris.cnrs.fr/david.coeurjolly/publications/heitz19.html
- Edge-aware `Atrous https://jo.dreggn.org/home/2010_atrous.pdf
- LBVH http://graphics.snu.ac.kr/class/graphics2011/references/2007_lauterbach.pdf
- PLOC https://meistdan.github.io/publications/ploc/paper.pdf
- PBRT book and pbrt-v3 code.
- https://vulkan-tutorial.com/ and https://github.com/SaschaWillems/Vulkan for my simple vulkan implementation
- https://github.com/NVIDIA/cuda-samples for common cuda functions
- http://raytracey.blogspot.com/2015/10/gpu-path-tracing-tutorial-1-drawing.html The blog series are great
- https://www.shadertoy.com/ great place to learn cool ideas about rendering
- https://www.scratchapixel.com/ basic stuffs. well explained
- https://www.iquilezles.org/www/index.htm All solid fun stuffs
- https://developer.nvidia.com/blog/thinking-parallel-part-i-collision-detection-gpu/ LBVH explained
- https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
- HLBVH https://research.nvidia.com/sites/default/files/pubs/2010-06_HLBVH-Hierarchical-LBVH/HLBVH-final.pdf
- AAC http://graphics.cs.cmu.edu/projects/aac/
- https://alain.xyz/blog/ray-tracing-denoising great coverage of denoising