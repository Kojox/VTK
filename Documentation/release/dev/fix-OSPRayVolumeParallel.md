fixes compositing artifacts when volume rendering in parallel with the OSPRay raycaster.  The fix was by reintroducing adaptive sampling rates based on volume size. Path tracing still has artifacts with parallel rendering due to lack of required data/ray communication between nodes to support that renderer.