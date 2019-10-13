# TODO

## Shapes

- SilhouettePolyShape: output segment for the silhouette, to allow easy rendering of curved shape (cones, cylinders, etc.)
- Striped cubes (as in fogleman/ln)


## Renderer

- Frustum filtering should take into account Z direction as well
- Frustum filtering should apply to faces as well
- Example OBJ with deer.obj has artifacts
- Example OBJ with cow.obj crashes
- Scene.render() should return a RenderedScene object
 
## RenderedScene

- Data members:
    - output segments
    - optimized output segments
    - unmasked input segments
    - faces
    - etc.
    
- Function member:
    - plot with matplotlib
    - output to svg
    - output to AxiDraw
    - etc. 

## Examples

- Map drawer
