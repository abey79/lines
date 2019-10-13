# TODO

## Shapes

- Striped cubes (as in fogleman/ln)

## Skins

- StripeSkin: vertical striping for arbitrary shape

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

- Automate example running:
    - factor all example in a function that returns a RenderedScene
    - generic example runner with option to run svg, matplotlib, timeit, logging level, etc.
    - script to run all example for automated testing purposes
- Map drawer

## Misc

- using Logging for all output

## Bugs

- empty scene crashes
- cow.obj crashes