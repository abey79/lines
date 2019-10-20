# TODO

## Skins

- Switch to duck typing with proper Typing annotation instead of base class. This will enable to cleanly increase the 
number of hooks. See: https://stackoverflow.com/questions/44155938/duck-typing-with-python-3-5-style-type-annotations
- Changed API to also allow the skin to operate at the vertex, seg_idx, face_idx step. 
- StripeSkin: vertical striping for arbitrary shape (use trimesh lib for that purpose!)


## Renderer

- Properly handle aspect ratio, which should be propagated to RenderedScene and, ultimately, to the export process
- Invert rendering pipeline (process per segment instead of per face), enables parallel processing!!
- Frustum filtering should take into account Z direction as well
- Frustum filtering should apply to faces as well
- Example OBJ with cow.obj crashes
- Option to render faces as well with shading (may require properly winded faces)
- Option to render hidden segments (might be possible with renderer V2)

 
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
- This creates a glitch:

 ```python
scene = Scene()
scene.add(Cube())
scene.look_at((2, 1, 2), (0, 0, 0))
scene.perspective(50 / 180 * math.pi, 0.1, 10)
```
