# cvtb
Pieces of tiny scripts which I find handy every now and then.

# utilities
## visualization
```python
from cvtb import vis

# dynamic point trajectories
vis.pcd(points, colors)  
vis.pcd_demo()

# static points
vis.pcd_static(points, colors)
vis.pcd_static_demo()

# lists of points
vis.pcds([points0, points1, points2], 
         [colors0, colors1, colors2])
vis.pcds_demo()
```

## color
```python
from cvtb import color

color.gene
```
