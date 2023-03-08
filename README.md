# Probablistic Roadmap Method

PRM constructs a configuration space of all possible configurations of a robot. Samples are 
either chosen uniformly, randomly or by gaussian distribution or bridge gap.

PRM.py is the file where PRM class with four different sampling methods is implemented.
main.py is the scrip that provides helper functions that load the map from an image and call the classes and functions from
WPI_map.jpg is a binary Worcester Polytechnic Institute map image of with school buildings.

Four sampling methods are defined: 

1) Uniform Sampling:
Points were considered such that they increment 20 in pot x and y direction. The grid 
formed is square grid. The path found on a uniform sampling is always the same as 
the same points are always popped and only one connection is feasible.
Adv : Same sampling points are the same, low computing cost 
Dis : Same sampling points are the same ,Not optimal sampling

![unirform](https://user-images.githubusercontent.com/64325043/223866007-de678883-179a-4e79-9f59-a4e020786d1f.png)


2) Random Sampling:
In random sampling, a random points are taken and sampled for. The number of 
sampled points depend on the n_pts given. The graph is always different as new 
random points are propped up every time.
Adv: Ranom sampling points, lead to better results and low computing cost
Dis : depends on points being sampled, the path maybe long or no path may be found

![Random](https://user-images.githubusercontent.com/64325043/223866044-a07e4419-84fa-44bc-b867-4b53ae54878d.png)

3) Gaussian Sampling: 
The gaussian sampling is done by selecting a random point and then getting the next point 
using gaussian distribution. It is checked if either one of the point is an obstacle and only 
then is the point that is not an obstacle appended.
Adv : Sampling around the mean, optimal path found, Efficient
Dis : doesn’t cover the graph, more computational power

![Gaussian](https://user-images.githubusercontent.com/64325043/223866089-042fd3fb-dfd5-46df-8e78-cad1bacbc4a5.png)

4) Bridge Gap:
The bridge gap is similar to gaussian sampling, a random point is taken and the gaussian 
distribution gives us the other point. But here both the points must be obstacles and only 
then does the algorithm calculate a midpoint which is not an obstacle.
Adv: Better exploration, narrow passages , most optimal path
Dis : Computationally intensive

![image](https://user-images.githubusercontent.com/64325043/223866408-2d1257e4-5ab6-443a-8baa-ba9b1ce3551c.png)
