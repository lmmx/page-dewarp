The moments of a shape are particular weighted averages (in statistics, "moments")
of the pixel intensities. In physics they're the distribution of matter about
a point or an axis etc.

They are usually described as a function $f(x,y)$ which can just be read as
"the value (or intensity) of the pixel in an image at position $(x,y)$", so $f$ is the 'image function'.

If the image is binary (e.g. a binary blob), then we'd say the image function f takes
values in $[0,1]$.

A moment of order $p + q$ is defined for a 2D continuous function on a region as:

$$M_{pq} = \iint x^p y^q f(x,y) \, dxdy$$

So if we talk about "first order moments" then we mean $M_{01}$, $M_{10}$,

- $M_{01} = \iint y \, f(x,y) \, dxdy$
- $M_{10} = \iint x \, f(x,y) \, dxdy$

and by "second order moments" we mean $M_{11}$, $M_{20}$, $M_{02}$.

Since we have discrete pixels not a continuous function, in fact the integrals
just become summations over $x$ and $y$. So in other words: moments $M_pq$ counts
the pixels over an image $f(x,y)$ and the $0$'th moment is just the total count
of pixels in an image, i.e. its area (it's usually called the area moment for binary
images or otherwise the 'mass' of the image for grayscale etc.).

Image moments are sensitive to position, but by subtracting the average $x$
and $y$ position in the above equation we can obtain a translation-invariant
"central moment".

$$\mu_{pq} = \iint (x-\bar{x})^p (y-\bar{y})^q f(x,y) \, dxdy$$

where

- $\bar{x} = \frac{M_{10}}{M_{00}}$ (first order x moment divided by the area moment)
- $\bar{y} = \frac{M_{01}}{M_{00}}$ (first order y moment divided by the area moment)
- $(\bar{x},\bar{y})$ is the centroid (a.k.a. centre of gravity)

So far so simple.

Second order moments $M_{20}$ and $M_{02}$ describe the "distribution of mass"
of the image with respect to the coordinate axes. In mechanics they're the _moments of inertia_.

Another mechanical quality is the _radius of gyration_
with respect to an axis, expressed as $\sqrt{\frac{M_{20}}{M_{00}}}$ and $\sqrt{\frac{M_{02}}{M_{00}}}$

There are others: scale invariant moments and rotation invariant moments
(the latter are a.k.a. Hu moments).

**The moment of interest for the code in this repo is a little more complicated.**

## Orientation information from image moments

The covariance matrix of an image can be obtained from second order central moments as below:

Information about image orientation can be derived by first using the second
order central moments to construct a covariance matrix.

- $\mu'_{20} = \mu_{20}/\mu_{00} = M_{20}/M_{00} - \bar{x}^2$
- $\mu'_{02} = \mu_{02}/\mu_{00} = M_{02}/M_{00} - \bar{y}^2$
- $\mu'_{11} = \mu_{11}/\mu_{00} = M_{11}/M_{00} - \bar{x}\bar{y}$

The covariance matrix of the image $I(x,y)$ is now

$$\text{cov}[I(x,y)] = \begin{bmatrix} \mu'_{20} & \mu'_{11} \\ \mu'_{11} & \mu'_{02} \end{bmatrix}.$$

The eigenvectors of this matrix correspond to the major and minor axes of the
image intensity, so the orientation can thus be extracted from the angle of the
eigenvector associated with the largest eigenvalue towards the axis closest to this
eigenvector.

It can be shown that this angle $\Theta$ is given by the formula:

$$\Theta = \frac{1}{2} \arctan\left(\frac{2\mu'_{11}}{\mu'_{20} - \mu'_{02}}\right)$$

The above formula holds as long as $\mu'_{20} - \mu'_{02} \neq 0$.

This is constructed in the `blob_mean_and_tangent` function of this library's
`contours.py` module, and then PCA is computed on it by Singular Value Decomposition.
(These are covered in most good introductions to linear algebra, I recommend Strang)

- See [this article](https://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/)
  for a nice geometric interpretation of covariance, which makes this usage intuitive

## Further info

In fact, the moments above are all _geometric moments_ since the polynomial basis
is a standard power basis $x^p$ multiplied by $y^q$.

If you want to read more, check out
Flusser et al's book _Moments and Moment Invariants in Pattern Recognition_.
