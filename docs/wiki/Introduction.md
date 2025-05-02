The technical term for dewarping is "homography", and the matrix to describe the 'dewarping' transform
is properly known as the homography matrix

- 📖 See Szeliski (2011) _Computer Vision: Algorithms and Applications_
  ([1st ed.](https://szeliski.org/Book/1stEdition.htm))
  - §2.1: _Geometric primitives and transformations_
  - §3.6.1: _Parametric transformations_

The homography matrix uses homogeneous coordinates, indicated by a tilde over the letters.

Homogeneous coordinates are a column vector of the form [**x̃ʹ**,**ỹʹ**,**w̃ʹ**] where the **w̃ʹ**
can be considered the 'weights' that you divide by to obtain the inhomogeneous coordinates
**x̃** and **ỹ** from **x̃ʹ** and **ỹʹ**.

## A concise explainer on homogeneous coordinates by Matt Zucker & links

[![](https://raw.githubusercontent.com/lmmx/shots/master/2021/Apr/zucker-homogeneous-coords.png)](https://mzucker.github.io/2016/10/11/unprojecting-text-with-ellipses.html)

- See also Matt's lecture whiteboard notes:
  - [Projective algebra fundamentals](https://mzucker.github.io/swarthmore/e27_s2021/notes/e27-2021-04-13.pdf)
    ([web link](https://mzucker.github.io/swarthmore/e27_s2021/index.html#schedule1_2021-4-13))
  - [Homographies and homogeneous least squares](https://mzucker.github.io/swarthmore/e27_s2021/notes/e27-2021-04-15.pdf)
    ([web link](https://mzucker.github.io/swarthmore/e27_s2021/index.html#schedule1_2021-4-15))

The important point here is that the inverse of a homography will undo it.

- Homographies are a group (Szeliski §2.1.2) meaning they're closed under composition and __have an inverse__
  (so if we can compute that inverse we can undo the transformation). This is the key idea in 'dewarping'

Additionally, I put together some further background notes:

- [Background on image moments][Background-on-image-moments]
