import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def surface_curvature_3d(X, Y, Z):
  # Shape
  (lr,lb) = X.shape

  # First Derivatives
  Xv,Xu = np.gradient(X)
  Yv,Yu = np.gradient(Y)
  Zv,Zu = np.gradient(Z)

  # Second Derivatives
  Xuv,Xuu = np.gradient(Xu)
  Yuv,Yuu = np.gradient(Yu)
  Zuv,Zuu = np.gradient(Zu)

  Xvv,Xuv = np.gradient(Xv)
  Yvv,Yuv = np.gradient(Yv)
  Zvv,Zuv = np.gradient(Zv)

  # 2D to 1D conversion
  # Reshape to 1D vectors
  Xu = np.reshape(Xu,lr*lb)
  Yu = np.reshape(Yu,lr*lb)
  Zu = np.reshape(Zu,lr*lb)
  Xv = np.reshape(Xv,lr*lb)
  Yv = np.reshape(Yv,lr*lb)
  Zv = np.reshape(Zv,lr*lb)
  Xuu = np.reshape(Xuu,lr*lb)
  Yuu = np.reshape(Yuu,lr*lb)
  Zuu = np.reshape(Zuu,lr*lb)
  Xuv = np.reshape(Xuv,lr*lb)
  Yuv = np.reshape(Yuv,lr*lb)
  Zuv = np.reshape(Zuv,lr*lb)
  Xvv = np.reshape(Xvv,lr*lb)
  Yvv = np.reshape(Yvv,lr*lb)
  Zvv = np.reshape(Zvv,lr*lb)

  Xu = np.c_[Xu, Yu, Zu]
  Xv = np.c_[Xv, Yv, Zv]
  Xuu = np.c_[Xuu, Yuu, Zuu]
  Xuv = np.c_[Xuv, Yuv, Zuv]
  Xvv = np.c_[Xvv, Yvv, Zvv]

  # First fundamental Coeffecients of the surface (E,F,G)
  E = np.einsum('ij,ij->i', Xu, Xu)
  F = np.einsum('ij,ij->i', Xu, Xv)
  G = np.einsum('ij,ij->i', Xv, Xv)

  m = np.cross(Xu, Xv, axisa=1, axisb=1)
  p = np.sqrt(np.einsum('ij,ij->i', m, m))
  n = m / np.c_[p,p,p] # n is the normal

  # Second fundamental Coeffecients of the surface (L,M,N), (e,f,g)
  L = np.einsum('ij,ij->i', Xuu, n) #e
  M = np.einsum('ij,ij->i', Xuv, n) #f
  N = np.einsum('ij,ij->i', Xvv, n) #g

  # Gaussian Curvature
  # Alternative formula for gaussian curvature in wiki
  # K = det(second fundamental) / det(first fundamental)
  K = (L*N-M**2) / (E*G-F**2)
  K = np.reshape(K, lr*lb)
  # wiki trace of (second fundamental)(first fundamental inverse)

  # Mean Curvature
  H = ((E*N + G*L - 2*F*M) / ((E*G - F**2))) / 2
  H = np.reshape(H,lr*lb)

  # Principle Curvatures
  Pmax = H + np.sqrt(H**2 - K)
  Pmin = H - np.sqrt(H**2 - K)

  # Curvedness
  # 3D Shape Modeling for Cell Nuclear Morphological Analysis and Classification
  CV = np.sqrt((Pmin**2 + Pmax**2) / 2)

  # Shape Index
  # 3D Shape Modeling for Cell Nuclear Morphological Analysis and Classification
  SI = np.zeros(Pmax.shape)
  idx = (Pmax - Pmin) != 0
  SI[idx] = (2 / np.pi) * np.arctan((Pmin[idx] + Pmax[idx])/(Pmax[idx] - Pmin[idx]))

  # Reshape
  K = np.reshape(K,(lr,lb))
  H = np.reshape(H,(lr,lb))
  Pmax = np.reshape(Pmax,(lr,lb))
  Pmin = np.reshape(Pmin,(lr,lb))
  CV = np.reshape(CV,(lr,lb))
  SI = np.reshape(SI,(lr,lb))

  return K, H, Pmax, Pmin, CV, SI


if __name__ == '__main__':

  # Generate surface data
  X = np.arange(-5, 5, 0.25)
  Y = np.arange(-5, 5, 0.25)
  X, Y = np.meshgrid(X, Y)
  Z = np.exp(-0.1*X**2-0.1*Y**2)

  # Curvature 3D
  K, H, Pmax, Pmin, CV, SI = surface_curvature_3d(X, Y, Z)

  # Plot
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  # Auto-adjust true radius into [0,1] for color mapping
  norm = Normalize()
  colors = norm(CV)
  cmap = cm.get_cmap("jet")

  # Surface
  ax.plot_surface(X, Y, Z, facecolors=cmap(colors),
                  linewidth=0, antialiased=False)

  # Add a color bar which maps values to colors
  # the surface is not mappable, we need to handle the colorbar manually
  mappable = cm.ScalarMappable(cmap=cmap)
  mappable.set_array(colors)
  fig.colorbar(mappable, shrink=0.9, aspect=5)

  # Save
  plt.savefig('./im/surface_curvature3d.png', bbox_inches = 'tight',
    pad_inches = 0)

  # Show
  plt.show()
