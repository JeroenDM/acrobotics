#ifndef _GEOMETRY_H_
#define _GEOMETRY_H_

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>

typedef Eigen::Transform<double, 3, Eigen::Affine> transform;
typedef Eigen::Matrix3d rmatrix;
typedef Eigen::Vector3d point;
typedef Eigen::Matrix<double, 6, 1> vector6d;

const double PI = 3.141592653589793238462643383279502884L;

rmatrix rotation(double angle);

class Box {
  double dx, dy, dz;
  Eigen::Affine3d tf;
  double tolerance_;

public:
  Box(double x_width, double y_width, double z_width);
  std::array<point, 8> _get_vertices();
  std::array<point, 6> _get_normals();
  std::array<vector6d, 12> _get_edges();
  std::array<double, 8> _get_projection(point axis,
                                        std::array<point, 8> vertices);
  bool is_in_collision(Box other);
  bool is_in_box(double x, double y, double z);
  void get_vertices(double mat[8][3]);
  void get_normals(double mat[6][3]);
  void get_edges(double mat[12][6]);
  void set_transform(double *data, int nrows, int ncols);
};

#endif