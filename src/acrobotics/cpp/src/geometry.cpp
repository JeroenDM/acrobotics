#include "geometry.h"

#include <array>
#include <iostream>
#include <math.h>
#include <vector>

rmatrix rotation(double angle) {
  rmatrix R;
  R << cos(angle), -sin(angle), 0.0, sin(angle), cos(angle), 0.0, 0.0, 0.0, 1.0;
  return R;
}

Box::Box(double x_width, double y_width, double z_width) {
  dx = x_width;
  dy = y_width;
  dz = z_width;
  tf = Eigen::Affine3d::Identity();
  tolerance_ = 1e-6;
}

std::array<point, 8> Box::_get_vertices() {
  double a, b, c;
  a = dx / 2;
  b = dy / 2;
  c = dz / 2;

  std::array<point, 8> v;
  v[0] << tf * point(-a, b, c);
  v[1] << tf * point(-a, b, -c);
  v[2] << tf * point(-a, -b, c);
  v[3] << tf * point(-a, -b, -c);

  v[4] << tf * point(a, b, c);
  v[5] << tf * point(a, b, -c);
  v[6] << tf * point(a, -b, c);
  v[7] << tf * point(a, -b, -c);
  return v;
}

std::array<point, 6> Box::_get_normals() {
  rmatrix r;
  r = tf.rotation();
  std::array<point, 6> n;
  n[0] << r * point(1, 0, 0);
  n[1] << r * point(-1, 0, 0);
  n[2] << r * point(0, 1, 0);
  n[3] << r * point(0, -1, 0);
  n[4] << r * point(0, 0, 1);
  n[5] << r * point(0, 0, -1);
  return n;
}

std::array<vector6d, 12> Box::_get_edges() {
  std::array<point, 8> v = _get_vertices();
  std::array<vector6d, 12> e;
  e[0] << v[0], v[1];
  e[1] << v[1], v[3];
  e[2] << v[3], v[2];
  e[3] << v[2], v[0];

  e[4] << v[0], v[4];
  e[5] << v[1], v[5];
  e[6] << v[3], v[7];
  e[7] << v[2], v[6];

  e[8] << v[4], v[5];
  e[9] << v[5], v[7];
  e[10] << v[7], v[6];
  e[11] << v[6], v[4];
  return e;
}

bool Box::is_in_collision(Box other) {
  std::array<point, 8> v_a, v_b;
  v_a = _get_vertices();
  v_b = other._get_vertices();

  std::array<point, 6> n_a, n_b;
  n_a = _get_normals();
  n_b = other._get_normals();

  std::vector<point> normals;
  normals.push_back(n_a[0]);
  normals.push_back(n_a[2]);
  normals.push_back(n_a[4]);
  normals.push_back(n_b[0]);
  normals.push_back(n_b[2]);
  normals.push_back(n_b[4]);

  // project on the 6 main normals
  std::array<double, 8> p_a, p_b;
  double max_a, min_a, max_b, min_b;
  for (auto ni : normals) {

    p_a = _get_projection(ni, v_a);
    max_a = *std::max_element(p_a.begin(), p_a.end());
    min_a = *std::min_element(p_a.begin(), p_a.end());

    p_b = other._get_projection(ni, v_b);
    max_b = *std::max_element(p_b.begin(), p_b.end());
    min_b = *std::min_element(p_b.begin(), p_b.end());

    // if no overlap, then we found a plane
    // in between the two boxes and there is no collision
    if ((max_a + tolerance_ < min_b) or (min_a > max_b + tolerance_)) {
      return false;
    }
  }

  // project on the cross products
  point ni;
  for (int i = 0; i < 3; ++i) {
    for (int j = 3; j < 6; ++j) {
      ni = normals[i].cross(normals[j]);

      p_a = _get_projection(ni, v_a);
      max_a = *std::max_element(p_a.begin(), p_a.end());
      min_a = *std::min_element(p_a.begin(), p_a.end());

      p_b = other._get_projection(ni, v_b);
      max_b = *std::max_element(p_b.begin(), p_b.end());
      min_b = *std::min_element(p_b.begin(), p_b.end());

      // if no overlap, then we found a plane
      // in between the two boxes and there is no collision
      if ((max_a + tolerance_ < min_b) or (min_a > max_b + tolerance_)) {
        return false;
      }
    }
  }

  // we did not found a separating plane, so there is collision
  return true;
}

/** Point expressed in world frame (same as where tf of box is expressed in)
 *  v = tf * vr => vr = tf^(-1) * v
 */
bool Box::is_in_box(double x, double y, double z) {
  double a, b, c;
  a = dx / 2;
  b = dy / 2;
  c = dz / 2;

  Eigen::Vector3d v(x, y, z);
  // transform point to box reference frame
  // TODO inverse of matrix inefficient
  Eigen::Vector3d vr = tf.inverse() * v;
  if (vr[0] + tolerance_ < -a or vr[0] > a + tolerance_)
    return false;
  if (vr[1] + tolerance_ < -b or vr[1] > b + tolerance_)
    return false;
  if (vr[2] + tolerance_ < -c or vr[2] > c + tolerance_)
    return false;
  return true;
}

std::array<double, 8> Box::_get_projection(point axis,
                                           std::array<point, 8> vertices) {
  std::array<double, 8> projection;
  for (int i = 0; i < 8; ++i) {
    projection[i] = axis.dot(vertices[i]);
  }
  return projection;
}

void Box::get_vertices(double mat[8][3]) {
  std::array<point, 8> temp = _get_vertices();
  for (int i = 0; i < temp.size(); ++i) {
    mat[i][0] = temp[i][0];
    mat[i][1] = temp[i][1];
    mat[i][2] = temp[i][2];
  }
}

void Box::get_normals(double mat[6][3]) {
  std::array<point, 6> temp = _get_normals();
  for (int i = 0; i < temp.size(); ++i) {
    mat[i][0] = temp[i][0];
    mat[i][1] = temp[i][1];
    mat[i][2] = temp[i][2];
  }
}

void Box::get_edges(double mat[12][6]) {
  std::array<vector6d, 12> v = _get_edges();
  for (int i = 0; i < 12; ++i) {
    for (int j = 0; j < 6; ++j) {
      mat[i][j] = v[i][j];
    }
  }
}

void Box::set_transform(double *data, int nrows, int ncols) {
  int index;
  for (int i = 0; i < nrows; ++i) {
    for (int j = 0; j < ncols; ++j) {
      index = j + ncols * i;
      tf(i, j) = data[index];
    }
  }
}