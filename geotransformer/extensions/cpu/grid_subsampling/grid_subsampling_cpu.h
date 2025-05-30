#pragma once

#include <vector>
#include <unordered_map>
#include "../../extra/cloud/cloud.h"
#include <cstdint> // for int64_t

class SampledData {
public:
  int count;
  PointXYZ point;

  SampledData() {
    count = 0;
    point = PointXYZ();
  }

  void update(const PointXYZ& p) {
    count += 1;
    point += p;
  }
};

void single_grid_subsampling_cpu(
  std::vector<PointXYZ>& o_points,
  std::vector<PointXYZ>& s_points,
  float voxel_size
);

// void grid_subsampling_cpu(
//   std::vector<PointXYZ>& o_points,
//   std::vector<PointXYZ>& s_points,
//   std::vector<long>& o_lengths,
//   std::vector<long>& s_lengths,
//   float voxel_size
// );

void grid_subsampling_cpu(
  std::vector<PointXYZ>& o_points,
  std::vector<PointXYZ>& s_points,
  std::vector<int64_t>& o_lengths,
  std::vector<int64_t>& s_lengths,
  float voxel_size
);