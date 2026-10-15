// Load a GLIM global-mapping dump and export the globally-optimized merged cloud as binary PLY.
// Usage: glim_export_ply <dump_dir> <out.ply> <config_dir> [voxel_leaf]
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include <glim/mapping/global_mapping.hpp>
#include <glim/util/config.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/types/point_cloud.hpp>

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: glim_export_ply <dump_dir> <out.ply> <config_dir> [voxel_leaf]\n";
    return 1;
  }
  const std::string dump = argv[1];
  const std::string out_ply = argv[2];
  const std::string config_dir = argv[3];
  const double leaf = (argc > 4) ? std::stod(argv[4]) : 0.0;

  glim::GlobalConfig::instance(config_dir);

  glim::GlobalMappingParams params;
  params.enable_gpu = false;          // CPU build
  params.enable_optimization = false; // dump is already optimized; just load + export
  glim::GlobalMapping gm(params);

  if (!gm.load(dump)) {
    std::cerr << "failed to load dump: " << dump << "\n";
    return 1;
  }

  gtsam_points::PointCloud::Ptr cloud = gm.export_points();
  if (!cloud || cloud->size() == 0) {
    std::cerr << "exported cloud is empty\n";
    return 1;
  }
  std::cerr << "merged points: " << cloud->size() << "\n";

  if (leaf > 0.0) {
    cloud = gtsam_points::voxelgrid_sampling(cloud, leaf);
    std::cerr << "after voxelgrid(" << leaf << "m): " << cloud->size() << "\n";
  }

  const bool has_i = cloud->has_intensities();
  const size_t n = cloud->size();

  std::ofstream ofs(out_ply, std::ios::binary);
  ofs << "ply\n";
  ofs << "format binary_little_endian 1.0\n";
  ofs << "element vertex " << n << "\n";
  ofs << "property float x\nproperty float y\nproperty float z\n";
  if (has_i) ofs << "property float intensity\n";
  ofs << "end_header\n";

  for (size_t i = 0; i < n; i++) {
    float xyz[3] = {(float)cloud->points[i].x(), (float)cloud->points[i].y(), (float)cloud->points[i].z()};
    ofs.write(reinterpret_cast<const char*>(xyz), sizeof(xyz));
    if (has_i) {
      float in = (float)cloud->intensities[i];
      ofs.write(reinterpret_cast<const char*>(&in), sizeof(in));
    }
  }
  ofs.close();
  std::cerr << "wrote " << out_ply << " (" << n << " points)\n";
  return 0;
}
