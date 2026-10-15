# Extend the official GLIM image with a headless dump->PLY exporter built against the
# image's own (consistent) glim/gtsam_points/gtsam.
FROM koide3/glim_ros2:jazzy
COPY CMakeLists.txt glim_export_ply.cpp /tmp/glim_tools/
RUN set -ex; \
    . /opt/ros/jazzy/setup.sh; \
    . /root/ros2_ws/install/setup.sh; \
    cd /tmp/glim_tools; \
    cmake -DCMAKE_BUILD_TYPE=Release .; \
    make -j"$(nproc)"; \
    cp glim_export_ply /usr/local/bin/glim_export_ply
