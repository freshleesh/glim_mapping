# glim_mapping

F1TENTH (Livox Mid-360) **ros2 bag → GLIM 오프라인 맵** 루틴. GLIM은 전부 Docker로 돌아서 호스트에 GLIM/GTSAM 설치가 필요 없다.

결과물 (`<out_root>/<map_name>/`):
- `<map_name>.pcd` — 3D 맵 (kiss_icp_localization `map_pcd_3d`)
- `<map_name>.pgm/.yaml` — 2D 트랙 그리드 (map_server `/map` + kiss detection track mask)
- GLIM 중간산출물은 `.work/`에 캐시 (재실행 시 매핑 단계 건너뜀, `--force`로 재계산)

## 새 컴퓨터에서 1회 설정
```bash
git clone git@github.com:freshleesh/glim_mapping.git ~/glim_mapping
cd ~/glim_mapping && ./setup.sh      # docker, nvidia-container-toolkit(GPU 있으면), 이미지 pull/build
```
Ubuntu + (GPU 쓰려면) NVIDIA 드라이버가 이미 있어야 한다. GPU 이미지는 `koide3/glim_ros2:jazzy_cuda13.1` — 드라이버가 CUDA 13.1을 지원해야 함.

## 루틴
```bash
# 1) bag 녹화 (livox 토픽 보이는 곳에서). /livox/lidar + /livox/imu만 녹화
./record_bag.sh ~/bag/my_track            # 전부 녹화하려면 뒤에 -a

# 2) 설정 복사 후 map_name / bag_path 수정
cp map_config.example.yaml my_track.yaml

# 3) 맵 생성
./map.sh my_track.yaml
```
녹화 팁: 천천히, 루프를 최소 한 번 닫고 출발점에서 끝낸다.

## 구성
| 경로 | 내용 |
|---|---|
| `glim_map_pipeline.py` | 본체: GLIM 매핑 → merged PLY export → 궤적 기준 수평화 → pcd + 벽 밴드 → 트랙 마스크 |
| `config/gpu` | 기본 GLIM 설정 (GPU, VGICP_GPU + sub_mapping BA) |
| `config/gpu_lowmem` | GPU 메모리 부족 시 (voxel 굵게, 샘플링 0.5) |
| `config/gpu_loopclosure` | 명시적 pose-graph 루프클로저 백엔드 |
| `config/cpu` | GPU 없는 머신용 (`use_gpu: false`) |
| `tools/export_ply/` | GLIM dump → PLY exporter (`glim_tools:jazzy` 이미지, 첫 실행 시 자동 빌드) |
| `tools/eval_map.py` | 맵 품질 지표 (wall RMS, MME) + 렌더 |
| `tools/make_occgrid.py`, `pcd_to_html.py`, `remove_dynamic.py`, `decimate_lidar.py` | 보조 도구 |

## 센서 관련 핵심 설정 (Mid-360, 바꾸면 맵이 망가짐)
- IMU 가속도 단위가 **g** → `acc_scale: 9.80665`
- `T_lidar_imu = [0.011, 0.02329, -0.04412, 0,0,0,1]`
- 점별 시간 = sensor-boot 기준 절대 ns (float64) → `autoconf_perpoint_times:false, prefer_frame_time:true, relative_time:false, time_scale:1e-9`
- `points_time_offset: -0.09` (LiDAR-IMU 동기 스윕 결과 — 벽 산포 4.6→1.6 cm)
- 토픽: `/livox/lidar`, `/livox/imu` (`config_ros.json`)
