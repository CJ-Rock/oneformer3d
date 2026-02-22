# Synthetic Rock-Pile + Circular LiDAR Dataset Generation

아래 스크립트로 다양한 형태의 암석 더미를 생성하고, 공중의 원형 궤적에서 360도 LiDAR 스캔을 수행해 점군 데이터를 만들 수 있습니다.

## 스크립트

- `tools/generate_synthetic_rockpile_lidar.py`

## 요청사항 반영 기본값

기본값으로 바로 아래 조건을 만족하도록 구성했습니다.

- 한 더미당 암석 수: **200개** (`--num-rocks 200`)
- 한 더미당 전체 점 개수: **약 10만~20만** (`200 x 500~1000`)
- 각 암석 OBB 특성 기록: `obb_x`, `obb_y`, `obb_z`, `obb_volume`
- 자연스러운 적층: 큰 암석부터 배치 + 중심부 채움 편향 + 코어 전용 채움 단계 + 중력 낙하(기본 10cm) + 3층 적층

## 예시: 500개 장면 생성

```bash
python tools/generate_synthetic_rockpile_lidar.py \
  --output-dir data/synth_rockpile_500 \
  --num-scenes 500 \
  --num-views 24 \
  --scan-radius 4.0 \
  --scan-height 2.0 \
  --export-ply
```

## 출력 구조

각 scene 폴더(`scene_0000`, `scene_0001`, ...)에 다음 파일이 저장됩니다.

- `scene_points.npy`: 전체 암석 더미 점군 `(N, 3)`
- `scene_instance_ids.npy`: 각 점의 암석 인스턴스 ID `(N,)`
- `lidar_points.npy`: LiDAR 스캔으로 취득된 점군 `(M, 3)`
- `lidar_view_ids.npy`: 각 LiDAR 점의 스캔 view 인덱스 `(M,)`
- `meta.json`: 장면별 메타데이터 (암석별 OBB/부피 포함)

`--export-ply`를 켜면 시각화를 위한 파일도 추가됩니다.

- `scene_instances.ply`: 암석 인스턴스별 색상 점군 (더미가 어떻게 쌓였는지 확인)
- `lidar_scan.ply`: view별 색상 LiDAR 점군
- `scan_trajectory.xyz`: 공중 원형 스캔 궤적(센서 위치)

CloudCompare / MeshLab / Open3D 등으로 `scene_instances.ply`를 열면 쌓임 형태를 직관적으로 볼 수 있습니다.

## 핵심 파라미터

- 암석 개수/밀도
  - `--num-rocks` (기본 200)
  - `--min-pts-per-rock`, `--max-pts-per-rock` (기본 500~1000)
  - `--min-total-points`, `--max-total-points` (기본 100000~200000)
- 더미 자연스러움
  - `--pile-radius`, `--heightmap-res`, `--target-peak-height`, `--drop-height`
  - `--core-fill-ratio`, `--core-radius-ratio`, `--num-layers` (기본 3)
  - `--upper-layer-spread`, `--top-center-penalty`
- 원형 LiDAR 스캔 궤적
  - `--num-views`: 원 궤적 위 센서 위치 수
  - `--scan-radius`: 센서 원 궤적 반지름
  - `--scan-height`: 센서 높이
- LiDAR 해상도/FOV
  - `--lidar-azimuth-bins`, `--lidar-elevation-bins`
  - `--lidar-elev-min`, `--lidar-elev-max`
  - `--lidar-max-range`
- 시각화 파일 출력
  - `--export-ply`

필요하면 `--seed`로 재현 가능한 데이터셋 생성이 가능합니다.


## 암석이 공중에 떠보이는 이유와 개선

기존 방식은 충돌 높이를 `max(local_surface - rock_z)`로 계산해서, 높이맵의 단일 스파이크(희소 점) 하나만 있어도 암석 전체가 위로 들릴 수 있었습니다.

현재는 아래처럼 개선했습니다.

- `--hmap-smooth-passes`(기본 1): 지지 높이맵을 3x3 평균으로 완만화
- `--collision-quantile`(기본 0.98): `max` 대신 상위 분위수 기반 충돌 높이 사용

이렇게 하면 침투는 억제하면서도 단일 이상치로 인한 부자연스러운 공중 부양이 크게 줄어듭니다.


## 상층에서 중앙만 높아지는 이유와 개선

원인은 보통 상층에서 반경이 과도하게 줄고(core 집중 + 작은 반경), top layer에서도 중심 채움이 계속되면서 중앙 기둥 형태가 생기기 때문입니다.

현재는 다음을 적용했습니다.

- top layer에서는 코어 전용 채움을 제한
- `--upper-layer-spread`(기본 0.80)로 상층 반경 축소를 완화
- `--top-center-penalty`(기본 0.20)로 상층의 과도한 중심 배치를 억제


## LiDAR 스캔 방식 (원 궤도, 일정 높이, 균일 커버리지)

현재 스크립트는 요청하신 방식대로 다음 절차로 스캔합니다.

1. `scan_height` 고정 높이에서, `scan_radius` 반지름의 원 궤도 위에 센서를 둡니다.
2. `num_views` 개 위치를 원 둘레에 **균일 간격**으로 배치합니다.
3. 각 위치에서 360°(azimuth) LiDAR를 수행하고, elevation 범위 내 최근접 hit를 취득합니다.
4. view별 점 수 편차를 줄이기 위해 `--max-points-per-view`로 상한을 적용해 균일도를 맞춥니다.

관련 파라미터:
- `--num-views`, `--scan-radius`, `--scan-height`
- `--random-scan-phase` (scene마다 시작 각도 랜덤)
- `--max-points-per-view` (기본 6000, <=0 이면 비활성화)
