"""4단계: 2x2 팔레트와 상자 환경.

이 파일은 '환경'만 담당한다. (로봇을 움직이는 코드는 없음)
  1) 장면 만들기   : Panda 장면 + 팔레트 + 구역 표시 + 상자
  2) 상태 측정     : 구역별 적재 높이 -> 상자 개수
                     (지금은 시뮬레이션 정답을 읽지만, 나중에 카메라 측정으로 바꿀 자리)
  3) 목표 좌표 계산: 상자 개수 -> 다음 상자를 놓을 위치

원본 Panda 모델 파일은 수정하지 않고, MjSpec으로 불러와 그 위에 물체를 추가한다.
"""

import mujoco
import numpy as np

from .robot import SCENE_XML

# ------------------------------------------------------------- 크기 설정 --
# 단위는 모두 m, kg. 좌표계: 로봇 받침(link0)이 원점, 로봇 정면이 +x, 왼쪽이 +y, 위가 +z.

# 팔레트: 로봇 정면 0.5m 앞, 받침대 위에 놓인 정사각형 판.
# 위치는 6D IK(손가락 수직 아래) 검사로 정했다 (5단계).
# 중심을 0.60에 두면 먼 줄(x=0.68)에서 손을 수직으로 유지한 채 닿는 최대 높이가
# 약 0.52m라서, 이동 높이(0.55m)와 최상단 대기 위치(0.525m)에 닿지 못했다.
# 0.50에서는 먼 줄(x=0.58) 최대 높이가 약 0.64m로 이동 높이보다 9cm 여유가 있다.
PALLET_CENTER_XY = np.array([0.50, 0.00])
PALLET_SIZE = 0.32       # 한 변 길이 (2x2 구역이므로 구역 한 변은 0.16)
STAND_HEIGHT = 0.20      # 받침대 높이
PALLET_THICKNESS = 0.05  # 팔레트 판 두께
PALLET_TOP_Z = STAND_HEIGHT + PALLET_THICKNESS  # 바닥에서 팔레트 윗면까지 = 0.25

# 상자: 한 변 5cm 정육면체.
# Panda 그리퍼는 최대 8cm까지 벌어지므로, 5cm면 양쪽에 1.5cm씩 여유가 있다.
BOX_SIZE = 0.05
BOX_MASS = 0.1         # 100g (Panda 최대 하중 3kg보다 훨씬 가벼움)
MAX_LAYERS = 4         # 구역마다 최대 적재 층수 (높이 20cm)

# 상자를 놓기 전에 위에서 내려오기 위한 대기 높이 (놓을 위치보다 이만큼 위)
APPROACH_CLEARANCE = 0.10

# 구역 이름과 색. 로봇에서 팔레트를 바라볼 때 기준.
ZONE_SIZE = PALLET_SIZE / 2
ZONE_LAYOUT = {
    #       (x 방향: -1 가까운 줄 / +1 먼 줄, y 방향: +1 왼쪽 / -1 오른쪽)
    "Z1": (-1, +1),  # 가까운 줄, 왼쪽
    "Z2": (-1, -1),  # 가까운 줄, 오른쪽
    "Z3": (+1, +1),  # 먼 줄, 왼쪽
    "Z4": (+1, -1),  # 먼 줄, 오른쪽
}
ZONE_COLORS = {
    "Z1": [0.90, 0.25, 0.25, 1],  # 빨강
    "Z2": [0.25, 0.70, 0.30, 1],  # 초록
    "Z3": [0.25, 0.45, 0.90, 1],  # 파랑
    "Z4": [0.95, 0.80, 0.20, 1],  # 노랑
}
PALLET_COLOR = [0.55, 0.40, 0.25, 1]  # 나무색
STAND_COLOR = [0.35, 0.35, 0.38, 1]   # 회색 받침대
BOX_COLORS = [[0.80, 0.62, 0.40, 1], [0.70, 0.52, 0.32, 1]]  # 골판지색 2가지 (층마다 번갈아)


def zone_center(zone):
    """구역 중심의 (x, y) 좌표."""
    sx, sy = ZONE_LAYOUT[zone]
    return PALLET_CENTER_XY + np.array([sx, sy]) * ZONE_SIZE / 2


# ------------------------------------------------------------ 장면 만들기 --
def build_model(box_counts):
    """Panda 장면에 팔레트, 구역 표시, 상자를 추가해 MjModel을 만든다.

    box_counts: 구역별 처음에 쌓아 둘 상자 개수. 예) {"Z1": 0, "Z2": 1, "Z3": 2, "Z4": 3}
    """
    return build_spec(box_counts).compile()


def build_spec(box_counts):
    """build_model과 같은 장면을 컴파일 전 상태(MjSpec)로 돌려준다. 다른 물체를 더 붙일 때 사용."""
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    world = spec.worldbody
    BOX = mujoco.mjtGeom.mjGEOM_BOX

    # 받침대와 팔레트: 움직이지 않는 물체이므로 world에 바로 붙인다. (size는 '절반 길이')
    world.add_geom(
        name="stand", type=BOX,
        size=[PALLET_SIZE / 2 - 0.02, PALLET_SIZE / 2 - 0.02, STAND_HEIGHT / 2],
        pos=[*PALLET_CENTER_XY, STAND_HEIGHT / 2],
        rgba=STAND_COLOR,
    )
    world.add_geom(
        name="pallet", type=BOX,
        size=[PALLET_SIZE / 2, PALLET_SIZE / 2, PALLET_THICKNESS / 2],
        pos=[*PALLET_CENTER_XY, STAND_HEIGHT + PALLET_THICKNESS / 2],
        rgba=PALLET_COLOR,
    )

    for zone, count in box_counts.items():
        if not 0 <= count <= MAX_LAYERS:
            raise ValueError(f"{zone}의 상자 개수 {count}는 0~{MAX_LAYERS} 사이여야 합니다.")
        cx, cy = zone_center(zone)

        # 구역 표시: 팔레트 위의 얇은 색 판. 그림만 있고 충돌은 없다(contype/conaffinity=0).
        # 구역보다 조금 작게 만들어 구역 사이에 나무색 경계선이 보이게 한다.
        world.add_geom(
            name=f"zone_{zone}", type=BOX,
            size=[ZONE_SIZE / 2 - 0.008, ZONE_SIZE / 2 - 0.008, 0.0005],
            pos=[cx, cy, PALLET_TOP_Z + 0.0005],
            rgba=ZONE_COLORS[zone], contype=0, conaffinity=0,
        )

        # 상자: 떨어지거나 밀릴 수 있도록 자유 관절(freejoint)을 가진 바디로 만든다.
        for layer in range(count):
            box = world.add_body(name=f"box_{zone}_{layer}", pos=[cx, cy, box_center_z(layer)])
            box.add_freejoint()
            box.add_geom(type=BOX, size=[BOX_SIZE / 2] * 3, mass=BOX_MASS,
                         rgba=BOX_COLORS[layer % 2])

    return spec


# -------------------------------------------- 상태 측정 (나중에 센서로 교체) --
def measure_zone_heights_ground_truth(model, data):
    """구역별 적재 높이[m]를 시뮬레이션 정답(상자 실제 위치)으로 측정한다.

    나중에 상단 카메라가 생기면, 같은 형식({"Z1": 높이, ...})을 돌려주는
    카메라 측정 함수로 이 함수만 바꾸면 된다.
    """
    heights = {zone: 0.0 for zone in ZONE_LAYOUT}
    for body_id in range(model.nbody):
        # 상자 = 자유 관절(free joint)을 가진 물체. 팔레트에 미리 쌓은 상자와 로봇이 옮겨 놓은 상자를 모두 센다.
        # (로봇이 구역 위에서 들고 있는 상자도 높이로 잡히므로, 놓은 뒤에 측정해야 한다.)
        if model.body_jntnum[body_id] != 1 or model.jnt_type[model.body_jntadr[body_id]] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        x, y, z = data.xpos[body_id]
        for zone in ZONE_LAYOUT:
            cx, cy = zone_center(zone)
            if abs(x - cx) < ZONE_SIZE / 2 and abs(y - cy) < ZONE_SIZE / 2:
                top = z + BOX_SIZE / 2 - PALLET_TOP_Z  # 상자 윗면의 팔레트 기준 높이
                heights[zone] = max(heights[zone], top)
    return heights


def heights_to_counts(heights):
    """적재 높이를 상자 개수로 바꾼다. (높이 / 상자 한 변, 반올림)"""
    return {zone: int(round(h / BOX_SIZE)) for zone, h in heights.items()}


# ---------------------------------------------------------- 목표 좌표 계산 --
def box_center_z(layer):
    """layer번째 층(0부터)에 놓인 상자 중심의 z 좌표."""
    return PALLET_TOP_Z + layer * BOX_SIZE + BOX_SIZE / 2


def stack_height(count):
    """상자가 count개 쌓였을 때 팔레트 윗면 기준 높이."""
    return count * BOX_SIZE


def next_place_target(zone, count):
    """다음 상자를 놓을 때 '상자 중심'(=손끝 목표)이 와야 할 좌표. 꽉 찼으면 None.

    손끝(TCP)이 상자 중심을 잡는다고 보고, 손끝 목표 = 상자 중심 좌표로 둔다.
    """
    if count >= MAX_LAYERS:
        return None
    cx, cy = zone_center(zone)
    return np.array([cx, cy, box_center_z(count)])


def approach_target(place_target):
    """놓을 위치 바로 위의 대기 위치 (위에서 수직으로 내려오기 위한 점)."""
    return place_target + np.array([0.0, 0.0, APPROACH_CLEARANCE])


# ------------------------------------------------------ 장면·명령줄 도구 --
DEFAULT_COUNTS = [0, 1, 2, 3]  # Z1, Z2, Z3, Z4


def parse_counts(args):
    """명령줄의 숫자 4개를 {"Z1": n1, ...} 형태로 바꾼다. 없으면 기본값."""
    numbers = [a for a in args if not a.startswith("--")]
    if not numbers:
        numbers = DEFAULT_COUNTS
    if len(numbers) != len(ZONE_LAYOUT):
        raise SystemExit("구역별 상자 개수 4개를 입력하세요. 예) pallet_demo.py 0 1 2 3")
    return dict(zip(ZONE_LAYOUT, (int(n) for n in numbers)))


def make_scene(counts):
    """장면을 만들고 로봇을 home 자세로 둔다."""
    model = build_model(counts)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data
