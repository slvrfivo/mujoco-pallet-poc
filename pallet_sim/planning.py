"""경로 계획 공통 도구: 이동 높이·속도, 여러 직선 구간을 6D IK로 계획하는 plan_legs.

5단계(pallet_reach_demo.py)에서 만든 것을 6~8단계에서도 함께 쓰도록 옮겨 둔 것이다. 내용은 원래와 같다.
"""

import numpy as np

from . import env, ik6d
from .robot import ARM_JOINTS, find_actuator_for_joint, gravity_compensated_ctrl

# 이동 높이: 가장 높게 쌓인 상자(4층) 윗면보다 10cm 위에서 구역 사이를 옮겨 다닌다.
SAFE_Z = env.PALLET_TOP_Z + env.MAX_LAYERS * env.BOX_SIZE + 0.10  # = 0.55 m

# 이동 속도 [m/s]. 관찰하기 쉽게 느리게 설정.
TRANSIT_SPEED = 0.10   # 이동 높이에서 수평 이동
VERTICAL_SPEED = 0.06  # 이동 높이 <-> 대기 위치
FINAL_SPEED = 0.03     # 대기 위치 <-> 놓기 목표 (가장 조심스러운 구간)
MIN_SEGMENT_TIME = 1.0

ZONE_ORDER = ["Z1", "Z2", "Z3", "Z4"]


def plan_legs(model, ik_data, base_qpos, start_q, start_pos, legs):
    """여러 직선 구간을 차례로 계획한다. legs = [(이름, 끝점 또는 None(정지), 속도 또는 정지시간)]

    각 구간마다 6D IK 경로, 중력 보상 모터 목표, 재생 시간, 최소 거리를 담아 돌려준다.
    하나라도 실패하면 (None, 실패 메시지).
    """
    arm_act_ids = [find_actuator_for_joint(model, n) for n in ARM_JOINTS]
    qpos_ids = model.jnt_qposadr[[model.joint(n).id for n in ARM_JOINTS]]
    segments = []
    q, pos = start_q, start_pos
    for name, end_pos, speed_or_time in legs:
        if end_pos is None:  # 제자리 정지 구간
            joint_path, tcp_path, duration, clearance = np.array([q]), np.array([pos]), speed_or_time, None
        else:
            seed = base_qpos.copy()
            seed[qpos_ids] = q
            ok, joint_path, tcp_path, message = ik6d.plan_line(model, ik_data, seed, pos, end_pos)
            if not ok:
                return None, f"[{name}] {message}"
            safe, clearance = ik6d.check_path_clearance(model, ik_data, base_qpos, joint_path)
            if not safe:
                return None, f"[{name}] 계획 경로에서 로봇이 물체에 닿음"
            duration = max(MIN_SEGMENT_TIME, np.linalg.norm(end_pos - pos) / speed_or_time)
            q, pos = joint_path[-1], end_pos

        ctrl_path = np.array([gravity_compensated_ctrl(model, ik_data, qa, arm_act_ids) for qa in joint_path])
        segments.append({"name": name, "joint_path": joint_path, "ctrl_path": ctrl_path,
                         "tcp_path": tcp_path, "duration": duration, "clearance": clearance})
    return segments, "성공"


def distance_to_path(point, path):
    """점에서 계획 경로(꺾은선)까지의 가장 짧은 거리. 늦게 따라오는 것과 옆으로 벗어난 것을 구분하기 위함."""
    if len(path) == 1:
        return float(np.linalg.norm(point - path[0]))
    a, b = path[:-1], path[1:]
    ab = b - a
    length2 = np.maximum((ab * ab).sum(axis=1), 1e-12)
    s = np.clip(((point - a) * ab).sum(axis=1) / length2, 0.0, 1.0)
    closest = a + s[:, None] * ab
    return float(np.min(np.linalg.norm(closest - point, axis=1)))
