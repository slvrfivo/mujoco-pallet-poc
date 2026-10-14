"""5단계: 그리퍼 방향까지 맞추는 6D IK와 경로 계획 도구.

ik_demo.py의 위치(3D) IK를 확장해서, 손끝 TCP의
  - 위치 (x, y, z)            : 3개
  - 방향 (어느 쪽을 보고 있나) : 3개
총 6개 오차를 함께 0으로 만든다.

손끝 정의, 로봇 충돌 판정, 중력 보상은 기존 파일의 함수를 그대로 사용한다.
"""

import math

import mujoco
import numpy as np

from fk_demo import SAFETY_MARGIN, get_fingertip_position
from ik_demo import ARM_JOINTS, count_robot_contacts

# 목표 그리퍼 방향 (3x3 회전행렬, 각 열 = hand의 x, y, z축이 월드에서 가리키는 방향).
# 모델 확인 결과 hand z축 = 손가락이 뻗은 방향, hand y축 = 손가락이 벌어지는 방향.
#   hand z축 -> 월드 (0, 0, -1) : 손가락이 수직 아래를 향함
#   hand y축 -> 월드 (1, 0, 0)  : 손가락이 x 방향으로 벌어짐 (상자의 앞/뒤 면을 잡는 방향)
#   hand x축 -> 월드 (0, 1, 0)  : 위 두 축에서 자동으로 정해짐
# home 자세의 손 방향과 같아서 손목을 크게 돌릴 필요가 없다.
GRIPPER_DOWN = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
])

POS_TOLERANCE = 1e-4                 # 위치 오차 0.1 mm 이하
ROT_TOLERANCE = math.radians(0.1)    # 방향 오차 0.1도 이하
MAX_ITERS = 300
DAMPING = 0.05
MAX_STEP = 0.2                       # 1회 반복당 관절 변화 최대 [rad]
ROT_WEIGHT = 0.3                     # 방향 오차 1 rad를 위치 오차 몇 m만큼 중요하게 볼지


def get_fingertip_pose(data):
    """손끝의 (위치, 방향 회전행렬). 손끝은 hand에서 z축으로만 떨어져 있어 방향은 hand와 같다."""
    return get_fingertip_position(data), data.body("hand").xmat.reshape(3, 3).copy()


def orientation_error(target_rot, current_rot):
    """현재 방향을 목표 방향으로 돌리는 회전을 '회전축 x 각도' 벡터(월드 기준, rad)로 구한다."""
    error_rot = target_rot @ current_rot.T   # 현재 -> 목표로 가는 회전
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, error_rot.flatten())
    rotvec = np.zeros(3)
    mujoco.mju_quat2Vel(rotvec, quat, 1.0)   # 쿼터니언 -> 회전 벡터
    return rotvec


def angle_between_deg(target_rot, current_rot):
    """두 방향 사이의 각도 차이 [도]."""
    return math.degrees(np.linalg.norm(orientation_error(target_rot, current_rot)))


def solve_ik_6d(model, ik_data, start_qpos, target_pos, target_rot=GRIPPER_DOWN):
    """6D DLS IK. 별도의 ik_data 안에서만 계산한다 (실제 시뮬레이션과 분리).

    반환값: (성공 여부, 팔 관절 7개, 반복 횟수, 위치 오차[m], 방향 오차[도], 메시지)
    """
    joint_ids = [model.joint(n).id for n in ARM_JOINTS]
    dof_ids = model.jnt_dofadr[joint_ids]
    qpos_ids = model.jnt_qposadr[joint_ids]
    low = model.jnt_range[joint_ids, 0] + SAFETY_MARGIN
    high = model.jnt_range[joint_ids, 1] - SAFETY_MARGIN
    hand_id = model.body("hand").id

    ik_data.qpos[:] = start_qpos
    iters = 0
    while True:
        mujoco.mj_kinematics(model, ik_data)
        mujoco.mj_comPos(model, ik_data)
        pos, rot = get_fingertip_pose(ik_data)
        pos_error = target_pos - pos
        rot_error = orientation_error(target_rot, rot)

        if (np.linalg.norm(pos_error) < POS_TOLERANCE and np.linalg.norm(rot_error) < ROT_TOLERANCE) \
                or iters >= MAX_ITERS:
            break

        # 위치 자코비안(jacp)과 회전 자코비안(jacr)을 손끝 점에서 함께 구한다.
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, ik_data, jacp, jacr, pos, hand_id)

        # 6행짜리 오차와 자코비안. 방향 쪽에는 가중치를 곱해 위치와 균형을 맞춘다.
        error = np.concatenate([pos_error, ROT_WEIGHT * rot_error])
        J = np.vstack([jacp[:, dof_ids], ROT_WEIGHT * jacr[:, dof_ids]])  # 6 x 7

        # DLS: dq = Jᵀ (J Jᵀ + λ² I)⁻¹ error  (3D 버전과 같은 식, 크기만 6x6)
        dq = J.T @ np.linalg.solve(J @ J.T + DAMPING**2 * np.eye(6), error)
        step = np.linalg.norm(dq)
        if step > MAX_STEP:
            dq *= MAX_STEP / step
        ik_data.qpos[qpos_ids] = np.clip(ik_data.qpos[qpos_ids] + dq, low, high)
        iters += 1

    q_arm = ik_data.qpos[qpos_ids].copy()
    pos_err = float(np.linalg.norm(pos_error))
    rot_err_deg = math.degrees(np.linalg.norm(rot_error))
    if pos_err >= POS_TOLERANCE or rot_err_deg >= math.degrees(ROT_TOLERANCE):
        return False, q_arm, iters, pos_err, rot_err_deg, "수렴 실패 (도달 불가능하거나 관절 한계에 막힘)"

    mujoco.mj_fwdPosition(model, ik_data)
    contacts = count_robot_contacts(model, ik_data)
    if contacts > 0:
        return False, q_arm, iters, pos_err, rot_err_deg, f"계산된 자세에서 로봇 충돌 {contacts}건"
    return True, q_arm, iters, pos_err, rot_err_deg, "성공"


# ------------------------------------------------------------- 거리 검사 --
def collision_geom_groups(model):
    """(로봇 충돌 형상 목록, 주변 물체 충돌 형상 목록). 그림 전용 형상(충돌 없음)은 뺀다."""
    robot_root = model.body("link0").id
    robot, others = [], []
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue  # 시각화 전용
        if model.geom(g).name == "floor":
            continue  # 바닥은 높이 제한(목표 z)으로 따로 막는다
        if model.body_rootid[model.geom_bodyid[g]] == robot_root:
            robot.append(g)
        else:
            others.append(g)
    return robot, others


def min_clearance(model, data, robot_geoms, other_geoms, distmax=0.05):
    """로봇과 주변 물체(팔레트·받침대·상자) 사이의 가장 가까운 거리 [m]. distmax보다 멀면 distmax."""
    fromto = np.zeros(6)
    best = distmax
    for g1 in robot_geoms:
        for g2 in other_geoms:
            best = min(best, mujoco.mj_geomDistance(model, data, g1, g2, distmax, fromto))
    return best


# ------------------------------------------------------------- 경로 계획 --
def plan_line(model, ik_data, start_qpos, start_pos, end_pos, target_rot=GRIPPER_DOWN, spacing=0.005):
    """start_pos -> end_pos 직선을 spacing 간격으로 나눠, 각 점의 6D IK 해를 차례로 구한다.

    앞 점의 해에서 다음 점을 풀기 때문에 관절이 갑자기 튀지 않는다.
    반환값: (성공 여부, 관절 경로 [N x 7], 손끝 위치 경로 [N x 3], 메시지)
    """
    qpos_ids = model.jnt_qposadr[[model.joint(n).id for n in ARM_JOINTS]]
    n_points = max(2, int(math.ceil(np.linalg.norm(end_pos - start_pos) / spacing)) + 1)
    qpos = start_qpos.copy()
    joint_path, tcp_path = [], []
    for s in np.linspace(0.0, 1.0, n_points):
        point = start_pos + s * (end_pos - start_pos)
        ok, q_arm, iters, pos_err, rot_err, message = solve_ik_6d(model, ik_data, qpos, point, target_rot)
        if not ok:
            return False, None, None, f"경로 점 ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) 에서 {message}"
        if joint_path and np.max(np.abs(q_arm - joint_path[-1])) > 0.2:
            return False, None, None, "경로 중 관절이 갑자기 크게 바뀜 (자세가 뒤집힐 위험)"
        joint_path.append(q_arm)
        tcp_path.append(point)
        qpos[qpos_ids] = q_arm
    return True, np.array(joint_path), np.array(tcp_path), "성공"


def check_path_clearance(model, ik_data, base_qpos, joint_path):
    """계획된 관절 경로의 모든 점에서 로봇 접촉 수와 최소 거리를 확인한다."""
    qpos_ids = model.jnt_qposadr[[model.joint(n).id for n in ARM_JOINTS]]
    robot_geoms, other_geoms = collision_geom_groups(model)
    worst = np.inf
    for q_arm in joint_path:
        ik_data.qpos[:] = base_qpos
        ik_data.qpos[qpos_ids] = q_arm
        mujoco.mj_fwdPosition(model, ik_data)
        if count_robot_contacts(model, ik_data) > 0:
            return False, 0.0
        worst = min(worst, min_clearance(model, ik_data, robot_geoms, other_geoms))
    return True, worst
