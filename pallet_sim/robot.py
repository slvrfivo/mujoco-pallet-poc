"""로봇 공통 도구: 모델 경로, 손끝(TCP) 정의, FK·위치 IK, 접촉 판정, 중력 보상.

2단계(fk_demo.py)와 3단계(ik_demo.py)에서 만든 함수를 이후 단계에서도 함께 쓰도록 옮겨 둔 것이다.
내용은 원래 함수와 같다.
"""

from pathlib import Path

import mujoco
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]  # 저장소 최상위 폴더
SCENE_XML = PROJECT_DIR / "models" / "franka_emika_panda" / "scene.xml"

# 손끝(TCP: 두 손가락 사이에서 물건을 집는 점)의 위치.
# 모델에 손끝 표시(site)가 없어서, "hand" 바디 기준으로 손가락 방향(z축)으로
# 0.1034m 떨어진 점을 손끝으로 정한다. (Franka 공식 사양의 TCP 거리)
FINGERTIP_OFFSET = np.array([0.0, 0.0, 0.1034])

# 관절 한계에서 이만큼(rad, 약 5.7도) 안쪽까지만 움직이게 한다.
SAFETY_MARGIN = 0.1

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]  # 팔 관절 7개 (손가락 관절은 제외)

# IK 설정
IK_TOLERANCE = 1e-4  # 계산상 오차가 0.1mm 이하이면 성공
IK_MAX_ITERS = 200   # 최대 반복 횟수
IK_DAMPING = 0.05    # DLS 감쇠값(λ). 클수록 안정적이지만 느리게 수렴
IK_MAX_STEP = 0.2    # 한 번 반복할 때 관절이 바뀌는 최대 크기 [rad]
MIN_TARGET_Z = 0.05  # 바닥(z=0)에 너무 가까운 목표는 거부 [m]


# ---------------------------------------------------------------- FK --
def find_actuator_for_joint(model, joint_name):
    """주어진 관절을 움직이는 액추에이터(모터)의 번호를 찾는다."""
    joint_id = model.joint(joint_name).id
    for i in range(model.nu):
        if model.actuator_trnid[i, 0] == joint_id:
            return i
    raise ValueError(f"{joint_name}을 움직이는 액추에이터가 없습니다.")


def get_fingertip_position(data):
    """FK의 마지막 단계: hand 바디의 위치·방향으로 손끝의 월드 좌표를 구한다.

    hand_pos : hand 바디 원점의 월드 좌표 (x, y, z)
    hand_rot : hand 바디의 방향 (3x3 회전행렬)
    손끝 = hand 원점 + (hand 방향으로 돌린 오프셋)
    """
    hand = data.body("hand")
    hand_pos = hand.xpos
    hand_rot = hand.xmat.reshape(3, 3)
    return hand_pos + hand_rot @ FINGERTIP_OFFSET


# ---------------------------------------------------------- 위치 IK --
def get_fingertip_jacobian(model, data):
    """손끝 점의 위치 자코비안(3 x nv)을 계산한다.

    자코비안 J의 각 열 = "그 관절을 아주 조금 돌렸을 때 손끝이 움직이는 방향과 크기".
    mj_jac에 hand 바디 중심이 아니라 '손끝의 월드 좌표'를 넘겨서,
    hand에 붙어 있는 손끝 점 자체의 자코비안을 얻는다.
    """
    jacp = np.zeros((3, model.nv))
    fingertip = get_fingertip_position(data)
    mujoco.mj_jac(model, data, jacp, None, fingertip, model.body("hand").id)
    return jacp


def solve_ik(model, ik_data, start_qpos, target):
    """DLS 방식 IK. 실제 시뮬레이션과 분리된 ik_data 안에서만 계산한다.

    반환값: (성공 여부, 팔 관절 각도 7개, 반복 횟수, 계산상 오차[m], 메시지)
    """
    dof_ids = [model.joint(name).dofadr[0] for name in ARM_JOINTS]    # 자코비안 열 번호
    qpos_ids = [model.joint(name).qposadr[0] for name in ARM_JOINTS]  # qpos 안의 위치
    low = model.jnt_range[[model.joint(n).id for n in ARM_JOINTS], 0] + SAFETY_MARGIN
    high = model.jnt_range[[model.joint(n).id for n in ARM_JOINTS], 1] - SAFETY_MARGIN

    if target[2] < MIN_TARGET_Z:
        return False, None, 0, float("nan"), f"목표 z가 {MIN_TARGET_Z}m보다 낮아 바닥과 충돌 위험"

    ik_data.qpos[:] = start_qpos  # 현재 자세에서 출발
    iters = 0
    while True:
        # 1) FK: 지금 관절 각도에서 손끝 위치를 계산 (mj_jac에 필요한 값도 함께 계산)
        mujoco.mj_kinematics(model, ik_data)
        mujoco.mj_comPos(model, ik_data)
        error = target - get_fingertip_position(ik_data)  # 목표까지 남은 벡터
        error_norm = np.linalg.norm(error)

        if error_norm < IK_TOLERANCE or iters >= IK_MAX_ITERS:
            break

        # 2) 손끝 자코비안에서 팔 관절 7개 부분만 사용 (3 x 7)
        J = get_fingertip_jacobian(model, ik_data)[:, dof_ids]

        # 3) DLS: dq = Jᵀ (J Jᵀ + λ² I)⁻¹ error
        #    "손끝을 error만큼 옮기려면 관절을 얼마나 돌려야 하나"를 구하되,
        #    λ² I 덕분에 특이 자세(팔이 쭉 펴진 자세 등)에서도 값이 폭주하지 않는다.
        dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING**2 * np.eye(3), error)

        # 4) 한 번에 너무 크게 움직이지 않게 제한하고, 관절 한계 안으로 자른다.
        step_size = np.linalg.norm(dq)
        if step_size > IK_MAX_STEP:
            dq *= IK_MAX_STEP / step_size
        ik_data.qpos[qpos_ids] = np.clip(ik_data.qpos[qpos_ids] + dq, low, high)
        iters += 1

    q_arm = ik_data.qpos[qpos_ids].copy()
    if error_norm >= IK_TOLERANCE:
        return False, q_arm, iters, error_norm, "반복 횟수 안에 수렴하지 못함 (도달 불가능하거나 관절 한계에 막힘)"

    # 5) 계산된 자세가 바닥, 다른 물체, 자기 몸과 부딪히는지 확인
    mujoco.mj_fwdPosition(model, ik_data)
    robot_contacts = count_robot_contacts(model, ik_data)
    if robot_contacts > 0:
        return False, q_arm, iters, error_norm, f"계산된 자세에서 충돌 {robot_contacts}건"

    return True, q_arm, iters, error_norm, "성공"


def count_robot_contacts(model, data):
    """로봇 부품(link0과 그 아래 모든 바디)이 관련된 접촉 수를 센다.

    상자가 팔레트 위에 놓인 것처럼 로봇과 상관없는 접촉은 충돌로 세지 않는다.
    """
    robot_root = model.body("link0").id
    count = 0
    for i in range(data.ncon):
        contact = data.contact[i]
        body1 = model.geom_bodyid[contact.geom1]
        body2 = model.geom_bodyid[contact.geom2]
        # body_rootid: 그 바디가 속한 덩어리의 맨 위 바디. 로봇 부품이면 link0이 된다.
        if robot_root in (model.body_rootid[body1], model.body_rootid[body2]):
            count += 1
    return count


def gravity_compensated_ctrl(model, ik_data, q_goal, arm_act_ids):
    """중력으로 처지는 만큼 모터 목표를 미리 높여 준다 (중력 보상).

    Panda 모터는 '힘 = kp x (목표각 - 현재각)'처럼 동작한다. 팔이 멈춰 있으려면
    이 힘이 중력 토크와 같아야 하므로, 목표각을 그대로 주면 중력토크/kp 만큼 처진다.
    그래서 목표 자세에서의 중력 토크를 계산해, 그만큼 목표각에 더해 준다.
    """
    qpos_ids = [model.joint(name).qposadr[0] for name in ARM_JOINTS]
    dof_ids = [model.joint(name).dofadr[0] for name in ARM_JOINTS]

    ik_data.qpos[qpos_ids] = q_goal
    ik_data.qvel[:] = 0                  # 멈춰 있을 때의 힘(=중력 토크)만 계산
    mujoco.mj_forward(model, ik_data)    # 실제 시뮬레이션 data는 건드리지 않음
    gravity_torque = ik_data.qfrc_bias[dof_ids]
    kp = model.actuator_gainprm[arm_act_ids, 0]  # 각 모터의 강성
    return q_goal + gravity_torque / kp
