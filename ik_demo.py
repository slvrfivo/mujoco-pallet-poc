"""3단계: IK(역기구학) 실습.

손끝(TCP)의 목표 좌표 (x, y, z)를 주면, 자코비안 + DLS(Damped Least Squares)로
7개 팔 관절 각도를 계산하고, 시뮬레이션에서 로봇을 그 자세로 천천히 움직인다.

실행 (PowerShell, 프로젝트 폴더에서):
    .\\.venv\\Scripts\\python.exe ik_demo.py              # 뷰어와 함께 실행
    .\\.venv\\Scripts\\python.exe ik_demo.py --headless   # 화면 없이 빠르게 실행(테스트용)
"""

import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

# 손끝 정의와 안전 여유값은 FK 데모와 똑같은 것을 사용한다.
from fk_demo import SAFETY_MARGIN, SCENE_XML, find_actuator_for_joint, get_fingertip_position

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]  # 팔 관절 7개 (손가락 관절은 제외)

# 손끝 목표 좌표 (x, y, z) [m]. home 자세의 손끝은 약 (0.555, 0.000, 0.521).
TARGETS = [
    np.array([0.50, 0.20, 0.45]),  # 왼쪽 앞, 조금 아래
    np.array([0.45, -0.20, 0.35]),  # 오른쪽, 더 아래
    np.array([0.60, 0.00, 0.30]),  # 정면 멀리, 낮게
    np.array([1.20, 0.00, 0.50]),  # 일부러 넣은 도달 불가능한 목표 (팔 길이 밖)
]

# IK 설정
IK_TOLERANCE = 1e-4  # 계산상 오차가 0.1mm 이하이면 성공
IK_MAX_ITERS = 200   # 최대 반복 횟수
IK_DAMPING = 0.05    # DLS 감쇠값(λ). 클수록 안정적이지만 느리게 수렴
IK_MAX_STEP = 0.2    # 한 번 반복할 때 관절이 바뀌는 최대 크기 [rad]
MIN_TARGET_Z = 0.05  # 바닥(z=0)에 너무 가까운 목표는 거부 [m]

MOVE_TIME = 3.0      # 목표 자세까지 이동하는 시간 [s]
SETTLE_TIME = 1.0    # 이동 후 자리 잡을 때까지 기다리는 시간 [s]

# 뷰어에 표시할 목표 구슬 색 (R, G, B, 투명도)
COLOR_WAITING = [1.0, 0.8, 0.0, 0.6]  # 노랑: 대기
COLOR_ACTIVE = [0.1, 0.9, 0.1, 0.8]   # 초록: 이동 중
COLOR_DONE = [0.2, 0.4, 1.0, 0.8]     # 파랑: 도달
COLOR_FAILED = [1.0, 0.1, 0.1, 0.8]   # 빨강: 실패


# ---------------------------------------------------------------- IK 부분 --
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

    # 5) 계산된 자세가 바닥이나 자기 몸과 부딪히는지 확인
    mujoco.mj_fwdPosition(model, ik_data)
    if ik_data.ncon > 0:
        return False, q_arm, iters, error_norm, f"계산된 자세에서 충돌 {ik_data.ncon}건"

    return True, q_arm, iters, error_norm, "성공"


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


# ------------------------------------------------------- 시뮬레이션 부분 --
def draw_targets(viewer, colors):
    """목표 위치에 반투명 구슬을 그린다. (그림만 그릴 뿐 물리에는 영향 없음)"""
    if viewer is None:
        return
    with viewer.lock():
        viewer.user_scn.ngeom = 0
        for target, rgba in zip(TARGETS, colors):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0, 0],  # 반지름 2cm
                pos=target,
                mat=np.eye(3).flatten(),
                rgba=np.array(rgba, dtype=np.float32),
            )
            viewer.user_scn.ngeom += 1


def simulate(model, data, viewer, seconds, set_ctrl, stats):
    """seconds초 동안 시뮬레이션한다. 매 스텝마다 set_ctrl(경과시간)으로 모터 목표를 정한다.

    이동 중 충돌·관절 한계 위반이 있었는지 stats에 센다.
    뷰어가 닫히면 False를 반환한다.
    """
    steps_per_frame = max(1, round((1.0 / 60) / model.opt.timestep))  # 화면은 60FPS
    arm_jnt_ids = [model.joint(name).id for name in ARM_JOINTS]
    start_time = data.time

    while data.time - start_time < seconds:
        frame_start = time.time()
        for _ in range(steps_per_frame):
            set_ctrl(data.time - start_time)
            mujoco.mj_step(model, data)

            if data.ncon > 0:
                stats["collision_steps"] += 1
            q = data.qpos[model.jnt_qposadr[arm_jnt_ids]]
            if np.any(q < model.jnt_range[arm_jnt_ids, 0]) or np.any(q > model.jnt_range[arm_jnt_ids, 1]):
                stats["limit_steps"] += 1

        if viewer is not None:
            if not viewer.is_running():
                return False
            viewer.sync()
            remaining = steps_per_frame * model.opt.timestep - (time.time() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    return True


def run(viewer, model, data):
    ik_data = mujoco.MjData(model)  # IK 전용 계산 공간 (실제 시뮬레이션 data와 분리)
    arm_act_ids = [find_actuator_for_joint(model, name) for name in ARM_JOINTS]
    colors = [COLOR_WAITING] * len(TARGETS)
    draw_targets(viewer, colors)
    results = []

    for i, target in enumerate(TARGETS, start=1):
        print(f"\n[목표 {i}] 손끝 목표 = ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")

        ok, q_goal, iters, ik_error, message = solve_ik(model, ik_data, data.qpos.copy(), target)
        print(f"  IK: {message} | 반복 {iters}회 | 계산상 위치 오차 {ik_error * 1000:.3f} mm")
        if not ok:
            colors[i - 1] = COLOR_FAILED
            draw_targets(viewer, colors)
            print("  -> 이 목표는 건너뛰고 로봇을 움직이지 않습니다.")
            results.append((i, False, iters, ik_error, None))
            continue
        print("  관절 목표 [rad]: " + " ".join(f"{q:+.2f}" for q in q_goal))

        colors[i - 1] = COLOR_ACTIVE
        draw_targets(viewer, colors)

        # 현재 모터 목표에서 (중력 보상한) IK 결과까지 부드럽게(smoothstep) 바꿔 가며 이동
        ctrl_start = data.ctrl[arm_act_ids].copy()
        ctrl_goal = gravity_compensated_ctrl(model, ik_data, q_goal, arm_act_ids)

        def move(elapsed):
            s = min(elapsed / MOVE_TIME, 1.0)
            s = 3 * s**2 - 2 * s**3  # 천천히 출발해서 천천히 멈춤
            data.ctrl[arm_act_ids] = ctrl_start + s * (ctrl_goal - ctrl_start)

        stats = {"collision_steps": 0, "limit_steps": 0}
        if not simulate(model, data, viewer, MOVE_TIME + SETTLE_TIME, move, stats):
            print("뷰어가 닫혀 종료합니다.")
            return results

        # 실제 시뮬레이션 상태에서 손끝 위치를 다시 계산해 오차 측정
        mujoco.mj_kinematics(model, data)
        actual = get_fingertip_position(data)
        sim_error = np.linalg.norm(target - actual)
        print(f"  이동 후 실제 손끝 = ({actual[0]:.3f}, {actual[1]:.3f}, {actual[2]:.3f}) "
              f"| 실제 위치 오차 {sim_error * 1000:.2f} mm")
        print(f"  이동 중 충돌 {stats['collision_steps']}스텝, 관절 한계 위반 {stats['limit_steps']}스텝")

        colors[i - 1] = COLOR_DONE
        draw_targets(viewer, colors)
        results.append((i, True, iters, ik_error, sim_error))

    print("\n===== 요약 =====")
    for i, ok, iters, ik_error, sim_error in results:
        if ok:
            print(f"목표 {i}: 성공 | IK 반복 {iters}회 | 계산 오차 {ik_error * 1000:.3f} mm | 실제 오차 {sim_error * 1000:.2f} mm")
        else:
            print(f"목표 {i}: IK 실패 -> 이동하지 않음 | 반복 {iters}회 | 남은 오차 {ik_error * 1000:.1f} mm")
    return results


def main():
    headless = "--headless" in sys.argv

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)

    if headless:
        run(None, model, data)
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        run(viewer, model, data)
        print("\n모든 목표를 처리했습니다. 뷰어 창을 닫으면 종료합니다.")
        hold = lambda elapsed: None  # 마지막 자세 유지
        while simulate(model, data, viewer, 1.0, hold, {"collision_steps": 0, "limit_steps": 0}):
            pass


if __name__ == "__main__":
    main()
