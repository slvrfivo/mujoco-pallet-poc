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

# 손끝 정의, IK 함수, 접촉 판정, 중력 보상은 여러 단계가 함께 쓰므로 pallet_sim/robot.py에 있다.
from pallet_sim.robot import (ARM_JOINTS, SCENE_XML, count_robot_contacts, find_actuator_for_joint,
                              get_fingertip_position, gravity_compensated_ctrl, solve_ik)

# 손끝 목표 좌표 (x, y, z) [m]. home 자세의 손끝은 약 (0.555, 0.000, 0.521).
TARGETS = [
    np.array([0.50, 0.20, 0.45]),  # 왼쪽 앞, 조금 아래
    np.array([0.45, -0.20, 0.35]),  # 오른쪽, 더 아래
    np.array([0.60, 0.00, 0.30]),  # 정면 멀리, 낮게
    np.array([1.20, 0.00, 0.50]),  # 일부러 넣은 도달 불가능한 목표 (팔 길이 밖)
]

MOVE_TIME = 3.0      # 목표 자세까지 이동하는 시간 [s]
SETTLE_TIME = 1.0    # 이동 후 자리 잡을 때까지 기다리는 시간 [s]

# 뷰어에 표시할 목표 구슬 색 (R, G, B, 투명도)
COLOR_WAITING = [1.0, 0.8, 0.0, 0.6]  # 노랑: 대기
COLOR_ACTIVE = [0.1, 0.9, 0.1, 0.8]   # 초록: 이동 중
COLOR_DONE = [0.2, 0.4, 1.0, 0.8]     # 파랑: 도달
COLOR_FAILED = [1.0, 0.1, 0.1, 0.8]   # 빨강: 실패


# ---------------------------------------------------------------- IK 부분 --
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

            if count_robot_contacts(model, data) > 0:
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

    if viewer is not None:
        print("[화면 범례]")
        print("  물리 시뮬레이션: 로봇팔 움직임 (IK 결과를 모터 목표로 주고, 모터가 중력을 이기며 따라감)")
        print("  시각화 전용 (물리 영향 없음): 목표 구슬 노랑=대기 / 초록=이동 중 / 파랑=도달 / 빨강=실패")
        print(f"  목표마다 {MOVE_TIME:.0f}초 동안 천천히 이동 후 {SETTLE_TIME:.0f}초 정지")

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
