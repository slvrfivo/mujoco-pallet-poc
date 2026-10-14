"""2단계: FK(정기구학) 실습.

로봇의 관절 몇 개를 천천히 움직이면서, 관절 각도로부터 계산한
손끝(TCP)의 (x, y, z) 좌표를 0.5초마다 터미널에 출력한다.

실행 (PowerShell, 프로젝트 폴더에서):
    .\\.venv\\Scripts\\python.exe fk_demo.py

뷰어 창을 닫으면 프로그램이 종료된다.
"""

import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
SCENE_XML = PROJECT_DIR / "models" / "franka_emika_panda" / "scene.xml"

# 손끝(TCP: 두 손가락 사이에서 물건을 집는 점)의 위치.
# 모델에 손끝 표시(site)가 없어서, "hand" 바디 기준으로 손가락 방향(z축)으로
# 0.1034m 떨어진 점을 손끝으로 정한다. (Franka 공식 사양의 TCP 거리)
FINGERTIP_OFFSET = np.array([0.0, 0.0, 0.1034])

# 움직일 관절: 이름 -> (진폭[rad], 주기[초])
# home 자세를 중심으로 사인파 모양으로 천천히 왕복한다.
MOTIONS = {
    "joint1": (0.6, 10.0),  # 베이스 회전: 팔 전체가 좌우로 돈다
    "joint2": (0.3, 8.0),   # 어깨: 팔이 앞뒤로 기울어진다
    "joint4": (0.4, 6.0),   # 팔꿈치: 팔이 굽혀졌다 펴진다
}

# 관절 한계에서 이만큼(rad, 약 5.7도) 안쪽까지만 움직이게 한다.
SAFETY_MARGIN = 0.1

PRINT_INTERVAL = 0.5  # 손끝 좌표 출력 간격(초, 시뮬레이션 시간 기준)


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


def make_motion_plan(model, home_ctrl):
    """관절별 움직임 계획을 만들고, 관절 한계 안에 있는지 확인한다."""
    plan = []
    print("움직일 관절과 안전 범위 확인 (단위: rad)")
    for joint_name, (amplitude, period) in MOTIONS.items():
        act_id = find_actuator_for_joint(model, joint_name)
        center = home_ctrl[act_id]
        low, high = model.actuator_ctrlrange[act_id]  # 모터 목표값으로 줄 수 있는 범위
        safe_low, safe_high = low + SAFETY_MARGIN, high - SAFETY_MARGIN

        move_low, move_high = center - amplitude, center + amplitude
        ok = safe_low <= move_low and move_high <= safe_high
        print(f"  {joint_name}: 움직임 [{move_low:+.2f}, {move_high:+.2f}] / "
              f"안전 범위 [{safe_low:+.2f}, {safe_high:+.2f}] -> {'OK' if ok else '범위 밖! 잘라서 사용'}")

        plan.append((joint_name, act_id, center, amplitude, period, safe_low, safe_high))
    return plan


def apply_motion(data, plan):
    """현재 시간에 맞는 관절 목표 각도를 모터에 넣는다."""
    for _, act_id, center, amplitude, period, safe_low, safe_high in plan:
        target = center + amplitude * math.sin(2 * math.pi * data.time / period)
        data.ctrl[act_id] = np.clip(target, safe_low, safe_high)  # 안전 범위 밖으로 못 나가게


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    # home 자세에서 시작 (run_panda.py와 같음)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    plan = make_motion_plan(model, data.ctrl.copy())

    # 화면은 초당 60번만 갱신 (저사양 PC 배려, run_panda.py와 같음)
    frame_time = 1.0 / 60
    steps_per_frame = max(1, round(frame_time / model.opt.timestep))
    next_print_time = 0.0

    print("\n시간[s] |  손끝 x     y      z  [m] | joint1 joint2 joint4 [rad]")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            frame_start = time.time()

            for _ in range(steps_per_frame):
                apply_motion(data, plan)      # 1) 관절 목표 각도 설정
                mujoco.mj_step(model, data)   # 2) 물리 시뮬레이션 (모터가 관절을 목표로 끌고 감)

            if data.time >= next_print_time:
                # 3) FK: 현재 관절 각도(qpos)로 모든 바디의 위치·방향을 계산
                mujoco.mj_kinematics(model, data)
                x, y, z = get_fingertip_position(data)
                j1, j2, j4 = (data.joint(name).qpos[0] for name in ("joint1", "joint2", "joint4"))
                print(f"{data.time:6.1f}  | {x:+.3f} {y:+.3f} {z:+.3f} | {j1:+.2f}  {j2:+.2f}  {j4:+.2f}")
                next_print_time += PRINT_INTERVAL

            viewer.sync()

            remaining = steps_per_frame * model.opt.timestep - (time.time() - frame_start)
            if remaining > 0:
                time.sleep(remaining)

    print("뷰어가 닫혀 프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
