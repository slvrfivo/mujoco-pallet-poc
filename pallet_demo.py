"""4단계: 2x2 팔레트 환경 데모.

실행 (PowerShell, 프로젝트 폴더에서):
    .\\.venv\\Scripts\\python.exe pallet_demo.py              # 뷰어 (구역별 상자 Z1~Z4 = 0 1 2 3개)
    .\\.venv\\Scripts\\python.exe pallet_demo.py 1 0 3 2      # 구역별 상자 개수를 직접 지정 (0~4)
    .\\.venv\\Scripts\\python.exe pallet_demo.py --headless   # 화면 없이 3초 동안 상자가 안정적인지 확인
    .\\.venv\\Scripts\\python.exe pallet_demo.py --check      # 기존 IK로 모든 적재 칸의 도달 가능성 검사

이번 단계에서는 로봇이 상자를 옮기지 않는다. 로봇은 home 자세로 서 있기만 한다.
"""

import math
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

import pallet_env as env
from ik_demo import count_robot_contacts, solve_ik

DEFAULT_COUNTS = [0, 1, 2, 3]  # Z1, Z2, Z3, Z4


def parse_counts(args):
    """명령줄의 숫자 4개를 {"Z1": n1, ...} 형태로 바꾼다. 없으면 기본값."""
    numbers = [a for a in args if not a.startswith("--")]
    if not numbers:
        numbers = DEFAULT_COUNTS
    if len(numbers) != len(env.ZONE_LAYOUT):
        raise SystemExit("구역별 상자 개수 4개를 입력하세요. 예) pallet_demo.py 0 1 2 3")
    return dict(zip(env.ZONE_LAYOUT, (int(n) for n in numbers)))


def make_scene(counts):
    """장면을 만들고 로봇을 home 자세로 둔다."""
    model = env.build_model(counts)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data


def hand_tilt_deg(data):
    """손이 수직 아래 방향에서 몇 도 기울었는지. (0도 = 손가락이 정확히 아래를 향함)"""
    hand_z_axis = data.body("hand").xmat.reshape(3, 3)[:, 2]
    return math.degrees(math.acos(np.clip(-hand_z_axis[2], -1.0, 1.0)))


def print_status(model, data):
    """구역별 측정 높이, 상자 개수, 다음 목표 좌표를 표로 출력한다."""
    heights = env.measure_zone_heights_ground_truth(model, data)  # 나중에 카메라 측정으로 교체할 부분
    counts = env.heights_to_counts(heights)

    print("구역 | 중심 (x, y)      | 상자 | 적재 높이 | 다음 놓기 목표 (x, y, z)  | 대기 목표 z")
    for zone in env.ZONE_LAYOUT:
        cx, cy = env.zone_center(zone)
        place = env.next_place_target(zone, counts[zone])
        if place is None:
            target_text = "가득 참 (더 놓을 수 없음)"
            approach_text = "-"
        else:
            target_text = f"({place[0]:.3f}, {place[1]:+.3f}, {place[2]:.3f})"
            approach_text = f"{env.approach_target(place)[2]:.3f}"
        print(f" {zone}  | ({cx:.3f}, {cy:+.3f}) | {counts[zone]:3d}개 | {heights[zone]:.3f} m  | "
              f"{target_text:25s} | {approach_text}")
    return counts


def draw_next_targets(viewer, counts):
    """다음 상자가 놓일 자리에 반투명 '유령 상자'를 그린다. (그림일 뿐, 물리에는 영향 없음)"""
    with viewer.lock():
        viewer.user_scn.ngeom = 0
        for zone in env.ZONE_LAYOUT:
            place = env.next_place_target(zone, counts[zone])
            if place is None:
                continue
            rgba = np.array(env.ZONE_COLORS[zone][:3] + [0.35], dtype=np.float32)
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[env.BOX_SIZE / 2] * 3,
                pos=place,
                mat=np.eye(3).flatten(),
                rgba=rgba,
            )
            viewer.user_scn.ngeom += 1


def box_positions(model, data):
    """모든 상자 바디의 현재 위치."""
    return {model.body(i).name: data.xpos[i].copy()
            for i in range(model.nbody) if model.body(i).name.startswith("box_")}


# ------------------------------------------------------------ 실행 모드 --
def run_viewer(counts):
    model, data = make_scene(counts)
    print_status(model, data)
    print("\n[화면 범례]")
    print("  물리 물체 (중력·충돌 있음): 로봇팔(흰색), 받침대(회색), 팔레트(나무색), 상자(골판지색, 층마다 명암 교대)")
    print("  시각화 전용 (물리 영향 없음): 구역 색판 Z1 빨강 / Z2 초록 / Z3 파랑 / Z4 노랑,")
    print("                                반투명 상자 = 다음 상자를 놓을 자리(계산된 목표)")
    print("  로봇: home 자세를 유지하며 움직이지 않음 (이번 단계는 환경만 구성)")
    print("\n뷰어 창을 직접 닫을 때까지 실행됩니다. (좌우 설정 패널: Tab / Shift+Tab)")

    steps_per_frame = max(1, round((1.0 / 60) / model.opt.timestep))
    # 작은 화면에서도 장면이 잘 보이도록 좌우 패널을 숨기고 시작한다.
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        # 카메라: 로봇과 팔레트가 함께 보이도록 오른쪽 앞 위에서 내려다본다.
        viewer.cam.lookat[:] = [0.40, 0.0, 0.30]
        viewer.cam.distance = 1.6
        viewer.cam.azimuth = 150
        viewer.cam.elevation = -30
        draw_next_targets(viewer, counts)
        while viewer.is_running():
            frame_start = time.time()
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            viewer.sync()
            remaining = steps_per_frame * model.opt.timestep - (time.time() - frame_start)
            if remaining > 0:
                time.sleep(remaining)


def settle_test(counts, seconds=3.0):
    """화면 없이 몇 초 시뮬레이션해서 상자가 제자리에 안정적으로 있는지 확인한다."""
    model, data = make_scene(counts)
    start = box_positions(model, data)
    robot_contact_steps = 0
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if count_robot_contacts(model, data) > 0:
            robot_contact_steps += 1

    end = box_positions(model, data)
    moved = max((np.linalg.norm(end[n] - start[n]) for n in start), default=0.0)
    print(f"{seconds:.0f}초 시뮬레이션 후")
    print(f"  상자 {len(start)}개 중 가장 많이 움직인 거리: {moved * 1000:.2f} mm")
    print(f"  로봇이 무언가에 닿은 스텝 수: {robot_contact_steps}")
    measured = print_status(model, data)
    print(f"  측정한 상자 개수가 처음 배치와 같음: {measured == counts}")


def check_reachability():
    """모든 구역 x 모든 층의 '놓기 목표'와 '대기 목표'에 기존 IK가 도달하는지 검사한다.

    검사할 때 장면 상태:
      - 검사하는 구역: 그 층 아래까지만 상자가 있음 (다음 상자를 놓기 직전 상태)
      - 나머지 구역 : 상자가 가득 참 (이웃 상자와 부딪히기 가장 쉬운 상태)
    IK는 home 자세에서 출발하고, 결과 자세가 충돌하면 실패로 본다.
    주의: 위치(x, y, z)만 맞춘 결과다. 손이 수직 아래를 향하는지는 보장하지 않는다.
    """
    print("구역 층 | 목표 종류 | 목표 z  | 결과 | 반복 | 계산 오차   | 손 기울기 | 메시지")
    total = success = 0
    for zone in env.ZONE_LAYOUT:
        for layer in range(env.MAX_LAYERS):
            counts = {z: env.MAX_LAYERS for z in env.ZONE_LAYOUT}
            counts[zone] = layer
            model, data = make_scene(counts)
            ik_data = mujoco.MjData(model)  # IK 전용 (실제 data와 분리)

            place = env.next_place_target(zone, layer)
            for kind, target in (("대기", env.approach_target(place)), ("놓기", place)):
                ok, q, iters, err, message = solve_ik(model, ik_data, data.qpos.copy(), target)
                mujoco.mj_kinematics(model, ik_data)
                total += 1
                success += ok
                print(f" {zone}  {layer}  | {kind}      | {target[2]:.3f} | {'OK  ' if ok else '실패'} | "
                      f"{iters:3d}  | {err * 1000:7.3f} mm | {hand_tilt_deg(ik_data):5.1f}도   | {message}")
    print(f"\n위치 IK 기준 충돌 없이 도달: {success}/{total}")
    print("※ 손 방향(수직 파지)은 아직 제어하지 않으므로, 실제로 상자를 잡고 놓을 수 있는지는 검증되지 않았습니다.")


def main():
    if "--check" in sys.argv:
        check_reachability()
        return

    counts = parse_counts(sys.argv[1:])
    if "--headless" in sys.argv:
        settle_test(counts)
    else:
        run_viewer(counts)


if __name__ == "__main__":
    main()
