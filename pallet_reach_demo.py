"""5단계: 6D IK로 팔레트 Z1 -> Z2 -> Z3 -> Z4 목표에 수직으로 접근하는 데모.

상자를 잡거나 옮기지는 않는다. 로봇이 각 구역의 '다음 상자 놓을 자리'까지
올바른 방향(손가락이 수직 아래)으로 안전하게 내려갔다 올라오는지만 확인한다.

실행 (PowerShell, 프로젝트 폴더에서):
    .\\.venv\\Scripts\\python.exe pallet_reach_demo.py              # 뷰어, 창을 닫을 때까지 반복
    .\\.venv\\Scripts\\python.exe pallet_reach_demo.py --once       # 뷰어, 한 번만 돌고 정지 상태 유지
    .\\.venv\\Scripts\\python.exe pallet_reach_demo.py 1 0 3 2      # 구역별 상자 개수 지정 (기본 0 1 2 3)
    .\\.venv\\Scripts\\python.exe pallet_reach_demo.py --headless   # 화면 없이 한 번 실행 (테스트용)
    .\\.venv\\Scripts\\python.exe pallet_reach_demo.py --check      # 32개 목표(대기+놓기) 6D IK 검사
"""

import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from pallet_sim import env, ik6d
from pallet_sim.env import make_scene, parse_counts
from pallet_sim.planning import (FINAL_SPEED, SAFE_Z, TRANSIT_SPEED, VERTICAL_SPEED, ZONE_ORDER,
                                 distance_to_path, plan_legs)
from pallet_sim.robot import ARM_JOINTS, count_robot_contacts, find_actuator_for_joint, gravity_compensated_ctrl

HOLD_TIME = 2.0        # 놓기 목표에서 멈춰 있는 시간
CYCLE_PAUSE = 3.0      # 한 바퀴 끝나고 다시 시작하기 전 대기 시간

# 판정 기준
POS_GOAL_MM = 5.0               # 도착 위치 오차 목표
ROT_GOAL_DEG = 5.0              # 도착 방향 오차 목표
STOP_TRACKING_ERROR = 0.02      # 이동 중 계획 경로에서 2cm 이상 벗어나면 비상 정지
CLEARANCE_WARN = 0.005          # 계획 경로가 주변 물체와 5mm보다 가까우면 경고

PRINT_INTERVAL = 0.5


# ------------------------------------------------------------- 경로 계획 --
def zone_legs(zone, counts, start_pos):
    """한 구역 방문 순서: 이동 높이로 구역 위 -> 대기 위치 -> 놓기 목표 -> 정지 -> 같은 길로 복귀."""
    place = env.next_place_target(zone, counts[zone])
    approach = env.approach_target(place)
    above = np.array([place[0], place[1], SAFE_Z])
    return [
        (f"{zone} 위 이동 높이로 이동", above, TRANSIT_SPEED),
        (f"{zone} 대기 위치로 하강", approach, VERTICAL_SPEED),
        (f"{zone} 놓기 목표로 하강", place, FINAL_SPEED),
        (f"{zone} 놓기 목표에서 정지", None, HOLD_TIME),
        (f"{zone} 대기 위치로 상승", approach, FINAL_SPEED),
        (f"{zone} 이동 높이로 상승", above, VERTICAL_SPEED),
    ], place, approach


# ------------------------------------------------------------- 화면 표시 --
def draw_markers(viewer, data, counts, active_zone, path_points):
    """시각화 전용 표시 (물리 영향 없음)."""
    scn = viewer.user_scn
    scn.ngeom = 0

    def add(geom_type, size, pos, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], type=geom_type, size=size, pos=pos,
                            mat=np.eye(3).flatten(), rgba=np.array(rgba, dtype=np.float32))
        scn.ngeom += 1

    for zone in ZONE_ORDER:
        place = env.next_place_target(zone, counts[zone])
        if place is None:
            continue
        alpha = 0.6 if zone == active_zone else 0.25
        color = env.ZONE_COLORS[zone][:3]
        add(mujoco.mjtGeom.mjGEOM_BOX, [env.BOX_SIZE / 2] * 3, place, color + [alpha])            # 놓기 목표
        add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0], env.approach_target(place), color + [alpha + 0.2])  # 대기 위치

    for p in path_points[::3]:  # 계획 경로 (회색 점)
        add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.003, 0, 0], p, [0.6, 0.6, 0.6, 0.7])

    tip, _ = ik6d.get_fingertip_pose(data)  # 실제 손끝 위치 (자홍색 점)
    add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.007, 0, 0], tip, [1.0, 0.0, 1.0, 1.0])


def print_legend():
    print("[화면 범례]")
    print("  물리 시뮬레이션: 로봇팔 움직임(모터가 계획된 관절 목표를 따라감), 받침대, 팔레트, 상자")
    print("  시각화 전용 (물리 영향 없음):")
    print("    반투명 상자 = 놓기 목표, 같은 색 구슬 = 대기 위치 (진하게 = 지금 방문 중인 구역)")
    print("    회색 점 = 계획된 손끝 경로, 자홍색 점 = 실제 손끝(TCP) 위치")
    print("  상자는 잡지 않음. 이번 단계는 접근 자세와 경로 안전성 확인만 함\n")


# ------------------------------------------------------------- 실행 부분 --
class Runner:
    """계획된 구간을 실제 물리 시뮬레이션으로 재생하고, 오차·충돌을 감시한다."""

    def __init__(self, model, data, viewer, counts):
        self.model, self.data, self.viewer, self.counts = model, data, viewer, counts
        self.arm_act_ids = [find_actuator_for_joint(model, n) for n in ARM_JOINTS]
        self.arm_jnt_ids = [model.joint(n).id for n in ARM_JOINTS]
        # 모터 힘 = kp x (목표각 - 현재각) - kv x 현재속도.
        # 움직이는 동안 kv 항 때문에 (kv/kp)초 정도 늦게 따라오므로, 목표를 그만큼 앞당긴다.
        kp = model.actuator_gainprm[self.arm_act_ids, 0]
        kv = -model.actuator_biasprm[self.arm_act_ids, 2]
        self.lag_time = kv / kp  # Panda는 모든 관절이 0.1초
        self.steps_per_frame = max(1, round((1.0 / 60) / model.opt.timestep))
        self.active_zone = None
        self.path_points = []

    def measure(self):
        """현재 실제 손끝 위치와 방향 (시뮬레이션 상태에서 FK로 계산)."""
        mujoco.mj_kinematics(self.model, self.data)
        return ik6d.get_fingertip_pose(self.data)

    def emergency_stop(self, reason):
        """지금 실제 관절 각도를 그대로 유지하도록 모터 목표를 고정한다."""
        q_now = self.data.qpos[self.model.jnt_qposadr[self.arm_jnt_ids]].copy()
        ik_tmp = mujoco.MjData(self.model)
        ik_tmp.qpos[:] = self.data.qpos
        self.data.ctrl[self.arm_act_ids] = gravity_compensated_ctrl(self.model, ik_tmp, q_now, self.arm_act_ids)
        print(f"\n!!! 비상 정지: {reason}")
        print("    로봇은 현재 자세를 유지합니다. 다음 동작은 실행하지 않습니다.")

    # --- 다른 데모(예: pick_lift_demo.py)가 바꿔 쓸 수 있는 부분. 기본 동작은 이 데모용 ---
    def before_step(self, seg, seg_time):
        """매 물리 스텝 직전에 호출. 기본은 아무것도 하지 않음."""

    def safety_check(self):
        """위험한 상황이면 이유 문자열, 아니면 None. 기본: 로봇이 무엇이든 닿으면 위험."""
        return "로봇 접촉 감지" if count_robot_contacts(self.model, self.data) > 0 else None

    def status_text(self):
        """0.5초마다 출력하는 줄 끝에 덧붙일 문구. 기본은 없음."""
        return ""

    def segment_done(self):
        """True를 돌려주면 구간을 일찍 끝낸다 (예: 접촉을 감지하면 하강 중단). 기본은 끝까지 재생."""
        return False

    def draw(self):
        """시각화 전용 표시."""
        draw_markers(self.viewer, self.data, self.counts, self.active_zone, self.path_points)

    def play(self, seg):
        """한 구간 재생. 반환: (상태 'ok'|'closed'|'stopped', 통계)"""
        model, data = self.model, self.data
        n = len(seg["ctrl_path"])
        stats = {"max_track_mm": 0.0, "max_offpath_mm": 0.0, "max_rot_deg": 0.0}
        start_time = data.time
        next_print = 0.0
        while True:
            frame_start = time.time()
            elapsed = data.time - start_time
            for _ in range(self.steps_per_frame):
                t = min((data.time - start_time) / seg["duration"], 1.0)
                s = 3 * t**2 - 2 * t**3  # 천천히 출발, 천천히 정지
                u = s * (n - 1)
                i = min(int(u), n - 2) if n > 1 else 0
                frac = u - i if n > 1 else 0.0
                ctrl = (1 - frac) * seg["ctrl_path"][i] + frac * seg["ctrl_path"][min(i + 1, n - 1)]
                if n > 1:
                    # 계획상 관절 속도 = (경로상 관절 변화) x (경로 진행 속도)
                    du_dt = (6 * t * (1 - t) / seg["duration"]) * (n - 1)
                    q_dot_plan = (seg["joint_path"][i + 1] - seg["joint_path"][i]) * du_dt
                    ctrl = ctrl + self.lag_time * q_dot_plan  # 속도 피드포워드
                data.ctrl[self.arm_act_ids] = ctrl
                self.before_step(seg, data.time - start_time)
                mujoco.mj_step(model, data)

                # 안전 감시 1: 위험한 접촉이 있으면 즉시 정지
                problem = self.safety_check()
                if problem:
                    self.emergency_stop(f"[{seg['name']}] {problem} (t={data.time:.2f}s)")
                    return "stopped", stats
                if self.segment_done():
                    return "ok", stats

            # 계획상 지금 있어야 할 손끝 위치와 실제 위치 비교
            planned = (1 - frac) * seg["tcp_path"][i] + frac * seg["tcp_path"][min(i + 1, n - 1)]
            tip, rot = self.measure()
            track_mm = np.linalg.norm(tip - planned) * 1000                  # 일정 대비 오차
            offpath_mm = distance_to_path(tip, seg["tcp_path"]) * 1000     # 계획 경로 선에서 벗어난 거리
            rot_deg = ik6d.angle_between_deg(ik6d.GRIPPER_DOWN, rot)
            stats["max_track_mm"] = max(stats["max_track_mm"], track_mm)
            stats["max_offpath_mm"] = max(stats["max_offpath_mm"], offpath_mm)
            stats["max_rot_deg"] = max(stats["max_rot_deg"], rot_deg)

            # 안전 감시 2: 계획 경로에서 크게 벗어나면 정지
            if track_mm > STOP_TRACKING_ERROR * 1000:
                self.emergency_stop(f"[{seg['name']}] 계획 경로에서 {track_mm:.1f} mm 벗어남")
                return "stopped", stats

            if self.viewer is not None:
                if elapsed >= next_print:
                    print(f"    t={elapsed:4.1f}s 손끝 ({tip[0]:.3f}, {tip[1]:+.3f}, {tip[2]:.3f}) | "
                          f"일정 대비 {track_mm:5.2f} mm | 경로 이탈 {offpath_mm:5.2f} mm | 방향 오차 {rot_deg:4.2f}도"
                          f"{self.status_text()}")
                    next_print += PRINT_INTERVAL
                if not self.viewer.is_running():
                    return "closed", stats
                with self.viewer.lock():
                    self.draw()
                self.viewer.sync()
                remaining = self.steps_per_frame * model.opt.timestep - (time.time() - frame_start)
                if remaining > 0:
                    time.sleep(remaining)

            if data.time - start_time >= seg["duration"]:
                return "ok", stats

    def arrival_error(self, target):
        tip, rot = self.measure()
        return np.linalg.norm(tip - target) * 1000, ik6d.angle_between_deg(ik6d.GRIPPER_DOWN, rot)

    def run_cycle(self, cycle):
        """Z1 -> Z4를 한 바퀴 방문하고 시작 위치로 돌아온다. 반환: 'ok' | 'closed' | 'stopped'"""
        model, data = self.model, self.data
        ik_data = mujoco.MjData(model)  # 계획(IK) 전용, 실제 시뮬레이션과 분리
        base_qpos = data.qpos.copy()
        qpos_ids = model.jnt_qposadr[self.arm_jnt_ids]
        start_q = data.qpos[qpos_ids].copy()
        start_pos, _ = self.measure()
        park_pos = start_pos.copy()
        results = []

        print(f"\n========== {cycle}번째 실행: {' -> '.join(ZONE_ORDER)} ==========")
        q, pos = start_q, start_pos
        for zone in ZONE_ORDER:
            if env.next_place_target(zone, self.counts[zone]) is None:
                print(f"\n[{zone}] 가득 참 (상자 {self.counts[zone]}개) -> 건너뜀")
                continue
            legs, place, approach = zone_legs(zone, self.counts, pos)
            print(f"\n[{zone}] 상자 {self.counts[zone]}개 | 대기 위치 ({approach[0]:.3f}, {approach[1]:+.3f}, {approach[2]:.3f})"
                  f" | 놓기 목표 ({place[0]:.3f}, {place[1]:+.3f}, {place[2]:.3f})")

            segments, message = plan_legs(model, ik_data, base_qpos, q, pos, legs)
            if segments is None:
                print(f"  계획 실패: {message}")
                print("  -> 안전을 위해 움직이지 않고 정지합니다.")
                return "stopped", results
            clearances = [s["clearance"] for s in segments if s["clearance"] is not None]
            min_clear = min(clearances) * 1000
            warn = " (경고: 5mm보다 가까움)" if min_clear < CLEARANCE_WARN * 1000 else ""
            print(f"  계획 성공: 경로 점 {sum(len(s['joint_path']) for s in segments)}개 모두 충돌 없음, "
                  f"주변 물체와 최소 거리 {min_clear:.1f} mm{warn}")

            self.active_zone = zone
            self.path_points = np.concatenate([s["tcp_path"] for s in segments])
            record = {"zone": zone, "clear_mm": min_clear, "max_track_mm": 0.0, "max_offpath_mm": 0.0, "max_rot_deg": 0.0}
            for seg in segments:
                print(f"  > {seg['name']} ({seg['duration']:.1f}초)")
                status, stats = self.play(seg)
                for key in ("max_track_mm", "max_offpath_mm", "max_rot_deg"):
                    record[key] = max(record[key], stats[key])
                if status != "ok":
                    return status, results
                if seg["name"].endswith("대기 위치로 하강"):
                    record["approach_err"] = self.arrival_error(approach)
                    print(f"    대기 위치 도착: 위치 오차 {record['approach_err'][0]:.2f} mm, 방향 오차 {record['approach_err'][1]:.2f}도")
                if seg["name"].endswith("정지"):
                    record["place_err"] = self.arrival_error(place)
                    pe, re = record["place_err"]
                    verdict = "목표 달성" if pe <= POS_GOAL_MM and re <= ROT_GOAL_DEG else "목표 미달"
                    print(f"    놓기 목표 정지 후: 위치 오차 {pe:.2f} mm, 방향 오차 {re:.2f}도 -> {verdict} "
                          f"(기준 {POS_GOAL_MM:.0f} mm / {ROT_GOAL_DEG:.0f}도)")
            results.append(record)
            q, pos = segments[-1]["joint_path"][-1], segments[-1]["tcp_path"][-1]

        # 시작 위치로 복귀
        self.active_zone = None
        segments, message = plan_legs(model, ik_data, base_qpos, q, pos, [("시작 위치로 복귀", park_pos, TRANSIT_SPEED)])
        if segments is None:
            print(f"\n복귀 계획 실패: {message} -> 정지합니다.")
            return "stopped", results
        self.path_points = segments[0]["tcp_path"]
        print(f"\n> 시작 위치로 복귀 ({segments[0]['duration']:.1f}초)")
        status, _ = self.play(segments[0])
        print_summary(results)
        return status, results


def print_summary(results):
    print("\n----- 요약 -----")
    print("구역 | 대기 위치 도착 오차 | 놓기 목표 정지 후 오차 | 이동 중 최대: 일정 대비 / 경로 이탈 / 방향 | 계획 경로 최소 거리")
    for r in results:
        ae, re_ = r.get("approach_err", (float("nan"),) * 2), r.get("place_err", (float("nan"),) * 2)
        print(f" {r['zone']}  | {ae[0]:5.2f} mm, {ae[1]:4.2f}도   | {re_[0]:5.2f} mm, {re_[1]:4.2f}도      | "
              f"{r['max_track_mm']:5.2f} mm / {r['max_offpath_mm']:5.2f} mm / {r['max_rot_deg']:4.2f}도   | {r['clear_mm']:5.1f} mm")
    print("(일정 대비 = 지금 있어야 할 점과의 거리, 경로 이탈 = 계획된 선에서 옆으로 벗어난 거리)")


def check_all_targets():
    """32개 목표(4구역 x 4층 x 대기/놓기)에 대해 6D IK, 방향, 경로 충돌, 최소 거리를 검사한다.

    장면: 검사 구역은 그 층 아래까지만 상자, 나머지 구역은 가득 참 (가장 좁은 조건).
    검사 1) 목표 자세: 대기·놓기 목표를 각각 home에서 바로 6D IK로 풀어 오차·반복·충돌 확인
    검사 2) 접근 경로: home -> 구역 위 이동 높이 -> 대기 위치 -> 놓기 목표 직선 경로의
            모든 점(5mm 간격)에서 충돌과 주변 물체까지 최소 거리 확인
    """
    print("구역 층 | 대기 목표: 위치 / 방향 / 반복     | 놓기 목표: 위치 / 방향 / 반복     | 접근 경로 (최소 거리) | 결과")
    passed = 0
    for zone in ZONE_ORDER:
        for layer in range(env.MAX_LAYERS):
            counts = {z: env.MAX_LAYERS for z in ZONE_ORDER}
            counts[zone] = layer
            model, data = make_scene(counts)
            ik_data = mujoco.MjData(model)
            qpos_ids = model.jnt_qposadr[[model.joint(n).id for n in ARM_JOINTS]]
            place = env.next_place_target(zone, layer)
            approach = env.approach_target(place)
            above = np.array([place[0], place[1], SAFE_Z])

            # 검사 1: 목표 자세
            target_ok, target_texts = {}, []
            for name, target in (("대기", approach), ("놓기", place)):
                ok, _, iters, pe, re_, message = ik6d.solve_ik_6d(model, ik_data, data.qpos.copy(), target)
                target_ok[name] = ok
                text = f"{pe * 1000:6.3f} mm / {re_:5.3f}도 / {iters:3d}회"
                target_texts.append(text if ok else f"실패: {message}")

            # 검사 2: 접근 경로
            q = data.qpos[qpos_ids].copy()
            pos, _ = ik6d.get_fingertip_pose(data)
            path_ok, path_text, worst_clear = True, "", np.inf
            for name, end in (("이동", above), ("대기 하강", approach), ("놓기 하강", place)):
                seed = data.qpos.copy()
                seed[qpos_ids] = q
                ok, joint_path, _, message = ik6d.plan_line(model, ik_data, seed, pos, end)
                if ok:
                    ok, clear = ik6d.check_path_clearance(model, ik_data, data.qpos, joint_path)
                    worst_clear = min(worst_clear, clear)
                    message = "경로 중 로봇 충돌"
                if not ok:
                    path_ok, path_text = False, f"{name} 실패: {message}"
                    break
                q, pos = joint_path[-1], end
            if path_ok:
                warn = " 경고:5mm 미만" if worst_clear < CLEARANCE_WARN else ""
                path_text = f"OK ({worst_clear * 1000:4.1f} mm){warn}"

            approach_pass = target_ok["대기"] and path_ok
            place_pass = target_ok["놓기"] and path_ok
            passed += approach_pass + place_pass
            print(f" {zone}  {layer}  | {target_texts[0]:32s} | {target_texts[1]:32s} | {path_text:21s} | "
                  f"{'OK' if approach_pass and place_pass else '실패'}")
    print(f"\n6D IK + 접근 경로 검사 통과: {passed}/32 (대기·놓기 목표 기준)")
    print("※ 최소 거리는 50 mm까지만 계산합니다 (50.0 = 50 mm 이상).")
    print("※ 목표 자세와 경로만 검증한 것이며, 상자를 실제로 잡거나 운반하는 것은 검증하지 않았습니다.")
    return passed


def main():
    if "--check" in sys.argv:
        passed = check_all_targets()
        sys.exit(0 if passed == 32 else 1)  # 하나라도 실패하면 종료 코드 1 (CI에서 사용)

    counts = parse_counts(sys.argv[1:])
    model, data = make_scene(counts)

    if "--headless" in sys.argv:
        Runner(model, data, None, counts).run_cycle(1)
        return

    print_legend()
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = [0.45, 0.0, 0.35]
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 150
        viewer.cam.elevation = -25
        runner = Runner(model, data, viewer, counts)

        cycle = 1
        status = "ok"
        while status == "ok":
            status, _ = runner.run_cycle(cycle)
            if status != "ok" or "--once" in sys.argv:
                break
            print(f"\n{CYCLE_PAUSE:.0f}초 후 다시 시작합니다. (그만 보려면 창을 닫으세요)")
            hold = {"name": "대기", "ctrl_path": np.array([data.ctrl[runner.arm_act_ids].copy()]),
                    "tcp_path": np.array([runner.measure()[0]]), "duration": CYCLE_PAUSE}
            status, _ = runner.play(hold)
            cycle += 1

        if status != "closed":
            print("\n동작을 마쳤습니다. 뷰어 창을 직접 닫을 때까지 현재 자세를 유지합니다.")
            while viewer.is_running():
                for _ in range(runner.steps_per_frame):
                    mujoco.mj_step(model, data)
                with viewer.lock():
                    draw_markers(viewer, data, counts, None, runner.path_points)
                viewer.sync()
                time.sleep(1.0 / 60)
    print("뷰어가 닫혀 프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
