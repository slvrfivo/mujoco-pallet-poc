"""7단계: Pick & Transport - 잡은 상자를 팔레트의 빈 구역 위 대기 위치까지 옮기고 3초 유지.

상자는 여전히 손가락의 접촉력과 마찰로만 들고 있다. 내려놓거나 그리퍼를 열지는 않는다.

이번 단계의 핵심: 경로를 계획할 때 '손에 든 상자'를 실제 크기·자세 그대로 경로를 따라 옮겨 가며
상자-환경 충돌과 여유 거리를 검사한다. (손가락-상자 접촉은 정상, 상자-팔레트·다른 상자는 위험)

실행 (PowerShell, 프로젝트 폴더에서):
    .\\.venv\\Scripts\\python.exe pick_transport_demo.py                  # 팔레트 상자 0 1 2 3개, 빈 구역 자동 선택
    .\\.venv\\Scripts\\python.exe pick_transport_demo.py 4 4 0 4          # 구역별 상자 개수 지정
    .\\.venv\\Scripts\\python.exe pick_transport_demo.py --zone Z3 0 1 0 3   # 빈 구역 중 목표 직접 지정
    .\\.venv\\Scripts\\python.exe pick_transport_demo.py --headless       # 화면 없이 물리 검증
    .\\.venv\\Scripts\\python.exe dev_run.py --manual pick_transport_demo.py
"""

import math
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

import ik6d
import pallet_env as env
from ik_demo import ARM_JOINTS
from pallet_demo import parse_counts
from pallet_reach_demo import SAFE_Z, TRANSIT_SPEED, VERTICAL_SPEED, ZONE_ORDER, plan_legs
from pick_lift_demo import MAX_PENETRATION, PickRunner, build_pick_model, run_pick_lift

TARGET_CLEARANCE = 0.020   # 운반 경로에서 확보하고 싶은 최소 여유 거리 [m]
CLEARANCE_RANGE = 0.10     # 거리 계산 범위 [m] (이보다 멀면 '100 mm 이상'으로 표시)
# 경로 후보: (이동 높이, 목표 위 대기 높이 = 놓을 위치 + 이 값). 앞쪽부터 시도하고 여유가 충분하면 채택.
ROUTE_CANDIDATES = [(SAFE_Z, 0.10), (SAFE_Z, 0.15), (SAFE_Z + 0.05, 0.15), (SAFE_Z + 0.05, 0.20)]
TARGET_HOLD = 3.0          # 목표 위에서 유지하는 시간 [s]
TRANSPORT_PHASES = ("운반", "목표 위 유지")
MAX_SLIP_MM, MAX_SLIP_DEG, MAX_TILT_DEG = 5.0, 5.0, 5.0
TRAIL_EVERY = 25           # 실제 이동 경로를 몇 스텝마다 기록할지 (25스텝 = 0.05초)


# ------------------------------------------------- 손에 든 상자 충돌 검사 --
class CarryChecker:
    """경로의 각 자세에서 '손에 든 상자'를 손과 함께 옮겨 놓고 충돌·여유 거리를 계산한다."""

    def __init__(self, model, rel_pos, rel_rot):
        self.model = model
        self.rel_pos, self.rel_rot = rel_pos, rel_rot  # 손 좌표계에서 본 상자 위치·자세 (실측값)
        self.box_body = model.body("pick_box").id
        self.box_geom = model.geom("pick_box_geom").id
        self.box_adr = model.jnt_qposadr[model.body_jntadr[self.box_body]]
        self.fingers = {model.body("left_finger").id, model.body("right_finger").id}
        self.robot_root = model.body("link0").id
        self.robot_geoms, others = ik6d.collision_geom_groups(model)
        self.env_geoms = [g for g in others if g != self.box_geom]  # 팔레트·받침대·다른 상자 (바닥 제외)
        self.qpos_ids = model.jnt_qposadr[[model.joint(n).id for n in ARM_JOINTS]]

    def place_box(self, d):
        """d의 손 자세에 맞춰 상자 자유 관절 값을 바꾼다 (계획용 데이터에서만 사용)."""
        hand_pos = d.body("hand").xpos
        hand_rot = d.body("hand").xmat.reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (hand_rot @ self.rel_rot).flatten())
        d.qpos[self.box_adr:self.box_adr + 3] = hand_pos + hand_rot @ self.rel_pos
        d.qpos[self.box_adr + 3:self.box_adr + 7] = quat

    def name(self, g):
        return self.model.geom(g).name or self.model.body(self.model.geom_bodyid[g]).name

    def problems(self, d):
        """위험한 접촉 목록. 손가락-상자는 정상이므로 뺀다."""
        m = self.model
        found = []
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if self.box_body in (b1, b2) and ({b1, b2} & self.fingers):
                continue  # 손가락-상자: 정상 파지 접촉
            robot = self.robot_root in (m.body_rootid[b1], m.body_rootid[b2])
            if robot or self.box_body in (b1, b2):
                found.append(f"{self.name(c.geom1)}-{self.name(c.geom2)}")
        return found

    def clearances(self, d):
        """(로봇-환경 최소 거리, 상자-환경 최소 거리, 가장 가까운 쌍 이름). 손가락-상자 쌍은 계산하지 않음."""
        fromto = np.zeros(6)
        best_robot, best_box = CLEARANCE_RANGE, CLEARANCE_RANGE
        best_any, pair = CLEARANCE_RANGE, ""
        for g2 in self.env_geoms:
            dist = mujoco.mj_geomDistance(self.model, d, self.box_geom, g2, CLEARANCE_RANGE, fromto)
            best_box = min(best_box, dist)
            if dist < best_any:
                best_any, pair = dist, f"들고 있는 상자-{self.name(g2)}"
            for g1 in self.robot_geoms:
                dist = mujoco.mj_geomDistance(self.model, d, g1, g2, CLEARANCE_RANGE, fromto)
                best_robot = min(best_robot, dist)
                if dist < best_any:
                    best_any, pair = dist, f"{self.name(g1)}-{self.name(g2)}"
        return best_robot, best_box, pair

    def check_path(self, ik_data, base_qpos, joint_path):
        """경로 전체 검사. 반환: (충돌 없음 여부, 로봇 최소 거리, 상자 최소 거리, 가장 가까운 쌍 또는 충돌 내용)"""
        worst_robot, worst_box, worst_pair = CLEARANCE_RANGE, CLEARANCE_RANGE, ""
        for q_arm in joint_path:
            ik_data.qpos[:] = base_qpos
            ik_data.qpos[self.qpos_ids] = q_arm
            mujoco.mj_kinematics(self.model, ik_data)
            self.place_box(ik_data)
            mujoco.mj_fwdPosition(self.model, ik_data)
            hits = self.problems(ik_data)
            if hits:
                return False, 0.0, 0.0, ", ".join(sorted(set(hits)))
            r, b, pair = self.clearances(ik_data)
            if min(r, b) < min(worst_robot, worst_box):
                worst_pair = pair
            worst_robot, worst_box = min(worst_robot, r), min(worst_box, b)
        return True, worst_robot, worst_box, worst_pair


def fmt_mm(dist):
    return "100 mm 이상" if dist >= CLEARANCE_RANGE else f"{dist * 1000:.1f} mm"


# ------------------------------------------------------------ 실행 부분 --
class TransportRunner(PickRunner):
    """Pick & Lift 실행기에 '들고 이동하는 단계'의 감시·표시를 더한다."""

    def __init__(self, model, data, viewer, counts):
        super().__init__(model, data, viewer)
        self.counts = counts
        self.checker = None          # 파지 후 실측한 상대 자세로 만든다
        self.trail = []              # 실제 손끝 이동 경로 (시각화 전용)
        self.target_place = None
        self.target_zone = None
        self.forced_zone = None      # --zone으로 지정한 목표 (없으면 첫 빈 구역)
        self.step_count = 0
        self.reset_transport_stats()

    def reset_transport_stats(self):
        self.tstats = {"steps": 0, "both": 0, "max_slip_mm": 0.0, "max_slip_deg": 0.0, "max_tilt_deg": 0.0,
                       "min_robot": CLEARANCE_RANGE, "min_box": CLEARANCE_RANGE, "pair": "", "ref": None,
                       "max_pen": 0.0, "limit_steps": 0}

    def box_env_contacts(self):
        """들고 있는 상자가 손가락 이외의 것(팔레트·다른 상자·받침대 등)에 닿은 목록."""
        m, d = self.model, self.data
        hits = []
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if self.box_id in (b1, b2):
                other = c.geom2 if b1 == self.box_id else c.geom1
                if m.geom_bodyid[other] not in (self.left_id, self.right_id):
                    hits.append(m.geom(other).name or m.body(m.geom_bodyid[other]).name)
        return hits

    def box_tilt_deg(self):
        """상자가 수평에서 기울어진 각도 (상자 z축과 월드 z축 사이 각)."""
        box_z = self.data.xmat[self.box_id].reshape(3, 3)[:, 2]
        return math.degrees(math.acos(np.clip(abs(box_z[2]), -1.0, 1.0)))

    def before_step(self, seg, seg_time):
        super().before_step(seg, seg_time)
        self.step_count += 1
        if self.step_count % TRAIL_EVERY == 0:
            self.trail.append(ik6d.get_fingertip_pose(self.data)[0].copy())
            self.trail = self.trail[-600:]

    def safety_check(self):
        if self.phase not in TRANSPORT_PHASES:
            return super().safety_check()
        info = self.contacts()
        if info["danger"]:
            return f"위험한 접촉: {', '.join(sorted(set(info['danger'])))}"
        hits = self.box_env_contacts()
        if hits:
            return f"들고 있는 상자가 {', '.join(sorted(set(hits)))}에 닿음"
        if info["penetration"] > MAX_PENETRATION:
            return f"손가락이 상자를 {info['penetration'] * 1000:.1f} mm 파고듦"

        st = self.tstats
        st["steps"] += 1
        st["both"] += info["left"] and info["right"]
        st["max_pen"] = max(st["max_pen"], info["penetration"])
        rel_pos, rel_rot = self.box_in_hand()
        if st["ref"] is None:
            st["ref"] = (rel_pos.copy(), rel_rot.copy())
        st["max_slip_mm"] = max(st["max_slip_mm"], np.linalg.norm(rel_pos - st["ref"][0]) * 1000)
        st["max_slip_deg"] = max(st["max_slip_deg"], ik6d.angle_between_deg(st["ref"][1], rel_rot))
        st["max_tilt_deg"] = max(st["max_tilt_deg"], self.box_tilt_deg())
        q = self.data.qpos[self.model.jnt_qposadr[self.arm_jnt_ids]]
        rng = self.model.jnt_range[self.arm_jnt_ids]
        st["limit_steps"] += bool(np.any(q < rng[:, 0]) or np.any(q > rng[:, 1]))
        if st["steps"] % TRAIL_EVERY == 0 and self.checker is not None:
            r, b, pair = self.checker.clearances(self.data)  # 실제 시뮬레이션 상태에서 측정
            if min(r, b) < min(st["min_robot"], st["min_box"]):
                st["pair"] = pair
            st["min_robot"], st["min_box"] = min(st["min_robot"], r), min(st["min_box"], b)
        if not (info["left"] or info["right"]):
            return "상자를 놓침 (양쪽 손가락 모두 접촉 없음)"
        return None

    def status_text(self):
        text = super().status_text()
        if self.phase in TRANSPORT_PHASES and self.checker is not None:
            r, b, pair = self.checker.clearances(self.data)
            rel_pos, rel_rot = self.box_in_hand()
            ref = self.tstats["ref"] or (rel_pos, rel_rot)
            text += (f"\n           여유 거리: 상자-환경 {fmt_mm(b)}, 로봇-환경 {fmt_mm(r)} | "
                     f"손 안 미끄러짐 {np.linalg.norm(rel_pos - ref[0]) * 1000:.2f} mm | 상자 기울기 {self.box_tilt_deg():.2f}도")
        return text

    def draw(self):
        super().draw()  # 계획 경로(회색), 목표 구슬, 실제 손끝(자홍)
        scn = self.viewer.user_scn

        def add(geom_type, size, pos, rgba):
            if scn.ngeom < scn.maxgeom:
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], type=geom_type, size=size, pos=pos,
                                    mat=np.eye(3).flatten(), rgba=np.array(rgba, dtype=np.float32))
                scn.ngeom += 1

        for p in self.trail[::2]:  # 실제 지나온 손끝 경로 (하늘색)
            add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.004, 0, 0], p, [0.0, 0.85, 1.0, 0.9])
        if self.target_place is not None:  # 목표 구역에 놓일 자리 (반투명, 구역 색)
            add(mujoco.mjtGeom.mjGEOM_BOX, [env.BOX_SIZE / 2] * 3, self.target_place,
                env.ZONE_COLORS[self.target_zone][:3] + [0.35])


def choose_target(model, data, forced):
    """구역별 높이를 측정해(지금은 시뮬레이션 정답) 빈 구역을 고른다."""
    counts = env.heights_to_counts(env.measure_zone_heights_ground_truth(model, data))
    empty = [z for z in ZONE_ORDER if counts[z] == 0]
    print(f"\n팔레트 측정(시뮬레이션 정답): " + ", ".join(f"{z} {counts[z]}개" for z in ZONE_ORDER)
          + f" -> 빈 구역: {', '.join(empty) or '없음'}")
    if forced:
        if forced not in empty:
            print(f"  지정한 {forced}는 비어 있지 않아 사용할 수 없습니다.")
            return None, counts
        return forced, counts
    return (empty[0] if empty else None), counts


def plan_transport(runner, ik_data, zone, counts):
    """후보 경로를 차례로 계획·검사해, 들고 있는 상자까지 포함한 여유 거리가 20 mm 이상인 첫 경로를 고른다."""
    model, data = runner.model, runner.data
    qpos_ids = model.jnt_qposadr[runner.arm_jnt_ids]
    q_now = data.qpos[qpos_ids].copy()
    tip, _ = runner.measure()
    place = env.next_place_target(zone, counts[zone])
    cx, cy = env.zone_center(zone)

    # IK 계산용 데이터에서는 쥔 상자를 치워 둔다 (손가락-상자 정상 접촉을 IK가 충돌로 보지 않도록).
    # 대신 아래 check_path에서 상자를 손과 함께 옮겨 놓고 충돌·거리를 다시 검사한다.
    ik_base = data.qpos.copy()
    ik_base[runner.checker.box_adr:runner.checker.box_adr + 3] = [5.0, 5.0, 0.5]

    print(f"\n운반 경로 후보 검사 (들고 있는 상자 포함, 목표 여유 {TARGET_CLEARANCE * 1000:.0f} mm)")
    print("  이동 높이 | 목표 위 대기 높이 | 결과")
    best = None
    for transit_z, hover_gap in ROUTE_CANDIDATES:
        hover = place + np.array([0.0, 0.0, hover_gap])
        legs = [("이동 높이로 상승", np.array([tip[0], tip[1], transit_z]), VERTICAL_SPEED),
                (f"{zone} 위로 이동", np.array([cx, cy, transit_z]), TRANSIT_SPEED),
                (f"{zone} 위 대기 위치로 하강", hover, VERTICAL_SPEED)]
        label = f"  {transit_z:.2f} m    | {hover[2]:.3f} m (+{hover_gap * 100:.0f} cm)    |"
        segments, message = plan_legs(model, ik_data, ik_base, q_now, tip, legs)
        if segments is None:
            print(f"{label} 계획 실패: {message}")
            continue
        ok, worst_r, worst_b, pair = True, CLEARANCE_RANGE, CLEARANCE_RANGE, ""
        for seg in segments:
            seg_ok, r, b, info = runner.checker.check_path(ik_data, data.qpos, seg["joint_path"])
            if not seg_ok:
                ok, pair = False, f"[{seg['name']}] 충돌: {info}"
                break
            if min(r, b) < min(worst_r, worst_b):
                pair = info
            worst_r, worst_b = min(worst_r, r), min(worst_b, b)
        if not ok:
            print(f"{label} 거부 - {pair}")
            continue
        worst = min(worst_r, worst_b)
        print(f"{label} 충돌 없음, 최소 여유 {fmt_mm(worst)} (상자-환경 {fmt_mm(worst_b)}, 로봇-환경 {fmt_mm(worst_r)}), "
              f"가장 가까운 곳: {pair}")
        candidate = {"segments": segments, "worst": worst, "worst_box": worst_b, "worst_robot": worst_r,
                     "pair": pair, "transit_z": transit_z, "hover": hover}
        if best is None or worst > best["worst"]:
            best = candidate
        if worst >= TARGET_CLEARANCE:
            return candidate, place
    if best is not None:
        print(f"  -> 여유 {TARGET_CLEARANCE * 1000:.0f} mm 이상인 경로가 없어, 충돌 없는 경로 중 여유가 가장 큰 경로를 씁니다 "
              f"({fmt_mm(best['worst'])}).")
    return best, place


def run_transport(runner):
    model, data = runner.model, runner.data
    pick = run_pick_lift(runner)
    if not pick["success"]:
        return {"success": False, "reason": f"Pick & Lift 실패: {pick['reason']}"}

    zone, counts = choose_target(model, data, runner.forced_zone)
    if zone is None:
        return {"success": False, "reason": "운반할 빈 구역이 없음"}
    runner.target_zone = zone

    # 파지 상태를 실측해 '손에 든 상자'의 상대 자세로 사용
    rel_pos, rel_rot = runner.box_in_hand()
    runner.checker = CarryChecker(model, rel_pos, rel_rot)
    print(f"손에 대한 상자 위치(실측): ({rel_pos[0] * 1000:+.1f}, {rel_pos[1] * 1000:+.1f}, {rel_pos[2] * 1000:+.1f}) mm")

    ik_data = mujoco.MjData(model)
    route, place = plan_transport(runner, ik_data, zone, counts)
    if route is None:
        return {"success": False, "reason": "상자를 든 채 충돌 없이 갈 수 있는 경로가 없음 (움직이지 않음)"}
    runner.target_place = place
    runner.path_points = np.concatenate([s["tcp_path"] for s in route["segments"]])
    runner.markers = [(route["hover"], env.ZONE_COLORS[zone][:3] + [0.8])]
    runner.add_payload(route["segments"], ik_data, env.BOX_MASS)

    box_start = data.xpos[runner.box_id].copy()
    tip_start, _ = runner.measure()
    print(f"\n[7] {zone}의 빈 자리 위로 운반 (상자를 든 채 이동, 그리퍼는 닫힌 상태 유지)")
    runner.reset_transport_stats()
    if not runner.step("운반", "운반", route["segments"]):
        return {"success": False, "reason": "운반 중 정지"}
    moving = dict(runner.tstats)

    print(f"\n[8] {zone} 위 대기 위치에서 {TARGET_HOLD:.0f}초 유지 (내려놓지 않음)")
    runner.reset_transport_stats()
    if not runner.step("유지", "목표 위 유지", [("정지", f"{TARGET_HOLD:.0f}초 유지", TARGET_HOLD)]):
        return {"success": False, "reason": "목표 위 유지 중 정지"}
    hold = runner.tstats

    # ---- 판정 ----
    box_end = data.xpos[runner.box_id].copy()
    tip_end, _ = runner.measure()
    box_move = box_end - box_start
    tip_move = tip_end - tip_start
    offset_xy = np.linalg.norm(box_end[:2] - np.array(env.zone_center(zone))) * 1000
    stack_top = env.PALLET_TOP_Z + env.stack_height(counts[zone])
    bottom_gap = (box_end[2] - env.BOX_SIZE / 2 - stack_top) * 100
    min_robot = min(moving["min_robot"], hold["min_robot"])
    min_box = min(moving["min_box"], hold["min_box"])
    checks = [
        ("상자가 손과 함께 이동", np.linalg.norm(box_move - tip_move) * 1000 <= MAX_SLIP_MM,
         f"상자 이동 {np.linalg.norm(box_move) * 100:.1f} cm, 손끝 이동 {np.linalg.norm(tip_move) * 100:.1f} cm, "
         f"차이 {np.linalg.norm(box_move - tip_move) * 1000:.2f} mm"),
        ("운반·유지 중 양쪽 손가락 접촉", moving["both"] == moving["steps"] and hold["both"] == hold["steps"],
         f"운반 {moving['both']}/{moving['steps']}, 유지 {hold['both']}/{hold['steps']} 스텝"),
        (f"손 안 미끄러짐 {MAX_SLIP_MM:.0f} mm·{MAX_SLIP_DEG:.0f}도 이하",
         max(moving["max_slip_mm"], hold["max_slip_mm"]) <= MAX_SLIP_MM
         and max(moving["max_slip_deg"], hold["max_slip_deg"]) <= MAX_SLIP_DEG,
         f"운반 중 {moving['max_slip_mm']:.2f} mm / {moving['max_slip_deg']:.2f}도, "
         f"유지 중 {hold['max_slip_mm']:.2f} mm / {hold['max_slip_deg']:.2f}도"),
        (f"상자 기울기 {MAX_TILT_DEG:.0f}도 이하", max(moving["max_tilt_deg"], hold["max_tilt_deg"]) <= MAX_TILT_DEG,
         f"최대 {max(moving['max_tilt_deg'], hold['max_tilt_deg']):.2f}도"),
        ("상자-환경 / 로봇-환경 충돌 없음", True, "충돌이 있었다면 비상 정지됨"),
        (f"실제 이동 중 최소 여유 거리 (목표 {TARGET_CLEARANCE * 1000:.0f} mm)", min(min_robot, min_box) >= TARGET_CLEARANCE,
         f"상자-환경 {fmt_mm(min_box)}, 로봇-환경 {fmt_mm(min_robot)} (0.05초마다 측정), 가장 가까운 곳: "
         f"{moving['pair'] or hold['pair'] or '-'}"),
        (f"{zone} 위에 정지", offset_xy <= 10 and bottom_gap > 0,
         f"상자 중심이 구역 중심에서 {offset_xy:.1f} mm, 상자 바닥이 아래 면보다 {bottom_gap:.1f} cm 위"),
        ("관절 한계 위반 없음", moving["limit_steps"] + hold["limit_steps"] == 0,
         f"{moving['limit_steps'] + hold['limit_steps']}스텝"),
    ]
    print("\n===== Pick & Transport 결과 =====")
    print(f"  계획 단계 최소 여유: {fmt_mm(route['worst'])} (상자-환경 {fmt_mm(route['worst_box'])}, "
          f"로봇-환경 {fmt_mm(route['worst_robot'])}), 가장 가까운 곳: {route['pair']}")
    for name, ok, value in checks:
        print(f"  [{'성공' if ok else '실패'}] {name}: {value}")
    success = all(ok for _, ok, _ in checks)
    print(f"  => {'성공' if success else '실패'} (상자는 내려놓지 않고 손에 든 상태)")
    return {"success": success, "reason": "" if success else "판정 항목 실패"}


def print_legend():
    print("[화면 범례]")
    print("  물리 시뮬레이션: 로봇팔, 그리퍼, 주황 상자(손가락 마찰로만 들고 있음), 공급 받침대, 팔레트와 팔레트 위 상자")
    print("  시각화 전용 (물리 영향 없음):")
    print("    회색 점 = 계획된 손끝 경로(현재 단계), 하늘색 점 = 실제로 지나온 손끝 경로, 자홍색 점 = 실제 손끝")
    print("    구역 색 반투명 상자 = 목표 구역에 놓일 자리, 구역 색 구슬 = 목표 위 대기 위치")
    print("  이번 단계는 목표 위에서 들고 정지하는 것까지 (내려놓기·그리퍼 열기 없음)")


def parse_args(argv):
    forced = None
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--zone" and i + 1 < len(argv):
            forced = argv[i + 1].upper()
            i += 2
            continue
        rest.append(argv[i])
        i += 1
    return forced, parse_counts(rest)


def main():
    forced, counts = parse_args(sys.argv[1:])
    model = build_pick_model(counts)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    if "--headless" in sys.argv:
        runner = TransportRunner(model, data, None, counts)
        runner.forced_zone = forced
        result = run_transport(runner)
        if not result["success"]:
            print(f"\n실패: {result['reason']}")
        return

    print_legend()
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        # 공급 받침대와 팔레트, 로봇이 함께 보이는 오른쪽 앞 시점
        viewer.cam.lookat[:] = [0.35, -0.12, 0.33]
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -25
        runner = TransportRunner(model, data, viewer, counts)
        runner.forced_zone = forced
        try:
            result = run_transport(runner)
            if not result["success"]:
                print(f"\n실패: {result['reason']}")
        except KeyboardInterrupt:
            pass

        if viewer.is_running():
            print("\n동작을 마쳤습니다. 마지막 상태(상자를 든 채)를 유지합니다. 물리 시뮬레이션은 계속 진행됩니다.")
            print("다시 실행: dev_run.py --manual 로 실행했다면 터미널에서 R, 아니면 창을 닫고 명령을 다시 실행하세요.")
            while viewer.is_running():
                for _ in range(runner.steps_per_frame):
                    mujoco.mj_step(model, data)
                with viewer.lock():
                    runner.draw()
                viewer.sync()
                time.sleep(1.0 / 60)
    print("뷰어가 닫혀 프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
