"""8단계: Pick & Place - 상자를 잡아 팔레트의 빈 구역으로 옮긴 뒤 실제로 내려놓고 손을 뺀다.

전체 순서: 파지 -> 들어 올리기 -> 운반 -> 목표 위 정지 -> 하강 -> 지지면 접촉 확인
          -> 그리퍼 열기 -> 수직 후퇴 -> 놓인 상자 3초 안정성 확인 -> 개수 측정 확인

상자는 처음부터 끝까지 자유 물체다. 손가락 마찰로 들고, 팔레트에 닿은 뒤 그리퍼를 열어 놓는다.
(상자를 특정 좌표에 고정하거나 팔레트에 밀어 넣지 않는다.)

실행 (PowerShell, 프로젝트 폴더에서):
    .\\.venv\\Scripts\\python.exe pick_place_demo.py                       # 빈 팔레트, Z1에 놓기
    .\\.venv\\Scripts\\python.exe pick_place_demo.py --zone Z3             # 빈 구역 지정
    .\\.venv\\Scripts\\python.exe pick_place_demo.py --zone Z2 1 0 2 3     # 팔레트 상자 개수 지정 (빈 구역에만 놓음)
    .\\.venv\\Scripts\\python.exe pick_place_demo.py --headless            # 화면 없이 물리 검증
    .\\.venv\\Scripts\\python.exe dev_run.py --manual pick_place_demo.py
"""

import math
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

import ik6d
import pallet_env as env
from fk_demo import FINGERTIP_OFFSET
from ik_demo import gravity_compensated_ctrl
from pallet_demo import parse_counts
from pallet_reach_demo import plan_legs
from pick_lift_demo import GRIP_CLOSE, GRIP_OPEN, MAX_PENETRATION, build_pick_model, plan_or_fail, run_pick_lift
from pick_transport_demo import (CLEARANCE_RANGE, CarryChecker, TransportRunner, choose_target, fmt_mm,
                                 plan_transport)

# ------------------------------------------------------------ 동작 설정 --
PRECONTACT_GAP = 0.010     # 계획된 하강의 끝: 상자 바닥이 지지면 10 mm 위
PRESS_LIMIT = 0.003        # 접촉 하강의 끝: 상자 바닥이 지지면보다 최대 3 mm 아래 (접촉하면 그 전에 멈춤)
DESCENT_SPEED = 0.02       # 대기 위치 -> 접촉 직전 [m/s]
CONTACT_SPEED = 0.005      # 접촉 하강 [m/s]
CONTACT_STEPS = 10         # 상자-지지면 접촉이 이만큼 연속되면 하강 중단 (0.02초)
CONTACT_SETTLE = 0.5       # 접촉 후 안정 확인 시간 [s]
OPEN_TIME = 1.5            # 그리퍼를 여는 시간 [s]
OPEN_SETTLE = 0.5
RETREAT_HEIGHT = 0.10      # 손을 수직으로 빼는 높이 [m]
RETREAT_SPEED = 0.02
STABLE_TIME = 3.0          # 손을 뺀 뒤 상자를 지켜보는 시간 [s]
HOVER_PAUSE = 1.0          # 목표 위에서 잠시 정지

# 놓기 단계의 여유 거리 기준 (운반용 20 mm 대신)
PLACE_CLEARANCE = 0.010    # 로봇-환경, 상자-지지면 이외 물체: 10 mm 이상

# 판정 기준
MAX_PLACE_ERROR_MM = 5.0   # 상자 중심(수평)의 목표 오차
MAX_TILT_DEG = 2.0         # 놓인 상자 기울기
MAX_SUPPORT_PEN = 0.002    # 상자가 지지면을 파고든 깊이
MAX_RELEASE_RISE_MM = 1.0  # 그리퍼를 열 때 상자가 끌려 올라간 높이
MAX_RELEASE_MOVE_MM = 2.0  # 그리퍼를 열 때 상자 이동
MAX_STABLE_MOVE_MM = 1.0   # 손을 뺀 뒤 3초 동안 상자 이동
PLACE_PHASES = ("하강", "접촉 하강", "접촉 확인", "놓기", "후퇴", "안정성 확인")


class PlaceChecker(CarryChecker):
    """들고 있는 상자 충돌 검사에 '목표 지지면'을 따로 다루는 기능을 더한다."""

    def __init__(self, model, rel_pos, rel_rot, support_geom):
        super().__init__(model, rel_pos, rel_rot)
        self.support_geom = support_geom
        self.other_geoms = [g for g in self.env_geoms if g != support_geom]

    def place_clearances(self, d):
        """(로봇-환경, 상자-지지면 이외 물체, 상자-지지면) 최소 거리, 가장 가까운 쌍."""
        fromto = np.zeros(6)
        support = mujoco.mj_geomDistance(self.model, d, self.box_geom, self.support_geom, CLEARANCE_RANGE, fromto)
        best_robot, best_other, best_any, pair = CLEARANCE_RANGE, CLEARANCE_RANGE, CLEARANCE_RANGE, ""
        for g2 in self.env_geoms:
            if g2 != self.support_geom:
                dist = mujoco.mj_geomDistance(self.model, d, self.box_geom, g2, CLEARANCE_RANGE, fromto)
                best_other = min(best_other, dist)
                if dist < best_any:
                    best_any, pair = dist, f"들고 있는 상자-{self.name(g2)}"
            for g1 in self.robot_geoms:
                dist = mujoco.mj_geomDistance(self.model, d, g1, g2, CLEARANCE_RANGE, fromto)
                best_robot = min(best_robot, dist)
                if dist < best_any:
                    best_any, pair = dist, f"{self.name(g1)}-{self.name(g2)}"
        return best_robot, best_other, support, pair

    def check_place_path(self, ik_data, base_qpos, joint_path, allow_support):
        """하강 경로 검사. allow_support=True면 상자-지지면 접촉만 정상으로 허용."""
        worst = [CLEARANCE_RANGE, CLEARANCE_RANGE, CLEARANCE_RANGE]
        pair = ""
        for q_arm in joint_path:
            ik_data.qpos[:] = base_qpos
            ik_data.qpos[self.qpos_ids] = q_arm
            mujoco.mj_kinematics(self.model, ik_data)
            self.place_box(ik_data)
            mujoco.mj_fwdPosition(self.model, ik_data)
            names = self.problems(ik_data)  # 손가락-상자를 뺀 위험 접촉
            if allow_support:  # 접촉 하강 구간에서만 상자-지지면 접촉을 정상으로 본다
                box, support = self.name(self.box_geom), self.name(self.support_geom)
                names = [n for n in names if n not in (f"{box}-{support}", f"{support}-{box}")]
            if names:
                return False, worst, ", ".join(sorted(set(names)))
            r, o, s, p = self.place_clearances(ik_data)
            if min(r, o) < min(worst[0], worst[1]):
                pair = p
            worst = [min(worst[0], r), min(worst[1], o), min(worst[2], s)]
        return True, worst, pair


class PlaceRunner(TransportRunner):
    """운반 실행기에 하강·놓기·후퇴 단계의 접촉 판정과 측정을 더한다."""

    def __init__(self, model, data, viewer, counts):
        super().__init__(model, data, viewer, counts)
        self.support_geom = model.geom("pallet").id  # 빈 구역의 지지면 = 팔레트 윗면
        self.support_top = env.PALLET_TOP_Z
        self.contact_run = 0
        self.first_contact = None
        self.reset_phase_stats()

    def reset_phase_stats(self):
        self.pst = {"steps": 0, "support_steps": 0, "max_support_pen": 0.0, "ref": None,
                    "max_move_mm": 0.0, "max_rise_mm": 0.0, "max_tilt": 0.0, "finger_steps": 0,
                    "min_robot": CLEARANCE_RANGE}

    def box_contacts_split(self):
        """들고 있는(또는 놓은) 상자의 접촉을 (지지면 접촉 목록, 그 외 물체 이름 목록)으로 나눈다. 손가락 제외."""
        m, d = self.model, self.data
        support, other = [], []
        for i in range(d.ncon):
            c = d.contact[i]
            if self.box_id not in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
                continue
            g = c.geom2 if m.geom_bodyid[c.geom1] == self.box_id else c.geom1
            if m.geom_bodyid[g] in (self.left_id, self.right_id):
                continue
            if g == self.support_geom:
                support.append(c)
            else:
                other.append(m.geom(g).name or m.body(m.geom_bodyid[g]).name)
        return support, other

    def box_bottom_gap(self):
        """상자 바닥과 지지면 윗면 사이 높이 [m] (음수면 파고듦)."""
        rot = self.data.xmat[self.box_id].reshape(3, 3)
        half_z = (np.abs(rot) @ np.full(3, env.BOX_SIZE / 2))[2]
        return self.data.xpos[self.box_id][2] - half_z - self.support_top

    def segment_done(self):
        return self.phase == "접촉 하강" and self.contact_run >= CONTACT_STEPS

    def safety_check(self):
        if self.phase not in PLACE_PHASES:
            return super().safety_check()
        info = self.contacts()
        if info["danger"]:
            return f"위험한 접촉: {', '.join(sorted(set(info['danger'])))}"
        support, other = self.box_contacts_split()
        if other:
            return f"상자가 지지면이 아닌 {', '.join(sorted(set(other)))}에 닿음"
        if info["penetration"] > MAX_PENETRATION:
            return f"손가락이 상자를 {info['penetration'] * 1000:.1f} mm 파고듦"

        st = self.pst
        st["steps"] += 1
        pen = max((-c.dist for c in support), default=0.0)
        st["max_support_pen"] = max(st["max_support_pen"], pen)
        if pen > MAX_SUPPORT_PEN:
            return f"상자가 지지면을 {pen * 1000:.1f} mm 파고듦 (물리적으로 불가능한 상태)"
        st["support_steps"] += bool(support)
        box_pos = self.data.xpos[self.box_id].copy()
        if st["ref"] is None:
            st["ref"] = box_pos
        st["max_move_mm"] = max(st["max_move_mm"], np.linalg.norm(box_pos - st["ref"]) * 1000)
        st["max_rise_mm"] = max(st["max_rise_mm"], (box_pos[2] - st["ref"][2]) * 1000)
        st["max_tilt"] = max(st["max_tilt"], self.box_tilt_deg())
        st["finger_steps"] += info["left"] or info["right"]
        if st["steps"] % 25 == 0 and self.checker is not None:
            r, _, _, _ = self.checker.place_clearances(self.data)
            st["min_robot"] = min(st["min_robot"], r)

        if self.phase == "하강":
            if support:
                return "계획보다 일찍 상자가 지지면에 닿음 (높이 계산이 어긋남)"
            if not (info["left"] or info["right"]):
                return "하강 중 상자를 놓침"
        elif self.phase == "접촉 하강":
            if support and self.first_contact is None:
                self.first_contact = "상자-지지면"
            self.contact_run = self.contact_run + 1 if support else 0
        elif self.phase == "놓기":
            if st["max_rise_mm"] > MAX_RELEASE_RISE_MM:
                return f"그리퍼를 열 때 상자가 {st['max_rise_mm']:.1f} mm 끌려 올라감"
        elif self.phase == "후퇴":
            if info["left"] or info["right"]:
                return "후퇴 중 손가락이 상자에 닿음 (상자를 끌고 갈 위험)"
        return None

    def status_text(self):
        if self.phase not in PLACE_PHASES:
            return super().status_text()
        info = self.contacts()
        support, _ = self.box_contacts_split()
        p = self.data.xpos[self.box_id]
        return (f"\n           상자 중심 ({p[0]:.3f}, {p[1]:+.3f}, {p[2]:.3f}) | 바닥-지지면 {self.box_bottom_gap() * 1000:+.1f} mm | "
                f"지지면 접촉 {'O' if support else 'X'} | 기울기 {self.box_tilt_deg():.2f}도 | "
                f"손가락 접촉 왼쪽 {'O' if info['left'] else 'X'} 오른쪽 {'O' if info['right'] else 'X'} | "
                f"손가락 간격 {self.finger_gap_mm():.1f} mm")

    def hold_current_pose(self):
        """지금 실제 자세를 그대로 유지하도록 모터 목표를 바꾼다 (누르는 힘·무게 보상 제거)."""
        q_now = self.data.qpos[self.model.jnt_qposadr[self.arm_jnt_ids]].copy()
        tmp = mujoco.MjData(self.model)
        tmp.qpos[:] = self.data.qpos
        self.data.ctrl[self.arm_act_ids] = gravity_compensated_ctrl(self.model, tmp, q_now, self.arm_act_ids)


def place_pose(rel_pos, rel_rot, zone, support_top):
    """실측한 '손에 대한 상자 위치·자세'로, 상자 바닥이 지지면에서 gap만큼 위일 때의 손끝 목표를 계산한다."""
    box_rot = ik6d.GRIPPER_DOWN @ rel_rot
    half_z = (np.abs(box_rot) @ np.full(3, env.BOX_SIZE / 2))[2]  # 상자의 실제 아래쪽 절반 높이
    cx, cy = env.zone_center(zone)
    box_center = np.array([cx, cy, support_top + half_z])
    tcp_from_box = ik6d.GRIPPER_DOWN @ (FINGERTIP_OFFSET - rel_pos)  # 상자 중심 -> 손끝

    def tcp_for(gap):
        return box_center + np.array([0.0, 0.0, gap]) + tcp_from_box

    return box_center, tcp_for, tcp_from_box


def plan_place_leg(runner, ik_data, name, end_pos, speed, allow_support):
    """들고 있는 상자를 포함해 하강 구간을 계획·검사한다."""
    model, data = runner.model, runner.data
    q = data.qpos[model.jnt_qposadr[runner.arm_jnt_ids]].copy()
    tip, _ = runner.measure()
    ik_base = data.qpos.copy()
    ik_base[runner.checker.box_adr:runner.checker.box_adr + 3] = [5.0, 5.0, 0.5]  # IK 수렴용으로만 치움
    segments, message = plan_legs(model, ik_data, ik_base, q, tip, [(name, end_pos, speed)])
    if segments is None:
        return None, f"계획 실패: {message}"
    ok, worst, info = runner.checker.check_place_path(ik_data, data.qpos, segments[0]["joint_path"], allow_support)
    if not ok:
        return None, f"들고 있는 상자 포함 검사에서 충돌: {info}"
    robot, other, support = worst
    text = (f"로봇-환경 {fmt_mm(robot)}, 상자-다른 물체 {fmt_mm(other)}, 상자-지지면 "
            f"{'접촉 허용' if allow_support else fmt_mm(support)}, 가장 가까운 곳: {info or '-'}")
    if min(robot, other) < PLACE_CLEARANCE:
        return None, f"여유 거리 부족 ({PLACE_CLEARANCE * 1000:.0f} mm 미만): {text}"
    runner.add_payload(segments, ik_data, env.BOX_MASS)
    return segments, text


def run_place(runner):
    model, data = runner.model, runner.data
    pick = run_pick_lift(runner)
    if not pick["success"]:
        return {"success": False, "reason": f"Pick & Lift 실패: {pick['reason']}"}

    zone, counts = choose_target(model, data, runner.forced_zone)
    if zone is None:
        return {"success": False, "reason": "놓을 빈 구역이 없음"}
    runner.target_zone = zone
    counts_before = dict(counts)

    rel_pos, rel_rot = runner.box_in_hand()
    runner.checker = PlaceChecker(model, rel_pos, rel_rot, runner.support_geom)
    box_target, tcp_for, tcp_from_box = place_pose(rel_pos, rel_rot, zone, runner.support_top)
    runner.target_place = box_target
    print(f"손에 대한 상자 위치(실측): ({rel_pos[0] * 1000:+.1f}, {rel_pos[1] * 1000:+.1f}, {rel_pos[2] * 1000:+.1f}) mm"
          f" -> 상자 중심에서 손끝까지 ({tcp_from_box[0] * 1000:+.1f}, {tcp_from_box[1] * 1000:+.1f}, {tcp_from_box[2] * 1000:+.1f}) mm")
    print(f"놓기 목표 (상자 중심): ({box_target[0]:.3f}, {box_target[1]:+.3f}, {box_target[2]:.4f}) "
          f"/ 이때 손끝: z {tcp_for(0.0)[2]:.4f} (next_place_target 기준 {env.next_place_target(zone, 0)[2]:.4f})")

    # ---- 운반 (7단계와 같음) ----
    ik_data = mujoco.MjData(model)
    route, _ = plan_transport(runner, ik_data, zone, counts)
    if route is None:
        return {"success": False, "reason": "상자를 든 채 충돌 없이 갈 수 있는 경로가 없음"}
    runner.path_points = np.concatenate([s["tcp_path"] for s in route["segments"]])
    runner.markers = [(route["hover"], env.ZONE_COLORS[zone][:3] + [0.8])]
    runner.add_payload(route["segments"], ik_data, env.BOX_MASS)
    print(f"\n[7] {zone} 위로 운반")
    runner.reset_transport_stats()
    if not runner.step("운반", "운반", route["segments"]):
        return {"success": False, "reason": "운반 중 정지"}
    transport = dict(runner.tstats)
    if not runner.step("정지", "목표 위 유지", [("정지", "목표 위에서 잠시 정지", HOVER_PAUSE)]):
        return {"success": False, "reason": "목표 위 정지 중 문제"}

    # ---- 하강 ----
    print(f"\n[8] 상자 바닥이 지지면 {PRECONTACT_GAP * 1000:.0f} mm 위에 올 때까지 하강 (접촉 없어야 함)")
    segments, text = plan_place_leg(runner, ik_data, "접촉 직전까지 하강", tcp_for(PRECONTACT_GAP), DESCENT_SPEED, False)
    print(f"  {'계획 성공' if segments else '거부'}: {text}")
    if segments is None:
        return {"success": False, "reason": "하강 경로 거부"}
    runner.path_points = segments[0]["tcp_path"]
    runner.reset_phase_stats()
    if not runner.step("하강", "하강", segments):
        return {"success": False, "reason": "하강 중 정지"}

    print(f"\n[9] 접촉 하강: {CONTACT_SPEED * 1000:.0f} mm/s로 내려가다 상자-지지면 접촉을 감지하면 정지 "
          f"(최대 {PRESS_LIMIT * 1000:.0f} mm 더)")
    segments, text = plan_place_leg(runner, ik_data, "접촉 하강", tcp_for(-PRESS_LIMIT), CONTACT_SPEED, True)
    print(f"  {'계획 성공' if segments else '거부'}: {text}")
    if segments is None:
        return {"success": False, "reason": "접촉 하강 경로 거부"}
    runner.path_points = segments[0]["tcp_path"]
    runner.contact_run = 0
    runner.reset_phase_stats()
    if not runner.step("접촉 하강", "접촉 하강", segments):
        return {"success": False, "reason": "접촉 하강 중 정지"}
    if runner.contact_run < CONTACT_STEPS:
        return {"success": False, "reason": f"{PRESS_LIMIT * 1000:.0f} mm 더 내려가도 상자가 지지면에 닿지 않음"}
    tip, _ = runner.measure()
    print(f"  접촉 감지: 첫 접촉 = {runner.first_contact}, 그때 손끝 z {tip[2]:.4f} (계산된 접촉 높이 {tcp_for(0.0)[2]:.4f}), "
          f"상자 바닥-지지면 {runner.box_bottom_gap() * 1000:+.2f} mm")

    # ---- 접촉 확인 ----
    runner.hold_current_pose()  # 더 누르지 않고 지금 자세 유지
    runner.reset_phase_stats()
    if not runner.step("접촉 확인", "접촉 확인", [("정지", "지지면 접촉 유지 확인", CONTACT_SETTLE)]):
        return {"success": False, "reason": "접촉 확인 중 정지"}
    contact = dict(runner.pst)
    stable_contact = contact["support_steps"] == contact["steps"] and contact["max_tilt"] <= MAX_TILT_DEG
    print(f"  지지면 접촉 유지 {contact['support_steps']}/{contact['steps']} 스텝, 상자 기울기 최대 {contact['max_tilt']:.2f}도, "
          f"파고듦 최대 {contact['max_support_pen'] * 1000:.2f} mm -> {'안정' if stable_contact else '불안정'}")
    if not stable_contact:
        return {"success": False, "reason": "지지면 접촉이 안정적이지 않아 그리퍼를 열지 않음"}

    # ---- 그리퍼 열기 ----
    print(f"\n[10] 그리퍼 열기 ({OPEN_TIME:.1f}초)")
    runner.grip_ramp = (GRIP_CLOSE, GRIP_OPEN, OPEN_TIME)
    runner.reset_phase_stats()
    ok = runner.step("놓기", "놓기", [("정지", "그리퍼 열기", OPEN_TIME + OPEN_SETTLE)])
    runner.grip_ramp = None
    runner.data.ctrl[runner.grip_id] = GRIP_OPEN
    if not ok:
        return {"success": False, "reason": "그리퍼 여는 중 정지"}
    release = dict(runner.pst)
    info = runner.contacts()
    released = not (info["left"] or info["right"])
    print(f"  손가락 간격 {runner.finger_gap_mm():.1f} mm, 손가락-상자 접촉 왼쪽 {'O' if info['left'] else 'X'} "
          f"오른쪽 {'O' if info['right'] else 'X'}, 여는 동안 상자 이동 {release['max_move_mm']:.2f} mm "
          f"(위로 {max(release['max_rise_mm'], 0):.2f} mm)")
    if not released:
        return {"success": False, "reason": "그리퍼를 열었는데 손가락이 상자에서 떨어지지 않음"}

    # ---- 후퇴 ----
    print(f"\n[11] 손을 수직으로 {RETREAT_HEIGHT * 100:.0f} cm 빼기 (놓인 상자는 이제 주변 물체로 검사)")
    tip, _ = runner.measure()
    segments = plan_or_fail(runner, ik_data, [("수직 후퇴", tip + np.array([0, 0, RETREAT_HEIGHT]), RETREAT_SPEED)])
    if segments is None:
        return {"success": False, "reason": "후퇴 경로 계획 실패"}
    runner.reset_phase_stats()
    if not runner.step("후퇴", "후퇴", segments):
        return {"success": False, "reason": "후퇴 중 정지"}
    retreat = dict(runner.pst)

    # ---- 안정성 확인 ----
    print(f"\n[12] 손을 뺀 뒤 {STABLE_TIME:.0f}초 동안 상자 관찰")
    runner.reset_phase_stats()
    if not runner.step("안정성", "안정성 확인", [("정지", "상자 안정성 확인", STABLE_TIME)]):
        return {"success": False, "reason": "안정성 확인 중 문제"}
    stable = dict(runner.pst)

    # ---- 판정 ----
    box = data.xpos[runner.box_id].copy()
    cx, cy = env.zone_center(zone)
    err_xy = np.linalg.norm(box[:2] - np.array([cx, cy])) * 1000
    err_z = (box[2] - box_target[2]) * 1000
    inside = abs(box[0] - cx) <= env.ZONE_SIZE / 2 - env.BOX_SIZE / 2 and abs(box[1] - cy) <= env.ZONE_SIZE / 2 - env.BOX_SIZE / 2
    tilt = runner.box_tilt_deg()
    support, _ = runner.box_contacts_split()
    counts_after = env.heights_to_counts(env.measure_zone_heights_ground_truth(model, data))
    count_ok = counts_after[zone] == counts_before[zone] + 1 and all(
        counts_after[z] == counts_before[z] for z in counts_before if z != zone)
    checks = [
        ("상자가 목표 구역 안에 놓임", inside, f"구역 중심에서 x {(box[0] - cx) * 1000:+.1f} mm, y {(box[1] - cy) * 1000:+.1f} mm"),
        (f"상자 중심 수평 오차 {MAX_PLACE_ERROR_MM:.0f} mm 이하", err_xy <= MAX_PLACE_ERROR_MM,
         f"{err_xy:.2f} mm (높이 차이 {err_z:+.2f} mm)"),
        (f"놓인 상자 기울기 {MAX_TILT_DEG:.0f}도 이하", tilt <= MAX_TILT_DEG, f"{tilt:.2f}도"),
        ("첫 접촉이 상자-지지면", runner.first_contact == "상자-지지면", f"{runner.first_contact}"),
        ("지지면을 파고들지 않음", max(contact["max_support_pen"], stable["max_support_pen"]) <= MAX_SUPPORT_PEN,
         f"최대 {max(contact['max_support_pen'], release['max_support_pen'], stable['max_support_pen']) * 1000:.2f} mm"),
        ("그리퍼를 열 때 상자가 튕기거나 끌려 올라가지 않음",
         release["max_move_mm"] <= MAX_RELEASE_MOVE_MM and release["max_rise_mm"] <= MAX_RELEASE_RISE_MM,
         f"이동 {release['max_move_mm']:.2f} mm, 위로 {max(release['max_rise_mm'], 0):.2f} mm"),
        ("그리퍼 개방 후 손가락-상자 접촉 해제", released and retreat["finger_steps"] == 0,
         f"개방 직후 해제 {'O' if released else 'X'}, 후퇴 중 손가락 접촉 {retreat['finger_steps']}스텝"),
        ("후퇴 중 위험한 충돌 없음, 상자 움직이지 않음", retreat["max_move_mm"] <= MAX_STABLE_MOVE_MM,
         f"후퇴 중 상자 이동 {retreat['max_move_mm']:.2f} mm, 로봇-환경 최소 거리 {fmt_mm(retreat['min_robot'])}"),
        (f"손을 뺀 뒤 {STABLE_TIME:.0f}초 동안 안정", stable["max_move_mm"] <= MAX_STABLE_MOVE_MM
         and stable["support_steps"] == stable["steps"] and stable["max_tilt"] <= MAX_TILT_DEG,
         f"이동 {stable['max_move_mm']:.3f} mm, 기울기 최대 {stable['max_tilt']:.2f}도, 지지면 접촉 "
         f"{stable['support_steps']}/{stable['steps']} 스텝"),
        ("개수 측정 함수가 해당 구역 +1 인식", count_ok,
         f"전 {' '.join(f'{z}:{counts_before[z]}' for z in counts_before)} -> 후 "
         f"{' '.join(f'{z}:{counts_after[z]}' for z in counts_after)}"),
    ]
    print("\n===== Pick & Place 결과 =====")
    print(f"  운반: 최소 여유 {fmt_mm(min(transport['min_robot'], transport['min_box']))}, "
          f"손 안 미끄러짐 {transport['max_slip_mm']:.2f} mm")
    for name, ok, value in checks:
        print(f"  [{'성공' if ok else '실패'}] {name}: {value}")
    print(f"  최종 상자 중심 ({box[0]:.4f}, {box[1]:+.4f}, {box[2]:.4f}), 지지면 접촉 {'O' if support else 'X'}")
    success = all(ok for _, ok, _ in checks)
    print(f"  => {'성공' if success else '실패'}")
    return {"success": success, "reason": "" if success else "판정 항목 실패", "err_xy": err_xy, "tilt": tilt}


def print_legend():
    print("[화면 범례]")
    print("  물리 시뮬레이션: 로봇팔, 그리퍼, 주황 상자(자유 물체), 공급 받침대, 팔레트와 팔레트 위 상자")
    print("  시각화 전용 (물리 영향 없음): 회색 점 = 지금 단계의 계획 경로, 하늘색 점 = 실제로 지나온 손끝 경로,")
    print("    자홍색 점 = 실제 손끝, 구역 색 반투명 상자 = 놓기 목표(상자 중심), 구역 색 구슬 = 목표 위 대기 위치")
    print("  상자는 손가락 마찰로 들고, 팔레트에 닿은 뒤 그리퍼를 열어 놓는다 (좌표 고정 없음)")


def parse_args(argv):
    forced, rest, i = None, [], 0
    while i < len(argv):
        if argv[i] == "--zone" and i + 1 < len(argv):
            forced = argv[i + 1].upper()
            i += 2
            continue
        rest.append(argv[i])
        i += 1
    numbers = [a for a in rest if not a.startswith("--")]
    counts = parse_counts(rest) if numbers else {z: 0 for z in env.ZONE_LAYOUT}  # 기본: 빈 팔레트
    return forced, counts


def main():
    forced, counts = parse_args(sys.argv[1:])
    model = build_pick_model(counts)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    if "--headless" in sys.argv:
        runner = PlaceRunner(model, data, None, counts)
        runner.forced_zone = forced
        result = run_place(runner)
        if not result["success"]:
            print(f"\n실패: {result['reason']}")
        return

    print_legend()
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = [0.35, -0.12, 0.33]
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -25
        runner = PlaceRunner(model, data, viewer, counts)
        runner.forced_zone = forced
        try:
            result = run_place(runner)
            if not result["success"]:
                print(f"\n실패: {result['reason']}")
        except KeyboardInterrupt:
            pass

        if viewer.is_running():
            print("\n동작을 마쳤습니다. 마지막 상태를 유지합니다 (물리 시뮬레이션은 계속 진행).")
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
