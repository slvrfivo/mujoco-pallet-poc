"""6단계: Pick & Lift - 상자 1개를 실제 그리퍼로 잡아 10cm 들어 올리고 3초 유지.

상자는 자유 물체(free joint)이고, 손가락과 상자 사이의 접촉력·마찰만으로 들어 올린다.
(weld, equality 구속, 상자 좌표 직접 변경 같은 방법은 쓰지 않는다.)

실행 (PowerShell, 프로젝트 폴더에서):
    .\\.venv\\Scripts\\python.exe pick_lift_demo.py              # 뷰어, 끝나면 마지막 상태 유지
    .\\.venv\\Scripts\\python.exe pick_lift_demo.py --headless   # 화면 없이 물리 검증
    .\\.venv\\Scripts\\python.exe dev_run.py --manual pick_lift_demo.py   # R 키로 다시 실행
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
from pallet_reach_demo import (FINAL_SPEED, SAFE_Z, TRANSIT_SPEED, VERTICAL_SPEED, Runner,
                               plan_legs)

# ---------------------------------------------------------- 장면 설정 --
# 공급 받침대: 팔레트(x 0.34~0.66, y -0.16~0.16) 오른쪽에 떨어뜨려 둔다.
SUPPLY_XY = np.array([0.45, -0.30])
SUPPLY_SIZE = 0.12                 # 받침대 윗면 한 변
SUPPLY_TOP_Z = env.PALLET_TOP_Z    # 0.25 m (팔레트와 같은 높이)
BOX_START = np.array([*SUPPLY_XY, SUPPLY_TOP_Z + env.BOX_SIZE / 2])  # 상자 중심 처음 위치
SUPPLY_COLOR = [0.25, 0.45, 0.45, 1]   # 청록 회색 받침대
PICK_BOX_COLOR = [0.95, 0.55, 0.15, 1]  # 주황 상자 (팔레트 상자와 구분)

# ---------------------------------------------------------- 동작 설정 --
GRIP_OPEN = 255.0     # 그리퍼 명령: 255 = 손가락 사이 8cm (완전히 열림)
GRIP_CLOSE = 0.0      # 0 = 닫힘 방향. 상자에 막히면 그 자리에서 조이는 힘을 냄
CLOSE_TIME = 1.5      # 손가락을 천천히 닫는 시간 [s]
GRIP_SETTLE = 1.0     # 닫은 뒤 접촉이 안정될 때까지 기다리는 시간 [s]
PAUSE = 1.0           # 단계 사이 대기 [s]
LIFT_HEIGHT = 0.10    # 들어 올릴 높이 [m]
HOLD_TIME = 3.0       # 들어 올린 뒤 유지 시간 [s]
NOSLIP_ITERATIONS = 3 # 미끄러짐 보정 계산 횟수 (build_pick_model 설명 참고). 3회로 30초 유지 시 0.04 mm

# 판정 기준
MIN_LIFT = 0.08                 # 상자가 최소 8cm는 올라가야 함
MAX_PENETRATION = 0.003         # 손가락이 상자를 3mm 넘게 파고들면 물리적으로 불가능한 상태로 봄
MAX_SLIP_MM, MAX_SLIP_DEG = 5.0, 5.0   # 손에 대한 상자 위치·방향 변화 허용치
# 상자에 실제로 닿는 것은 손가락 끝의 작은 돌기 패드 4개(fingertip_pad_collision_2~5)이고,
# 그 안쪽 면은 손가락 몸체 원점과 같은 위치다 (panda.xml: pos y 0.002, 두께 0.002).
FINGER_PAD_INSET = 0.0


def build_pick_model():
    """팔레트 장면(상자 없음) + 공급 받침대 + 집을 상자 1개."""
    spec = env.build_spec({zone: 0 for zone in env.ZONE_LAYOUT})
    BOX = mujoco.mjtGeom.mjGEOM_BOX
    spec.worldbody.add_geom(
        name="supply_table", type=BOX,
        size=[SUPPLY_SIZE / 2, SUPPLY_SIZE / 2, SUPPLY_TOP_Z / 2],
        pos=[*SUPPLY_XY, SUPPLY_TOP_Z / 2], rgba=SUPPLY_COLOR,
    )
    box = spec.worldbody.add_body(name="pick_box", pos=BOX_START)
    box.add_freejoint()  # 진짜 자유 물체: 중력·접촉·마찰로만 움직인다
    box.add_geom(name="pick_box_geom", type=BOX, size=[env.BOX_SIZE / 2] * 3,
                 mass=env.BOX_MASS, rgba=PICK_BOX_COLOR)  # 마찰계수는 MuJoCo 기본값 1.0
    model = spec.compile()
    # 미끄러짐 보정 (noslip): MuJoCo 접촉은 약간 말랑하게 계산돼서, 그리퍼처럼 조이는 힘이 작으면
    # 마찰 여유가 충분해도 상자가 천천히 미끄러진다 (기본 설정에서 초당 약 0.84 mm, 약 25초 후 낙하).
    # 조이는 힘 약 2.35 N x 마찰 1.0 > 상자 무게 0.98 N 이므로 실제로는 미끄러지지 않아야 하는 상황이라,
    # MuJoCo의 noslip 계산을 켜서 이 계산 오차를 없앤다. (마찰계수·그리퍼·로봇 모델은 바꾸지 않음)
    model.opt.noslip_iterations = NOSLIP_ITERATIONS
    return model


class PickRunner(Runner):
    """Runner(5단계)의 경로 재생을 그대로 쓰고, 그리퍼 제어와 접촉 판정만 바꾼다."""

    def __init__(self, model, data, viewer):
        super().__init__(model, data, viewer, {zone: 0 for zone in env.ZONE_LAYOUT})
        self.grip_id = model.actuator("actuator8").id
        self.box_id = model.body("pick_box").id
        self.left_id = model.body("left_finger").id
        self.right_id = model.body("right_finger").id
        self.robot_root = model.body("link0").id
        self.table_geom = model.geom("supply_table").id
        self.finger_qpos = model.jnt_qposadr[[model.joint("finger_joint1").id, model.joint("finger_joint2").id]]
        self.phase = "준비"           # 지금 단계 (접촉 허용 여부가 단계마다 다름)
        self.grip_ramp = None         # (시작 명령, 끝 명령, 시간) - 손가락을 천천히 닫을 때 사용
        self.markers = []             # (위치, 색) - 시각화 전용
        self.reset_stats()

    # ------------------------------------------------------------ 측정 --
    def contacts(self):
        """현재 접촉을 종류별로 나눈다."""
        m, d = self.model, self.data
        info = {"left": False, "right": False, "box_table": False, "box_floor": False,
                "penetration": 0.0, "danger": []}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            robot1 = m.body_rootid[b1] == self.robot_root
            robot2 = m.body_rootid[b2] == self.robot_root
            if self.box_id in (b1, b2):
                other_body = b2 if b1 == self.box_id else b1
                other_geom = c.geom2 if b1 == self.box_id else c.geom1
                if other_body == self.left_id or other_body == self.right_id:
                    side = "left" if other_body == self.left_id else "right"
                    info[side] = True  # 손가락-상자: 파지 단계에서는 정상 접촉
                    info["penetration"] = max(info["penetration"], -c.dist)
                elif robot1 or robot2:
                    info["danger"].append(f"{m.body(other_body).name}-상자")  # 손바닥·팔이 상자에 닿음
                elif other_geom == self.table_geom:
                    info["box_table"] = True  # 상자-받침대: 정상 (놓여 있을 때)
                elif m.geom(other_geom).name == "floor":
                    info["box_floor"] = True
            elif robot1 or robot2:
                names = sorted(m.body(b).name or m.geom(g).name for b, g in ((b1, c.geom1), (b2, c.geom2)))
                info["danger"].append("-".join(names))  # 로봇-받침대/팔레트/바닥/자기 몸
        return info

    def finger_gap_mm(self):
        """두 손가락 패드 안쪽 면 사이 거리 [mm]."""
        q1, q2 = self.data.qpos[self.finger_qpos]
        return (q1 + q2 + 2 * FINGER_PAD_INSET) * 1000

    def box_in_hand(self):
        """손(hand) 좌표계에서 본 상자 위치와 방향. 손에 대해 상자가 미끄러지는지 볼 때 사용."""
        hand_pos = self.data.body("hand").xpos
        hand_rot = self.data.body("hand").xmat.reshape(3, 3)
        box_pos = self.data.xpos[self.box_id]
        box_rot = self.data.xmat[self.box_id].reshape(3, 3)
        return hand_rot.T @ (box_pos - hand_pos), hand_rot.T @ box_rot

    def reset_stats(self):
        self.stats = {"steps": 0, "both_contact_steps": 0, "box_table_steps": 0, "max_pen": 0.0,
                      "max_slip_mm": 0.0, "max_slip_deg": 0.0, "limit_steps": 0, "ref": None}

    # ------------------------------------------- Runner 확장 지점 (매 스텝) --
    def before_step(self, seg, seg_time):
        if self.grip_ramp is not None:
            start, end, duration = self.grip_ramp
            s = min(seg_time / duration, 1.0)
            self.data.ctrl[self.grip_id] = start + s * (end - start)

    def safety_check(self):
        info = self.contacts()
        if info["danger"]:
            return f"위험한 접촉: {', '.join(sorted(set(info['danger'])))}"
        if self.phase == "접근" and (info["left"] or info["right"]):
            return "그리퍼를 닫기 전에 손가락이 상자에 닿음 (접근 위치가 어긋남)"
        if info["penetration"] > MAX_PENETRATION:
            return f"손가락이 상자를 {info['penetration'] * 1000:.1f} mm 파고듦 (물리적으로 불가능한 상태)"

        # 들어 올리기·유지 단계 기록
        if self.phase in ("들어 올리기", "유지"):
            st = self.stats
            st["steps"] += 1
            st["both_contact_steps"] += info["left"] and info["right"]
            st["box_table_steps"] += info["box_table"]
            st["max_pen"] = max(st["max_pen"], info["penetration"])
            rel_pos, rel_rot = self.box_in_hand()
            if st["ref"] is None:
                st["ref"] = (rel_pos.copy(), rel_rot.copy())
            st["max_slip_mm"] = max(st["max_slip_mm"], np.linalg.norm(rel_pos - st["ref"][0]) * 1000)
            st["max_slip_deg"] = max(st["max_slip_deg"], ik6d.angle_between_deg(st["ref"][1], rel_rot))
            q = self.data.qpos[self.model.jnt_qposadr[self.arm_jnt_ids]]
            rng = self.model.jnt_range[self.arm_jnt_ids]
            st["limit_steps"] += bool(np.any(q < rng[:, 0]) or np.any(q > rng[:, 1]))
            if not (info["left"] or info["right"]):
                return "상자를 놓침 (양쪽 손가락 모두 접촉 없음)"
        return None

    def status_text(self):
        info = self.contacts()
        box_z = self.data.xpos[self.box_id][2]
        return (f"\n           상자 z {box_z:.3f} m (처음보다 {(box_z - BOX_START[2]) * 100:+5.1f} cm) | "
                f"손가락 간격 {self.finger_gap_mm():4.1f} mm | 손가락 접촉 왼쪽 {'O' if info['left'] else 'X'} "
                f"오른쪽 {'O' if info['right'] else 'X'} | 상자-받침대 {'접촉' if info['box_table'] else '떨어짐'}")

    def draw(self):
        scn = self.viewer.user_scn
        scn.ngeom = 0
        points = [(p, [0.6, 0.6, 0.6, 0.7], 0.003) for p in self.path_points[::3]]          # 계획 경로
        points += [(p, rgba, 0.012) for p, rgba in self.markers]                             # 목표 지점
        points.append((ik6d.get_fingertip_pose(self.data)[0], [1.0, 0.0, 1.0, 1.0], 0.007))  # 실제 손끝
        for pos, rgba, radius in points:
            if scn.ngeom >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                size=[radius, 0, 0], pos=pos, mat=np.eye(3).flatten(),
                                rgba=np.array(rgba, dtype=np.float32))
            scn.ngeom += 1

    # -------------------------------------------------------- 동작 도구 --
    def hold_segment(self, name, duration):
        """지금 모터 목표를 유지하는 정지 구간."""
        tip, _ = self.measure()
        return {"name": name, "ctrl_path": np.array([self.data.ctrl[self.arm_act_ids].copy()]),
                "joint_path": np.array([self.data.qpos[self.model.jnt_qposadr[self.arm_jnt_ids]].copy()]),
                "tcp_path": np.array([tip]), "duration": duration}

    def add_payload(self, segments, ik_data, mass):
        """상자 무게를 버티는 관절 토크를 모터 목표에 더한다.

        상자 무게 F = (0, 0, -m g)가 손끝에 걸리면 관절에는 Jᵀ F 만큼의 토크가 생긴다.
        모터가 이를 이기려면 -Jᵀ F 만큼 더 힘을 내야 하고, 위치 서보에서는 목표각을 (그 토크 / kp)만큼 올리면 된다.
        """
        qpos_ids = self.model.jnt_qposadr[self.arm_jnt_ids]
        dof_ids = self.model.jnt_dofadr[self.arm_jnt_ids]
        kp = self.model.actuator_gainprm[self.arm_act_ids, 0]
        weight = np.array([0.0, 0.0, -mass * 9.81])
        largest = 0.0
        for seg in segments:
            for k, q in enumerate(seg["joint_path"]):
                ik_data.qpos[qpos_ids] = q
                mujoco.mj_kinematics(self.model, ik_data)
                mujoco.mj_comPos(self.model, ik_data)
                tip, _ = ik6d.get_fingertip_pose(ik_data)
                jacp = np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, ik_data, jacp, None, tip, self.model.body("hand").id)
                torque = -jacp[:, dof_ids].T @ weight
                seg["ctrl_path"][k] = seg["ctrl_path"][k] + torque / kp
                largest = max(largest, np.max(np.abs(torque)))
        return largest

    def step(self, title, phase, segments):
        """여러 구간을 재생한다. 실패하면 이유를 출력하고 False.

        ("정지", 이름, 시간) 형태는 그 구간을 시작하는 순간의 자세로 정지 구간을 만든다.
        """
        self.phase = phase
        for seg in segments:
            if isinstance(seg, tuple):
                seg = self.hold_segment(seg[1], seg[2])
            print(f"  > {seg['name']} ({seg['duration']:.1f}초)")
            status, _ = self.play(seg)
            if status == "closed":
                raise KeyboardInterrupt
            if status != "ok":
                return False
        return True


def plan_or_fail(runner, ik_data, legs, carrying_box=False):
    """현재 자세에서 legs를 계획한다. 실패하면 None.

    carrying_box=True: 손에 쥔 상자는 계획용 데이터(ik_data)에서만 멀리 치워 두고 계획한다.
    (쥔 상자와 손가락의 정상 접촉을 '충돌'로 보지 않기 위함. 실제 시뮬레이션의 상자는 그대로.)
    한계: 쥔 상자 자체가 다른 물체에 부딪히는지는 계획 단계에서 보지 않고, 실행 중 접촉 감시로만 확인한다.
    """
    model, data = runner.model, runner.data
    q = data.qpos[model.jnt_qposadr[runner.arm_jnt_ids]].copy()
    tip, _ = runner.measure()
    base_qpos = data.qpos.copy()
    if carrying_box:
        adr = model.jnt_qposadr[model.body("pick_box").jntadr[0]]
        base_qpos[adr:adr + 3] = [5.0, 5.0, 0.5]
    segments, message = plan_legs(model, ik_data, base_qpos, q, tip, legs)
    if segments is None:
        print(f"  계획 실패: {message} -> 움직이지 않고 정지합니다.")
        return None
    clear = min(s["clearance"] for s in segments if s["clearance"] is not None) * 1000
    print(f"  계획 성공: 경로 점 {sum(len(s['joint_path']) for s in segments)}개 충돌 없음, 주변 물체와 최소 거리 {clear:.1f} mm")
    runner.path_points = np.concatenate([s["tcp_path"] for s in segments])
    return segments


def run_pick_lift(runner):
    """그리퍼 열기 -> 접근 -> 하강 -> 파지 -> 상승 -> 유지. 반환: 결과 dict (성공 여부 포함)"""
    model, data = runner.model, runner.data
    ik_data = mujoco.MjData(model)  # 계획 전용 (실제 시뮬레이션과 분리)
    above = np.array([*SUPPLY_XY, SAFE_Z])
    approach = BOX_START + np.array([0.0, 0.0, 0.10])
    grasp = BOX_START.copy()  # 손끝(=손가락 패드 중심 높이)을 상자 중심에 맞춤
    lifted = grasp + np.array([0.0, 0.0, LIFT_HEIGHT])
    runner.markers = [(approach, [0.2, 0.6, 1.0, 0.6]), (lifted, [0.2, 0.9, 0.3, 0.6])]
    result = {"success": False, "reason": ""}

    print(f"\n상자 처음 위치 (중심): ({BOX_START[0]:.3f}, {BOX_START[1]:+.3f}, {BOX_START[2]:.3f}), "
          f"크기 {env.BOX_SIZE * 100:.0f} cm, 무게 {env.BOX_MASS * 1000:.0f} g")

    print("\n[1/6] 그리퍼 열기")
    data.ctrl[runner.grip_id] = GRIP_OPEN
    if not runner.step("열기", "접근", [("정지", "그리퍼 열린 상태 확인", PAUSE)]):
        return result | {"reason": "그리퍼 열기 중 정지"}
    print(f"    손가락 간격 {runner.finger_gap_mm():.1f} mm (상자 폭 {env.BOX_SIZE * 1000:.0f} mm)")

    print("\n[2/6] 상자 위 안전 높이로 이동 후 대기 위치로 하강")
    segments = plan_or_fail(runner, ik_data, [("상자 위 이동 높이로 이동", above, TRANSIT_SPEED),
                                              ("상자 위 대기 위치로 하강", approach, VERTICAL_SPEED)])
    if segments is None or not runner.step("접근", "접근", segments + [("정지", "대기", PAUSE)]):
        return result | {"reason": "접근 실패"}

    print("\n[3/6] 상자 양옆을 잡을 높이로 하강 (손가락이 상자 옆을 지나 내려감)")
    segments = plan_or_fail(runner, ik_data, [("파지 높이로 하강", grasp, FINAL_SPEED)])
    if segments is None or not runner.step("하강", "접근", segments + [("정지", "대기", PAUSE)]):
        return result | {"reason": "하강 실패"}
    err_mm, err_deg = runner.arrival_error(grasp)
    print(f"    파지 위치 도착 오차: {err_mm:.2f} mm, {err_deg:.2f}도")

    print("\n[4/6] 손가락 닫기 (실제 그리퍼 모터, 접촉력과 마찰로 파지)")
    runner.grip_ramp = (GRIP_OPEN, GRIP_CLOSE, CLOSE_TIME)
    ok = runner.step("파지", "파지", [("정지", "손가락 닫기", CLOSE_TIME + GRIP_SETTLE)])
    runner.grip_ramp = None
    data.ctrl[runner.grip_id] = GRIP_CLOSE
    if not ok:
        return result | {"reason": "파지 중 정지"}
    info = runner.contacts()
    gap = runner.finger_gap_mm()
    print(f"    파지 확인: 손가락 간격 {gap:.1f} mm, 왼쪽 접촉 {'O' if info['left'] else 'X'}, "
          f"오른쪽 접촉 {'O' if info['right'] else 'X'}, 파고든 깊이 {info['penetration'] * 1000:.2f} mm")
    if not (info["left"] and info["right"]):
        return result | {"reason": "양쪽 손가락이 상자에 닿지 않음 (파지 실패)"}
    if gap < env.BOX_SIZE * 1000 - MAX_PENETRATION * 2000:
        return result | {"reason": f"손가락 간격 {gap:.1f} mm가 상자 폭보다 너무 작음 (상자를 뚫고 들어감)"}

    print("\n[5/6] 상자 들어 올리기 (10 cm)")
    segments = plan_or_fail(runner, ik_data, [("10 cm 들어 올리기", lifted, FINAL_SPEED)], carrying_box=True)
    if segments is None:
        return result | {"reason": "상승 계획 실패"}
    torque = runner.add_payload(segments, ik_data, env.BOX_MASS)
    print(f"    상자 무게 보상: 관절 토크 최대 {torque:.3f} N·m 를 모터 목표에 더함")
    box_z0 = data.xpos[runner.box_id][2]
    runner.reset_stats()
    if not runner.step("상승", "들어 올리기", segments):
        return result | {"reason": "들어 올리는 중 정지", "box_lift_cm": (data.xpos[runner.box_id][2] - box_z0) * 100}

    print(f"\n[6/6] {HOLD_TIME:.0f}초 유지")
    lift_stats = dict(runner.stats)
    runner.reset_stats()
    if not runner.step("유지", "유지", [("정지", f"{HOLD_TIME:.0f}초 유지", HOLD_TIME)]):
        return result | {"reason": "유지 중 정지"}
    hold = runner.stats

    # ---- 판정 ----
    info = runner.contacts()
    box_z = data.xpos[runner.box_id][2]
    lift = box_z - BOX_START[2]
    box_bottom_gap = box_z - env.BOX_SIZE / 2 - SUPPLY_TOP_Z
    tip, _ = runner.measure()
    checks = [
        ("상자가 받침대에서 떨어짐 (유지 중 받침대 접촉 0스텝)", hold["box_table_steps"] == 0 and box_bottom_gap > 0.01,
         f"받침대 접촉 {hold['box_table_steps']}스텝, 상자 바닥-받침대 간격 {box_bottom_gap * 100:.1f} cm"),
        (f"상자 상승 {MIN_LIFT * 100:.0f} cm 이상", lift >= MIN_LIFT, f"처음보다 {lift * 100:.2f} cm 상승"),
        ("유지 중 양쪽 손가락 접촉 유지", hold["both_contact_steps"] == hold["steps"],
         f"{hold['both_contact_steps']}/{hold['steps']} 스텝"),
        ("유지 끝까지 떨어지지 않음", info["left"] and info["right"] and not info["box_floor"],
         f"마지막 접촉 왼쪽 {'O' if info['left'] else 'X'} 오른쪽 {'O' if info['right'] else 'X'}"),
        (f"손에 대한 상자 위치 변화 {MAX_SLIP_MM:.0f} mm, {MAX_SLIP_DEG:.0f}도 이하 (상승+유지)",
         max(lift_stats["max_slip_mm"], hold["max_slip_mm"]) <= MAX_SLIP_MM
         and max(lift_stats["max_slip_deg"], hold["max_slip_deg"]) <= MAX_SLIP_DEG,
         f"상승 중 {lift_stats['max_slip_mm']:.2f} mm / {lift_stats['max_slip_deg']:.2f}도, "
         f"유지 중 {hold['max_slip_mm']:.2f} mm / {hold['max_slip_deg']:.2f}도"),
        ("위험한 접촉·관절 한계 위반 없음", lift_stats["limit_steps"] + hold["limit_steps"] == 0,
         f"위험 접촉 0 (있었다면 비상 정지), 관절 한계 위반 {lift_stats['limit_steps'] + hold['limit_steps']}스텝"),
        (f"손가락이 상자를 {MAX_PENETRATION * 1000:.0f} mm 넘게 파고들지 않음", max(lift_stats["max_pen"], hold["max_pen"]) <= MAX_PENETRATION,
         f"최대 {max(lift_stats['max_pen'], hold['max_pen']) * 1000:.2f} mm"),
    ]
    print("\n===== Pick & Lift 결과 =====")
    for name, ok, value in checks:
        print(f"  [{'성공' if ok else '실패'}] {name}: {value}")
    print(f"  참고: 손끝 높이 {tip[2]:.3f} m, 상자 중심 높이 {box_z:.3f} m, 손가락 간격 {runner.finger_gap_mm():.1f} mm")
    success = all(ok for _, ok, _ in checks)
    print(f"  => {'성공' if success else '실패'}")
    return {"success": success, "reason": "" if success else "판정 항목 실패", "lift_cm": lift * 100}


def print_legend():
    print("[화면 범례]")
    print("  물리 시뮬레이션: 로봇팔, 그리퍼 손가락, 주황 상자(자유 물체), 공급 받침대(청록 회색), 팔레트")
    print("  시각화 전용 (물리 영향 없음): 파란 구슬 = 상자 위 대기 위치, 초록 구슬 = 들어 올릴 목표,")
    print("                                회색 점 = 계획된 손끝 경로, 자홍색 점 = 실제 손끝(TCP)")
    print("  상자는 손에 고정하지 않음: 손가락의 접촉력과 마찰로만 들어 올림")


def main():
    model = build_pick_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    if "--headless" in sys.argv:
        result = run_pick_lift(PickRunner(model, data, None))
        if not result["success"]:
            print(f"\n실패: {result['reason']}")
        return

    print_legend()
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        # 손가락이 x 방향으로 벌어지므로, 그에 수직인 오른쪽(-y)에서 봐야 두 손가락 사이의 상자가 보인다.
        viewer.cam.lookat[:] = [0.30, -0.25, 0.33]
        viewer.cam.distance = 1.3
        viewer.cam.azimuth = 105
        viewer.cam.elevation = -18
        runner = PickRunner(model, data, viewer)
        try:
            result = run_pick_lift(runner)
            if not result["success"]:
                print(f"\n실패: {result['reason']}")
        except KeyboardInterrupt:
            pass

        if viewer.is_running():
            print("\n동작을 마쳤습니다. 마지막 상태를 유지합니다 (물리 시뮬레이션은 계속 진행).")
            print("다시 실행: dev_run.py --manual 로 실행했다면 터미널에서 R, 아니면 창을 닫고 명령을 다시 실행하세요.")
            runner.phase = "유지"
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
