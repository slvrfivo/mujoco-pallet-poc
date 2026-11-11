"""README 그림 2장을 시뮬레이션 코드로 다시 그린다 (오프스크린 렌더링, 화면 창 없음).

  assets/ik_position_vs_6d.png
      팔레트 Z1 0층 놓기 목표(이웃 구역은 4층)를 위치 IK와 6D IK로 각각 풀었을 때의 자세.
  assets/place_rejected_4404.png
      "4 4 0 4" 배치에서 Z3에 놓으려 할 때, 운반 전 사전 검사가 계산한 하강 경로 중
      로봇과 주변 물체가 가장 가까워지는 자세와 그 두 점.

그림의 숫자(손 기울기, 접촉 수, 최소 거리)는 실행할 때마다 코드로 다시 계산한다.
Pillow가 필요하다 (그림 저장용, 시뮬레이션 데모에는 필요 없음).

실행 (저장소 최상위에서):
    python docs/render_figures.py
"""

import contextlib
import io
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pallet_sim import env, ik6d  # noqa: E402
from pallet_sim.env import make_scene  # noqa: E402
from pallet_sim.planning import plan_legs  # noqa: E402
from pallet_sim.robot import ARM_JOINTS, count_robot_contacts, solve_ik  # noqa: E402

ASSETS = ROOT / "assets"
W, H = 640, 480


# ------------------------------------------------------------------ 공통 --
def font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def hand_tilt_deg(data):
    """손가락 방향이 수직 아래에서 몇 도 기울었는지 (0 = 정확히 아래)."""
    z = data.body("hand").xmat.reshape(3, 3)[:, 2]
    return math.degrees(math.acos(np.clip(-z[2], -1.0, 1.0)))


def camera(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def add_sphere(scene, pos, radius, rgba):
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [radius, 0, 0], np.asarray(pos, float),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scene.ngeom += 1


def add_line(scene, p1, p2, width, rgba):
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(p1, float), np.asarray(p2, float))
    scene.ngeom += 1


def caption(img, lines, colour=(17, 24, 39)):
    """그림 왼쪽 위에 흰 띠와 글씨를 얹는다."""
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f_title, f_body = font(22), font(18)
    heights = [28] + [24] * (len(lines) - 1)
    d.rectangle((0, 0, W, 12 + sum(heights)), fill=(255, 255, 255))
    y = 8
    for i, text in enumerate(lines):
        d.text((12, y), text, font=f_title if i == 0 else f_body, fill=colour if i == 0 else (55, 65, 81))
        y += heights[i]
    return im


def side_by_side(images, gap=12):
    w = sum(im.width for im in images) + gap * (len(images) - 1)
    sheet = Image.new("RGB", (w, images[0].height), "white")
    x = 0
    for im in images:
        sheet.paste(im, (x, 0))
        x += im.width + gap
    return sheet


# ------------------------------------------------ 1) 위치 IK vs 6D IK --
def render_ik_comparison():
    counts = {z: env.MAX_LAYERS for z in env.ZONE_LAYOUT}
    counts["Z1"] = 0
    model, data = make_scene(counts)
    target = env.next_place_target("Z1", 0)
    ids = model.jnt_qposadr[[model.joint(n).id for n in ARM_JOINTS]]
    renderer = mujoco.Renderer(model, H, W)
    cam = camera([0.42, 0.08, 0.33], 1.0, 280, -12)

    panels = []
    for name, solve in (("Position IK", solve_ik), ("6D IK (gripper pointing down)", ik6d.solve_ik_6d)):
        ik_data = mujoco.MjData(model)
        result = solve(model, ik_data, data.qpos.copy(), target)
        pose = mujoco.MjData(model)
        pose.qpos[:] = data.qpos
        pose.qpos[ids] = result[1]
        mujoco.mj_forward(model, pose)
        tilt, contacts = hand_tilt_deg(pose), count_robot_contacts(model, pose)
        renderer.update_scene(pose, cam)
        add_sphere(renderer.scene, target, 0.012, [1.0, 0.85, 0.1, 0.9])  # 놓기 목표 (시각화 전용)
        status = "collision" if contacts else "no contact"
        panels.append(caption(renderer.render(), [name, f"hand tilt {tilt:.1f} deg, robot contacts {contacts} ({status})"]))
        print(f"{name}: tilt {tilt:.2f} deg, robot contacts {contacts}")
    renderer.close()
    out = ASSETS / "ik_position_vs_6d.png"
    side_by_side(panels).save(out, optimize=True)
    print(f"wrote {out.relative_to(ROOT).as_posix()}")


# --------------------------------------- 2) "4 4 0 4" 운반 전 거부 자세 --
REJECT_AZIMUTH = 135
CLOSEUP_AZIMUTH = 0


def rejection_pose():
    """Pick & Lift를 실제로 시뮬레이션한 뒤, 사전 검사가 거부한 하강 경로에서 가장 가까운 자세를 찾는다."""
    import pick_place_demo as ppd
    from pick_lift_demo import build_pick_model, run_pick_lift
    from pick_transport_demo import choose_target, plan_transport

    counts = {"Z1": 4, "Z2": 4, "Z3": 0, "Z4": 4}
    model = build_pick_model(counts)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    runner = ppd.PlaceRunner(model, data, None, counts)
    runner.forced_zone = None

    with contextlib.redirect_stdout(io.StringIO()):  # 데모의 단계별 출력은 숨긴다
        pick = run_pick_lift(runner)
        if not pick["success"]:
            raise SystemExit(f"Pick & Lift 실패: {pick['reason']}")
        zone, measured = choose_target(model, data, None)
        rel_pos, rel_rot = runner.box_in_hand()
        runner.checker = ppd.PlaceChecker(model, rel_pos, rel_rot, runner.support_geom)
        _, tcp_for, _ = ppd.place_pose(rel_pos, rel_rot, zone, runner.support_top)
        ik_data = mujoco.MjData(model)
        route, _ = plan_transport(runner, ik_data, zone, measured)
        ok, text = ppd.precheck_place_legs(runner, ik_data, route, tcp_for)
    if ok:
        raise SystemExit("사전 검사가 통과했습니다. 이 그림은 거부되는 배치를 보여 주기 위한 것입니다.")

    # 사전 검사와 같은 두 하강 구간(접촉 직전까지, 접촉 하강)을 다시 계획해서,
    # 자세마다 로봇-환경 최소 거리와 그 두 점을 찾는다.
    checker = runner.checker
    q_end = route["segments"][-1]["joint_path"][-1]
    ik_base = data.qpos.copy()
    ik_base[checker.box_adr:checker.box_adr + 3] = [5.0, 5.0, 0.5]
    segments, message = plan_legs(model, ik_data, ik_base, q_end, route["hover"],
                                  [("descent", tcp_for(ppd.PRECONTACT_GAP), ppd.DESCENT_SPEED),
                                   ("contact descent", tcp_for(-ppd.PRESS_LIMIT), ppd.CONTACT_SPEED)])
    if segments is None:
        raise SystemExit(f"하강 구간 계획 실패: {message}")
    joint_path = np.concatenate([s["joint_path"] for s in segments])

    pose = mujoco.MjData(model)
    fromto = np.zeros(6)
    best = (np.inf, None, None, None)
    for q_arm in joint_path:
        pose.qpos[:] = data.qpos
        pose.qpos[checker.qpos_ids] = q_arm
        mujoco.mj_kinematics(model, pose)
        checker.place_box(pose)
        mujoco.mj_fwdPosition(model, pose)
        for g1 in checker.robot_geoms:
            for g2 in checker.other_geoms:
                dist = mujoco.mj_geomDistance(model, pose, g1, g2, 0.05, fromto)
                if dist < best[0]:
                    best = (dist, q_arm.copy(), fromto.copy(), (checker.name(g1), checker.name(g2)))
    dist, q_arm, points, pair = best
    print(f"closest during descent: {dist * 1000:.1f} mm ({pair[0]} - {pair[1]}); precheck said: {text}")

    pose.qpos[:] = data.qpos
    pose.qpos[checker.qpos_ids] = q_arm
    mujoco.mj_kinematics(model, pose)
    checker.place_box(pose)
    mujoco.mj_forward(model, pose)
    return model, pose, points, dist, pair


def render_place_rejection():
    model, pose, points, dist, pair = rejection_pose()
    renderer = mujoco.Renderer(model, H, W)
    mid = (points[:3] + points[3:]) / 2
    views = [camera(mid, 0.9, REJECT_AZIMUTH, -18), camera(mid, 0.3, CLOSEUP_AZIMUTH, 0)]
    panels = []
    for i, cam in enumerate(views):
        renderer.update_scene(pose, cam)
        add_line(renderer.scene, points[:3], points[3:], 0.002, [0.9, 0.1, 0.1, 1.0])
        add_sphere(renderer.scene, points[:3], 0.008, [0.9, 0.1, 0.1, 1.0])
        add_sphere(renderer.scene, points[3:], 0.008, [0.9, 0.1, 0.1, 1.0])
        lines = (["Pallet 4 4 0 4, place in Z3: rejected before transport",
                  f"closest robot-environment gap {dist * 1000:.1f} mm (limit 10 mm)"] if i == 0 else
                 ["Close-up of the closest point (red)", f"{pair[0]} - {pair[1]}"])
        panels.append(caption(renderer.render(), lines))
    renderer.close()
    out = ASSETS / "place_rejected_4404.png"
    side_by_side(panels).save(out, optimize=True)
    print(f"wrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    render_ik_comparison()
    render_place_rejection()
