"""Franka Panda 로봇팔을 MuJoCo Viewer에 띄우는 1단계 스크립트.

실행 (PowerShell, 프로젝트 폴더에서):
    .\\.venv\\Scripts\\python.exe run_panda.py

뷰어 창을 닫으면 프로그램이 종료된다.
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer

# 이 파일이 있는 폴더를 기준으로 경로를 잡아, 어디서 실행해도 모델을 찾을 수 있게 한다.
PROJECT_DIR = Path(__file__).resolve().parent
SCENE_XML = PROJECT_DIR / "models" / "franka_emika_panda" / "scene.xml"


def main():
    # 1) XML 파일을 읽어 로봇 모델(model)과 시뮬레이션 상태(data)를 만든다.
    #    model: 링크 길이, 관절, 질량 등 변하지 않는 정보
    #    data : 관절 각도, 속도 등 시뮬레이션 중에 계속 바뀌는 정보
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    # 2) 모델에 정의된 "home" 자세(키프레임)로 로봇을 초기화한다.
    #    관절 각도(qpos)와 모터 목표값(ctrl)이 함께 설정되어 로봇이 이 자세를 유지한다.
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)

    print(f"모델 로드 완료: {SCENE_XML}")
    print(f"관절 수: {model.njnt}, 액추에이터 수: {model.nu}, timestep: {model.opt.timestep}s")

    # 화면은 초당 60번만 갱신한다. 물리 계산(0.002초 간격)마다 그리면
    # 초당 500번 그리게 되어 내장 그래픽/저사양 PC에서 CPU를 많이 쓴다.
    frame_time = 1.0 / 60
    steps_per_frame = max(1, round(frame_time / model.opt.timestep))  # 약 8스텝

    # 3) 뷰어를 열고, 창이 열려 있는 동안 시뮬레이션을 진행한다.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            frame_start = time.time()

            # 한 프레임(약 1/60초) 분량의 물리 시뮬레이션을 진행
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            viewer.sync()  # 바뀐 상태를 화면에 반영

            # 실제 시간과 비슷한 속도로 움직이도록 남은 시간만큼 쉰다.
            # (쉬지 않으면 CPU를 100% 쓰면서 너무 빠르게 돌아간다.)
            remaining = steps_per_frame * model.opt.timestep - (time.time() - frame_start)
            if remaining > 0:
                time.sleep(remaining)

    print("뷰어가 닫혀 프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
