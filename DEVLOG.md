# DEVLOG

## 2026-10-14 — 1단계: Franka Panda를 MuJoCo Viewer에 띄우기

### 오늘의 목표
- 프로젝트·설치 환경 확인
- MuJoCo Menagerie의 Franka Panda 모델 가져오기
- `run_panda.py`로 로봇팔을 MuJoCo Viewer에 띄우기 (Windows, PowerShell)
- Git 저장소 초기화와 `.gitignore` 작성

### 확인한 환경
- Windows / Python 3.13.1 (`.venv`)
- 설치된 패키지: mujoco 3.15.0, glfw 2.10.2, numpy 2.5.4, PyOpenGL 3.1.10 등 (새로 설치한 패키지 없음)
- Git 2.52.0
- 그래픽: Intel UHD Graphics 620 + NVIDIA GeForce MX150, RAM 약 7.6GB

### 완료한 작업
- Menagerie 저장소에서 `franka_emika_panda` 폴더만 받아 `models/franka_emika_panda/`에 복사
  - 출처: https://github.com/google-deepmind/mujoco_menagerie (커밋 `d7b481bac63d3d600ab404ee74f7474c5efe8a67`)
  - 라이선스: 폴더 안 `LICENSE` 파일(Apache-2.0) 그대로 유지
- `run_panda.py` 작성
  - `scene.xml`(로봇 + 바닥 + 조명)을 불러와서 `home` 키프레임 자세로 초기화
  - `mujoco.viewer.launch_passive`로 뷰어를 열고, 화면은 약 60FPS로 갱신하며 실제 시간 속도로 시뮬레이션
- 실행 테스트 결과
  - 뷰어 창 `MuJoCo : panda scene`이 열렸고, 오류 출력은 없었음
  - 화면 없이 2초 시뮬레이션: home 자세에서 관절 변화는 최대 약 0.0066 rad (자세를 유지함)
  - 자원 사용량(시작 후 안정 상태): 메모리 약 0.6~0.8GB, CPU 약 코어 1개
    - 뷰어만 열어 두어도 CPU를 코어 1개만큼 쓰므로, 이 부하는 MuJoCo 뷰어의 렌더링에서 나오는 것으로 보임
- `git init`과 `.gitignore` 작성 (아직 커밋은 하지 않음)

### 아직 하지 않은 것
- 2×2 팔레트, 상자 모델
- FK, IK
- 적재 위치 결정 알고리즘
- 작업시간 측정

### 실행 방법
```powershell
.\.venv\Scripts\python.exe run_panda.py
```
