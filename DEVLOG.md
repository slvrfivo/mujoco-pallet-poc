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

## 2026-10-14 — 2단계: FK(정기구학) 실습

### 오늘의 목표
- 1단계 결과를 첫 Git 커밋으로 남기기
- `fk_demo.py`: 관절 일부를 천천히 움직이며 손끝 (x, y, z)를 0.5초마다 출력

### 완료한 작업
- 첫 커밋 `d3224ed` "Initial MuJoCo Panda simulation" (Panda 모델 폴더 포함)
  - 모델 출처·원본 커밋·라이선스를 `models/README.md`에 기록, 원본 `LICENSE`는 그대로 유지
  - 이 저장소에만 Git 사용자 정보 설정 (`git config user.name/user.email`)
- 모델 구조 확인
  - 바디 연결: link0 → link1 → … → link7 → hand → left/right_finger
  - 모델에 손끝 site가 없음 → hand 바디 z축으로 0.1034m 떨어진 점을 손끝(TCP)으로 사용 (원본 XML은 수정하지 않음)
  - 관절 한계·모터 목표값 범위(ctrlrange) 확인: joint1 ±2.90, joint2 ±1.76, joint4 −3.07~−0.07 rad 등
- `fk_demo.py` 작성 (`run_panda.py`는 그대로 유지)
  - home 자세를 중심으로 joint1(±0.6rad, 10초), joint2(±0.3rad, 8초), joint4(±0.4rad, 6초)를 사인파로 왕복
  - 목표 각도는 관절 한계에서 0.1rad 안쪽으로 잘라서(clip) 사용
  - `mj_kinematics`로 FK 계산 후 손끝 좌표와 관절 각도를 0.5초(시뮬레이션 시간)마다 출력
- 테스트 결과
  - 뷰어 실행: 창이 열리고 좌표가 0.5초마다 출력됨, 오류 없음
  - 화면 없이 120초 검증: 관절은 실제 한계에서 0.05rad 이상 안쪽 유지, 목표 대비 관절각 오차 최대 0.047rad, 손끝 높이 0.21~0.85m, 충돌 없음
  - MuJoCo FK 결과를 Franka 공식 DH 파라미터로 직접 계산한 값과 비교: 최대 차이 약 8e-16m (일치)
  - 자원 사용량: 메모리 약 0.86GB, CPU 약 코어 1개 (1단계와 비슷)

### 아직 하지 않은 것
- IK
- 2×2 팔레트, 상자 모델과 적재
- 적재 위치 결정 알고리즘
- 작업시간 측정

### 실행 방법
```powershell
.\.venv\Scripts\python.exe fk_demo.py
```

## 2026-10-14 — 3단계: IK(역기구학) 실습

### 오늘의 목표
- FK 데모 커밋
- `ik_demo.py`: 손끝 목표 (x, y, z) → 자코비안 + DLS로 7개 팔 관절 각도 계산 → 시뮬레이션에서 이동

### 완료한 작업
- 커밋 `063187a` "Add FK demonstration" (`fk_demo.py`, `DEVLOG.md`)
- `ik_demo.py` 작성 (`run_panda.py`, `fk_demo.py`는 그대로 유지, 모델 파일 수정 없음)
  - 손끝 정의(`get_fingertip_position`)와 안전 여유(0.1rad)를 `fk_demo.py`에서 그대로 가져와 사용
  - 손끝 자코비안: `mj_jac`에 hand 중심이 아닌 손끝 월드 좌표를 넘겨 계산
  - DLS IK: `dq = Jᵀ(JJᵀ + λ²I)⁻¹e`, λ=0.05, 1회 최대 0.2rad, 허용 오차 0.1mm, 최대 200회
  - 관절 한계: 매 반복마다 실제 한계에서 0.1rad 안쪽으로 clip
  - 실패 처리: 목표 z < 0.05m 거부 / 200회 안에 수렴 못 하면 실패 / 계산된 자세에 충돌이 있으면 실패 → 실패한 목표는 로봇을 움직이지 않음
  - IK는 별도 `MjData`(`ik_data`)에서 계산하고, 결과만 모터 목표로 전달
  - 이동: 3초 동안 smoothstep 보간 후 1초 대기, 이동 중 충돌·관절 한계 위반 스텝 수를 셈
  - 뷰어에 목표 위치를 반투명 구슬로 표시 (노랑 대기 / 초록 이동 중 / 파랑 도달 / 빨강 실패)
  - `--headless` 옵션: 화면 없이 같은 동작을 빠르게 실행
- 목표: (0.50, 0.20, 0.45), (0.45, −0.20, 0.35), (0.60, 0.00, 0.30) + 일부러 넣은 도달 불가능 목표 (1.20, 0.00, 0.50)

### 발견한 문제와 수정
- 첫 테스트에서 IK 계산 오차는 0.1mm 이하였지만, 실제 이동 후 손끝이 6~8mm 아래로 처짐
- 원인: Panda 모터는 위치 서보(힘 = kp × 각도 차이)라서 중력을 버티는 만큼 각도 차이가 남음.
  정지 상태에서 `목표각 − 실제각`이 `중력토크 / kp`와 일치함을 확인 (예: joint2 0.0066rad)
- 수정: 목표 자세의 중력 토크를 `ik_data`에서 계산해, 모터 목표에 `중력토크 / kp`를 더함 (중력 보상)
- 결과: 실제 위치 오차 6~8mm → 0.01~0.09mm

### 검증 결과
| 목표 | IK 반복 | 계산상 오차 | 이동 후 실제 오차 | 충돌 / 관절 한계 위반 |
|---|---|---|---|---|
| 1 (0.50, 0.20, 0.45) | 4회 | 0.005 mm | 0.01 mm | 0 / 0 |
| 2 (0.45, −0.20, 0.35) | 5회 | 0.086 mm | 0.09 mm | 0 / 0 |
| 3 (0.60, 0.00, 0.30) | 4회 | 0.033 mm | 0.03 mm | 0 / 0 |
| 4 (1.20, 0.00, 0.50) | 200회 (실패) | 265 mm 남음 | 이동 안 함 | — |

- 뷰어 실행과 headless 실행 결과 동일. 뷰어 화면 캡처로 목표 구슬 표시 확인
- IK 계산 전후로 시뮬레이션 `data`(qpos, ctrl, time, xpos)가 바뀌지 않음 확인
- 손끝 자코비안 vs 수치 미분: 최대 차이 8.7e-11 (일치). hand 중심 자코비안과는 최대 0.103 차이 (오프셋이 반영됨)
- 실패 처리: 바닥 아래 목표 → 사전 거부, 로봇 베이스 안쪽 → 충돌로 거부, 머리 위·뒤쪽 먼 목표 → 수렴 실패로 거부
- 무작위 목표 200개 (x 0.3~0.7, y ±0.35, z 0.15~0.7 m): 200개 성공, 반복 평균 4.5회 / 최대 7회, 모든 해가 안전 범위 안
- 자원 사용량 (뷰어): 메모리 약 0.6GB, CPU 약 1.4코어. headless 전체 실행 약 2.5초

### 한계 (아직 하지 않은 것)
- IK는 위치(x, y, z)만 맞추고 손의 방향(그리퍼가 아래를 향하는지 등)은 맞추지 않음
- 이동 경로는 관절 공간 보간이라, 손끝이 직선으로 움직이지는 않음. 경로 중 충돌은 시뮬레이션에서 세기만 하고 미리 피하지는 않음
- 2×2 팔레트, 상자 모델과 적재 / 적재 위치 결정 알고리즘 / 작업시간 측정

### 실행 방법
```powershell
.\.venv\Scripts\python.exe ik_demo.py              # 뷰어
.\.venv\Scripts\python.exe ik_demo.py --headless   # 화면 없이 테스트
```
