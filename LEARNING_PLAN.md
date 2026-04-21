# MuJoCo 학습 커리큘럼 플랜

## Context

사용자는 MuJoCo를 first-principles로 익히려고 한다. 오늘(2026-04-21) hello_world 수준의 워크스페이스 셋업을 막 끝냈고 (damping·탄성 스윕까지 체험), 이제 며칠~몇 주에 걸쳐 체계적으로 진도를 빼려고 한다.

운영 방식이 일반적인 "한 번에 구현하는 플랜"과 다르다 — 이 플랜은 **며칠 동안 매일 돌아와서 다음 챕터를 실행할 때 참고하는 커리큘럼 문서**다. 사용자는 매 세션 시작 시 `"나 어디까지 했어? 다음 진행"` 식으로 호출하고, Claude Code는 이 파일 + `chapters/` 디렉토리 상태를 보고 다음 챕터의 **복사-붙여넣기용 프롬프트**를 전달한다. 사용자는 그걸 다음 메시지에 붙여넣어 실제 작업을 시작한다.

이 문서의 목적: (1) Stage 1~5의 큰 흐름을 고정, (2) Stage 1 챕터들은 지금 당장 붙여넣을 수 있는 수준으로 상세화, (3) Stage 2~5는 주제·산출물·참고자료만 스켈레톤으로 남겨두고 진입 시 상세화한다.

---

## 사전 상태 (이미 설치/준비된 것 vs 챕터 진입 시 확인할 것)

| 영역 | 상태 | 비고 |
|---|---|---|
| Python 3.11 + `mujoco 3.7.0` | ✅ `~/robotics/mujoco_stack/.venv` | 이 폴더 전용 venv 없음. 공유 사용. |
| `mjpython` (macOS viewer 메인스레드) | ✅ `~/robotics/mujoco_stack/.venv/bin/mjpython` | Stage 1 전 챕터에서 계속 사용. |
| `mujoco.viewer`, `mujoco.Renderer`, `mediapy` | ✅ | 스크립트 `scripts/run_hello.py`, `scripts/render_hello.py`가 레퍼런스. |
| `dm_control`, `mink`, `pin`, `casadi`, `osqp`, `qpsolvers[daqp]`, `robot_descriptions`, `jupyterlab`, `rerun-sdk` | ✅ | Stage 2~3에서 유용. |
| `jax` (CPU), `mujoco-mjx 3.7.0`, `brax`, `flax`, `optax`, `mujoco-warp`, `mjlab`, `playground`(editable) | ✅ import 수준 | Stage 4 실학습은 NVIDIA 필요. 개념 확인·디버그만 Mac에서. |
| `mujoco_menagerie` (하드웨어 모델) | ✅ `~/robotics/mujoco_stack/external/mujoco_menagerie/` | Stage 2에서 참조. |
| `mujoco_mpc` native build + `mjpc` CLI | ✅ `~/robotics/mujoco_stack/external/mujoco_mpc/build/bin/mjpc` | Stage 3에서 바로 실행 가능. |
| `stable-baselines3`, `imitation` | ❌ | Stage 4.0 설치 챕터에서 설치. |
| `LeRobot`, `OpenVLA` | ❌ | Stage 5.0 설치 챕터에서 판단 후 설치. |
| `sjchoi86/yet-another-mujoco-tutorial-v3` | ❌ | 챕터 1.8 진입 시 clone. |

환경변수 요약: `PYBIN=~/robotics/mujoco_stack/.venv/bin/python`, `MJPY=~/robotics/mujoco_stack/.venv/bin/mjpython`. 각 프롬프트에서 이걸 그대로 씀.

---

## 플랜 사용법 (매일 복귀 흐름)

**Step 1.** 세션 열고 한 줄 날림:
> 나 어디까지 했어? 다음 챕터 진행해줘.

**Step 2.** Claude Code는 아래를 순서대로 실행:
1. 이 플랜 파일을 읽음 (경로: `~/.claude/plans/majestic-hatching-starlight.md`, 그리고 `mujoko-practice/LEARNING_PLAN.md`의 사본).
2. `~/Codes/mujoko-practice/chapters/` 스캔 → 존재하는 폴더 목록과 기대 산출물 체크.
3. 완료 여부: 해당 챕터 폴더 안에 `done.md`가 있으면 완료로 간주(없으면 진행 중/미착수).
4. "지금 상태: Stage X, Chapter Y.Z까지 완료. 다음은 Y.(Z+1)입니다." 라고 요약.
5. 다음 챕터의 **복사-붙여넣기 프롬프트**를 코드 블록으로 제시.

**Step 3.** 사용자는 그 프롬프트를 새 메시지에 붙여넣어 실제 실습 진행.

**Step 4.** 챕터 끝나면 Claude Code가 해당 폴더에 `done.md`(요약 + 무엇을 배웠는지 3줄)를 남김.

---

## 파일 규약

모든 챕터 산출물은 `~/Codes/mujoko-practice/chapters/` 아래 폴더 하나로 묶는다.

```
chapters/
  s1_01_mjcf_anatomy/
    prompt.md       # 실제 붙여넣은 프롬프트 원문 (재현용)
    notes.md        # 학습 노트
    annotated.xml   # 챕터에 따라
    demo.py
    output.mp4
    done.md         # 완료 시그널: 한 문단 요약 + 핵심 배움 3줄
  s1_02_state_inspection/
    ...
```

네이밍: `s<stage>_<##>_<slug>/`. 알파벳/숫자 정렬이 곧 학습 순서.

---

## 코드 생성 정책 (내가 만드나 vs 다운로드하나)

| 유형 | 정책 |
|---|---|
| Stage 1 전반 | Claude Code가 프롬프트 받아 그 자리에서 MJCF + Python 생성. 다운로드 없음. |
| Stage 2 접촉 실험 | Claude Code 생성. 단, Menagerie 손 모델은 **이미 디스크에 있으므로** `external/mujoco_menagerie/...` 경로를 직접 로드. 복사본 안 만듦. |
| Stage 2~3 sjchoi86 tutorial | 챕터 1.8에서 `git clone` 한 번. 이후 로컬 폴더 참조. |
| Stage 3 MPC | 이미 빌드된 `mjpc` CLI 실행. Claude Code는 task 정의 파일/Python wrapper만 생성. |
| Stage 4 RL | `pip install stable-baselines3 imitation`. 훈련 스크립트는 Claude Code 생성. 대형 베이스라인은 SB3 내장 사용. |
| Stage 5 VLA | `git clone lerobot` / `git clone openvla`. 각 repo의 공식 튜토리얼 따라감. Claude Code는 glue 코드만. |

원칙: **가능하면 Claude Code가 그 자리에서 최소 코드를 생성**. 대형 공식 튜토리얼은 그 repo 구조를 존중해서 복사 없이 참조.

---

## Stage 1 — MuJoCo 몸에 익히기 (상세)

목표: 모델을 읽고, 수정하고, 물리가 어떻게 바뀌는지 체감. RL 미도입. `hello_world.xml` 범주에서 충분히 놀기.

### s1_01 — MJCF 해부

**목표**: `models/hello_world.xml`의 모든 태그 의미/속성/변경 시 체감 변화를 정리.

**복사-붙여넣기 프롬프트**:

```
MuJoCo 커리큘럼 Stage 1, Chapter s1_01 (MJCF 해부) 진행해줘.

1. chapters/s1_01_mjcf_anatomy/ 폴더 생성.
2. models/hello_world.xml 의 모든 태그(<mujoco>, <option>, <visual>, <default>, <worldbody>, <body>, <joint>, <geom>, <actuator>, <motor>)에 대해
   "이 태그의 역할 / 주요 속성 3개 / 대표 속성을 바꾸면 어떤 물리적 변화가 오는지"
   3줄짜리 블록 주석을 단 annotated.xml 을 해당 폴더에 저장.
3. notes.md 에 표(태그 | 속성 | 기본값 | 변경 시 관찰 포인트) 작성.
4. annotated.xml 이 실제 로드되는지 검증:
   `~/robotics/mujoco_stack/.venv/bin/python -c "import mujoco; m=mujoco.MjModel.from_xml_path('chapters/s1_01_mjcf_anatomy/annotated.xml'); print(m.nq, m.nv, m.nu)"`
5. done.md 에 한 문단 요약 + 핵심 배움 3줄.
```

**참고**: MuJoCo XML Reference https://mujoco.readthedocs.io/en/stable/XMLreference.html

---

### s1_02 — 상태 벡터 해부 (qpos / qvel / nq vs nv)

**목표**: free joint의 7/6 비대칭, quaternion 해석, joint 타입별 크기 이해.

**프롬프트**:

```
MuJoCo 커리큘럼 Chapter s1_02 (상태 벡터 해부) 진행.

1. chapters/s1_02_state_inspection/ 생성.
2. model.xml: worldbody에 아래 4개 body를 각각 다른 joint로 달아라.
   - free joint (box)
   - hinge joint (rod)
   - slide joint (puck, z축)
   - ball joint (small sphere)
3. script.py: 모델 로드 후 mj_step 1번만 돌리고
   print: nq, nv, qpos.shape, qvel.shape, 각 joint의 qposadr/qveladr/type 이름 매핑.
   그리고 free joint qpos[3:7]을 quat으로 해석해 mujoco.mju_quat2Mat 으로 회전행렬 구해 출력.
4. notes.md 에 "왜 nq != nv?" 1문단.
5. 실행: `$PYBIN chapters/s1_02_state_inspection/script.py`
6. done.md.
```

**관찰**: quaternion은 4-벡터지만 각속도는 3-벡터라 `nq - nv = 1` per free joint, ball joint도 동일.

**참고**: https://mujoco.readthedocs.io/en/stable/computation/index.html#coordinates

---

### s1_03 — mass & inertia (density vs explicit <inertial>)

**목표**: mass 자동 계산 vs 직접 지정의 차이, pendulum 주기에 미치는 영향.

**프롬프트**:

```
Chapter s1_03 (mass & inertia).

1. chapters/s1_03_mass/ 생성.
2. 같은 pendulum을 3가지 버전으로: (a) default density (mujoco 기본 1000), (b) density=500 지정, (c) <inertial> 직접 지정(mass=2kg, diaginertia 임의값).
3. 각 모델의 자유진동(초기 각도 30도, actuator off, damping 0.05) 1.5초 시뮬 → hinge qpos 시계열 저장.
4. mediapy 대신 matplotlib로 3개 시계열 한 plot에 그려 plot.png 저장.
5. notes.md: 주기가 왜 다른지 (T = 2π√(I_eff / mgl)) 설명 + 실측 주기 3개 비교.
6. done.md.
```

**참고**: https://mujoco.readthedocs.io/en/stable/modeling.html#inertia

---

### s1_04 — damping / stiffness / armature 스윕

**목표**: joint의 damping, springref, springdamper, armature가 각각 어떤 항에 들어가는지.

**프롬프트**:

```
Chapter s1_04 (damping · stiffness · armature 스윕).

1. chapters/s1_04_joint_params/ 생성.
2. pendulum 모델 하나 기준으로 4가지 파라미터를 각각 2~3개 값으로 스윕:
   damping ∈ {0.05, 0.5, 2.0}, stiffness ∈ {0, 5, 20}, armature ∈ {0, 0.01, 0.1}.
3. 각 조합에서 초기 각도 45도, 2초 자유진동 → 최대 진폭, 정착 시간, 주파수 피크 기록.
4. sweep_results.csv 로 저장, summary.md 에 관찰 정리.
5. 흥미로운 2~3 조합을 render_hello.py 스타일로 mp4 저장.
6. done.md.
```

**참고**: XML reference `<joint>` + https://mujoco.readthedocs.io/en/stable/computation/index.html#equations-of-motion

---

### s1_05 — 액추에이터 4종 비교 (motor / position / velocity / general)

**목표**: 같은 목표 각도를 달성할 때 actuator 타입별 ctrl 의미와 실제 joint torque 차이.

**프롬프트**:

```
Chapter s1_05 (actuator 비교).

1. chapters/s1_05_actuators/ 생성.
2. 단일 hinge pendulum 모델에 4가지 actuator를 각각 적용한 4개 XML:
   - motor(gear=1), position(kp=10, kv=1), velocity(kv=5), general(gainprm 직접 지정으로 position과 동치 구성).
3. 목표: hinge 각도를 30도로 유지. 각 버전에서 ctrl 값 의미를 주석으로 설명.
4. script.py: 4개 모델 순차 시뮬 2초 → 각 step의 data.ctrl[0], data.qfrc_actuator[0], data.qpos[0] 로그.
   그래프 하나에 4개 qpos(t) overlay, 다른 그래프에 qfrc_actuator(t) overlay 저장.
5. notes.md: 각 actuator의 내부 수식과 차이점 (motor force = gear·ctrl; position force = kp(ctrl - qpos) - kv·qvel; 등).
6. done.md.
```

**참고**: https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator

---

### s1_06 — camera / site / sensor 붙이기

**목표**: named camera로 카메라 워크 만들기, site는 "이름 있는 점", sensor로 data.sensordata 읽기.

**프롬프트**:

```
Chapter s1_06 (camera, site, sensor).

1. chapters/s1_06_camera_sensor/ 생성.
2. hello_world 확장: 2개의 named camera(side view, top view), ball 중심에 site 하나, pendulum 끝에 site 하나.
3. sensor 블록에 아래 추가:
   - framepos(site=ball 중심)
   - framexaxis(site=pendulum 끝)
   - jointpos(joint=hinge)
   - touch(geom=ball_geom) — 이건 site 기반이라 geom이 붙은 site 필요함. 구조 맞춰 추가.
4. script.py: 1초 시뮬. 매 스텝 data.sensordata를 tag+이름과 함께 출력.
5. render_two_cams.py: 두 카메라 각각에서 mp4 렌더 후 side.mp4 / top.mp4 저장.
6. done.md.
```

**참고**: https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor

---

### s1_07 — FK / IK / ID 수치 검증

**목표**: `mj_forward`, `mj_jac`, `mj_inverse`를 호출해 수식과 맞는지 직접 확인.

**프롬프트**:

```
Chapter s1_07 (FK · IK · ID 수치 검증).

1. chapters/s1_07_kinematics/ 생성.
2. 2-link 평면 로봇 MJCF(armature, motors 포함)를 models 대신 이 챕터 폴더 안에 만든다.
3. fk_check.py: 특정 qpos 집합에 대해 mj_forward 호출 후
   data.xpos[end_effector_body]를 직접 sin/cos 수식으로 계산한 결과와 비교. assert.
4. jac_check.py: mj_jac으로 end-effector Jacobian을 뽑아 finite-difference (qpos ± 1e-5)로 계산한 근사 Jacobian과 비교. relative error 출력.
5. id_check.py: 임의의 qpos/qvel/qacc를 세팅하고 mj_inverse로 qfrc를 얻은 뒤, 그 qfrc를 ctrl에 넣고 mj_forward한 qacc와 원래 qacc가 일치하는지 확인.
6. notes.md: FK/IK/ID의 MuJoCo 호출 순서 정리. Modern Robotics Ch 6/8 참고.
7. done.md.
```

**참고**: https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html

---

### s1_08 — sjchoi86 튜토리얼 추적

**목표**: 외부 커리큘럼 한 바퀴. 이 튜토리얼의 핵심 3개 노트북(혹은 스크립트)을 실행하고 notes로 정리.

**프롬프트**:

```
Chapter s1_08 (sjchoi86 tutorial).

1. chapters/s1_08_sjchoi86/ 생성.
2. 다음 repo를 clone: https://github.com/sjchoi86/yet-another-mujoco-tutorial-v3
   → ~/robotics/external/yet-another-mujoco-tutorial-v3 에 (이미 있으면 건너뜀).
3. 그 repo의 README에서 가장 기초 노트북 3개 식별. Jupyter kernel "MuJoCo Stack (Py 3.11)"로 실행.
4. 각 노트북당 notes.md 에 3줄: 주제 / 사용 API / 놓치면 안 될 트릭.
5. done.md.
```

**참고**: https://github.com/sjchoi86/yet-another-mujoco-tutorial-v3

---

## Stage 2 — 접촉과 closed-chain (스켈레톤)

진입 시 각 챕터를 s1_01과 같은 수준으로 상세화한다. 여기서는 주제·산출물·참고만.

| ID | 주제 | 주요 산출물 | 주 참고 |
|---|---|---|---|
| s2_01 | 접촉 anatomy 심화 (data.contact, efc_*, contact frame) | `contact_log.py`, `notes.md` | MuJoCo Computation chapter (Constraint model) |
| s2_02 | friction / solref / solimp / iter / timestep 스윕 | `sweep.py`, CSV, 2~3 mp4 | Computation chapter |
| s2_03 | equality constraint (connect, weld, joint) → pseudo closed-chain | `four_bar.xml`, `verify.py` | XML reference `<equality>` |
| s2_04 | Menagerie 손 모델 tour (Shadow 또는 Allegro) | `hand_tour.py`, mp4 | `external/mujoco_menagerie/shadow_hand/` |
| s2_05 | grasp stability 비교 (fingertip 형상·마찰·solimp) | `grasp_exp.py`, 결과 표 | MIT Robotic Manipulation, Modern Robotics Ch 12 |

외부 참고(노트/비디오 병행):
- MuJoCo Computation chapter: https://mujoco.readthedocs.io/en/stable/computation/index.html
- MIT Robotic Manipulation: https://manipulation.csail.mit.edu/
- Modern Robotics Ch 12: https://hades.mech.northwestern.edu/index.php/Modern_Robotics

Stage 2 진입 트리거: Stage 1 모든 챕터의 `done.md` 존재.

---

## Stage 3 — Planning / Control (스켈레톤)

| ID | 주제 | 주요 산출물 | 주 참고 |
|---|---|---|---|
| s3_01 | PID로 pendulum 각도 유지 + 외란 거부 | `pid.py`, plot | 기본 제어 교재 |
| s3_02 | 선형화 + LQR (scipy.linalg.solve_continuous_are) | `lqr.py`, plot | Underactuated Robotics Ch LQR |
| s3_03 | MJPC CLI 데모: `mjpc` 실행, 기본 task에서 predictive sampling/iLQG 비교 | `mjpc_cli_notes.md` | MJPC README |
| s3_04 | MJPC 커스텀 task 정의 → `external/mujoco_mpc/mjpc/tasks/` 스타일로 | 새 task folder, 빌드 확인 | MJPC source `mjpc/tasks/*` |
| s3_05 | contact-rich task에서 cost term 변경 → behavior 차이 | 실험 로그 + 영상 | MIT Underactuated contact chapter |

외부 참고:
- MuJoCo MPC: https://github.com/google-deepmind/mujoco_mpc
- MIT Underactuated Robotics: https://underactuated.csail.mit.edu/

Stage 3 진입 트리거: Stage 2 완료. MJPC는 이미 빌드되어 있음 — 설치 필요 없음.

---

## Stage 4 — RL / Imitation Learning (설치 필요, 스켈레톤)

| ID | 주제 | 설치/사전조건 | 메모 |
|---|---|---|---|
| s4_00 | 설치: SB3, imitation, (선택) sb3-contrib | `pip install "stable-baselines3[extra]" imitation sb3-contrib` | 위치: `mujoco_stack/.venv`에 추가 (requirements-optional.txt에 기록). |
| s4_01 | dm_control으로 pendulum/cartpole env 직접 작성 | dm_control 이미 있음 | env observation/reward/termination 설계 연습 |
| s4_02 | SB3 PPO 짧은 학습 (10만 step 수준) | s4_00 | Mac CPU에서도 돌 정도로 가볍게 |
| s4_03 | Behavior Cloning (demonstration 수집 → BC) | imitation lib | handcrafted demo로 시작 |
| s4_04 | SAC 비교: PPO와 같은 env에서 학습 곡선 비교 | s4_00 | 수치 비교가 목적 |
| s4_05 | Offline RL 감 잡기 (d3rlpy 또는 imitation 사용) | 필요 시 추가 설치 | 데이터 고정 후 policy 개선 |
| s4_06 | Model-based RL 개념 (PETS/Dreamer 류 설명만, 코드 실습 선택) | — | Mac에서 실학습은 비추 |

외부 참고:
- Stanford CS224R: https://cs224r.stanford.edu/
- Berkeley CS285: https://rail.eecs.berkeley.edu/deeprlcourse/
- dm_control: https://github.com/google-deepmind/dm_control
- MuJoCo Playground (대규모 학습은 GPU 필요): https://github.com/google-deepmind/mujoco_playground

Stage 4 진입 전 체크: "지금까지 이해한 접촉/동역학이 환경 설계에 반영 가능한지" 자가 질문. Stage 3 완료 이전엔 진입 비추.

---

## Stage 5 — VLA (설치 필요, 스켈레톤)

| ID | 주제 | 설치/사전조건 | 메모 |
|---|---|---|---|
| s5_00 | LeRobot / OpenVLA 개요 + 진입 판단 | 사용자 의도 확인 | 순서: LeRobot 먼저, OpenVLA 나중 |
| s5_01 | LeRobot quickstart — local sim env + policy 로드 | `git clone https://github.com/huggingface/lerobot` + 그 repo의 install 따름 | `pip install -e .` in lerobot 폴더 |
| s5_02 | OpenVLA inference demo (사전학습 모델 + LoRA) | `git clone https://github.com/openvla/openvla` + weights 다운로드 | full FT는 GPU 필요 — inference 중심 |

외부 참고:
- LeRobot: https://github.com/huggingface/lerobot
- OpenVLA: https://github.com/openvla/openvla
- NVIDIA Physical AI Learning: https://developer.nvidia.com/learn/physical-ai

Stage 5 진입 트리거: Stage 4에서 BC/SAC를 실제로 돌려봤고, VLA의 "왜"에 대한 동기가 명확해졌을 때.

---

## "나 어디까지 했어?" — 복귀 처리 레시피 (Claude Code 참고용)

사용자가 `"나 어디까지 했어? 다음 진행"` 유사 발화를 하면:

1. `ls ~/Codes/mujoko-practice/chapters/` → 존재하는 챕터 폴더 나열.
2. 각 폴더에 `done.md`가 있는지 체크 → 완료/진행중/미착수 구분.
3. 이 플랜 파일의 Stage/Chapter 표와 대조 → 다음 챕터 식별.
4. "지금 상태: Stage X, s<a>_<b>까지 완료. 진행 중인 챕터: (있으면). 다음은 s<a>_<c>(<제목>)입니다." 3줄 요약.
5. 다음 챕터의 프롬프트를 `**복사-붙여넣기**:` 헤더 아래 코드펜스로 전달.
6. 사용자는 그 프롬프트를 새 메시지에 붙여넣음.

Stage 전환 구간 (s1_08 → s2_01 등)에서는 Stage 2 챕터를 이 플랜에서 상세화(프롬프트 풀어쓰기)한 뒤 제시. 이 플랜 자체를 업데이트해서 기록을 남긴다.

---

## 승인 후 즉시 할 후속 액션 (Plan Mode 해제 후)

플랜 승인되면 다음을 실행:

1. `chapters/` 디렉토리 생성: `mkdir -p ~/Codes/mujoko-practice/chapters`.
2. 이 플랜 파일을 repo에도 복사: `cp ~/.claude/plans/majestic-hatching-starlight.md ~/Codes/mujoko-practice/LEARNING_PLAN.md`. (미래 세션이 repo 안에서 찾기 쉽게)
3. CLAUDE.md 하단에 포인터 한 줄 추가: `학습 커리큘럼은 LEARNING_PLAN.md 참고. "나 어디까지 했어?" 발화 시 chapters/ 스캔.`
4. 메모리에 학습 커리큘럼 존재를 기록 (project memory) — 미래 세션이 첫 턴부터 맥락을 갖도록.
5. 첫 챕터(s1_01) 실행을 위한 복사-붙여넣기 프롬프트를 사용자에게 출력 — "복귀 흐름" 1회 리허설.

---

## 검증 (플랜이 제대로 작동하는지)

- **End-to-end 리허설**: 승인 직후 s1_01 프롬프트를 사용자가 복사-붙여넣기 한 번 해봐서 Claude Code가 그 프롬프트만으로 자립적으로 작업을 완료할 수 있는지 확인 (의존 맥락이 전부 프롬프트 안에 있는지).
- **복귀 시뮬레이션**: s1_01 완료 후 사용자가 새 세션에서 `"나 어디까지 했어?"` 라고 했을 때 Claude Code가 s1_02 프롬프트를 정확히 뱉는지.
- **Stage 경계**: s1_08 완료 시점에 이 플랜 파일이 업데이트되어 Stage 2 챕터들이 상세화되는지.

문제 생기면 이 플랜 파일을 수정해서 챕터 순서/내용 조정. 플랜은 살아 있는 문서다.
