# mujoko-practice

MuJoCo를 first-principles로 익히기 위한 최소 워크스페이스.

학습은 [`LEARNING_PLAN.md`](./LEARNING_PLAN.md) 커리큘럼에 따라 진행한다. 챕터별 산출물은 `chapters/s<stage>_<##>_<slug>/` 아래. 세션 시작 시 `"나 어디까지 했어? 다음 진행"` 이라고 하면 다음 챕터의 프롬프트가 제공됨.

## 구성

```
models/hello_world.xml   # 바닥 + hinge pendulum + free ball (MJCF)
scripts/run_hello.py     # 모델 로드 → 300 step → qpos/qvel/actuator/contact 출력 (headless/viewer)
scripts/render_hello.py  # 오프스크린 렌더 → logs/hello_world.mp4 로 영상 저장
```

## Python 환경

이 폴더는 전용 venv를 만들지 않고, 이미 구축된 `~/robotics/mujoco_stack/.venv` (Python 3.12, mujoco 3.7.0)를 재사용한다.

편의상 아래 2개를 기억해두면 됨:

```bash
PYBIN=~/robotics/mujoco_stack/.venv/bin/python
MJPY=~/robotics/mujoco_stack/.venv/bin/mjpython   # macOS에서 viewer 필요할 때
```

## 실행

Headless (어디서든 동작):

```bash
$PYBIN scripts/run_hello.py
$PYBIN scripts/run_hello.py --steps 600 --print-every 100
```

인터랙티브 viewer (macOS는 반드시 `mjpython`):

```bash
$MJPY scripts/run_hello.py --viewer
```

3D를 영상으로만 확인하고 싶을 때 (mp4 저장):

```bash
$PYBIN scripts/render_hello.py                          # logs/hello_world.mp4 기본 출력
$PYBIN scripts/render_hello.py --steps 1200 --fps 60    # 더 긴 영상
$PYBIN scripts/render_hello.py --out logs/slow.mp4 --render-every 1 --fps 30  # 슬로우
open logs/hello_world.mp4                                # macOS 기본 플레이어
```

출력 예(headless):

```
nq=8, nv=7, nu=1, ngeom=3, timestep=0.002
[step   1 t=0.002] qpos=[...]  qvel=[...]  qfrc_act=[...]  ncon=0
...
[step 250 t=0.500] ncon=1      # ball이 floor에 닿는 시점
contacts: 1
  #0: floor <-> ball_geom  dist=-0.00119  pos=[0.3 0. -0.0006]
```

## 실험 포인트

`models/hello_world.xml` 하나만 바꿔가며 관찰하기 좋게 단순하게 유지했다.

- **mass** — `<geom density="...">` 추가 또는 `<inertial mass="...">` 직접 지정
- **damping** — `<joint damping="...">` 값 조정 (기본 0.05)
- **friction** — `<default><geom friction="slide spin roll"/></default>` 혹은 per-geom
- **actuator** — `<motor gear=...>`, `<position kp=...>`, `<velocity kv=...>` 로 교체
- **geom size / pos** — ball 출발 위치, rod 길이 등

바꾼 뒤 script 재실행 → `qpos/qvel/qfrc_act/ncon` 수치가 어떻게 달라지는지 비교.
