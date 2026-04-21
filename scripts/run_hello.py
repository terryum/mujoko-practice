"""
run_hello.py — models/hello_world.xml 을 로드하고 300 step 시뮬레이션하는 최소 예제.

목표(학습용):
    1) MJCF → MjModel / MjData 로 로드하는 흐름을 눈으로 보기.
    2) mj_step 1회당 상태(qpos, qvel, qfrc_actuator) 가 어떻게 변하는지 관찰.
    3) 접촉(data.ncon, data.contact)이 언제 생기는지 관찰.
    4) viewer가 되면 viewer로, 아니면 headless로 돌아가게.

사용 예:
    # 헤드리스 (기본) — 어느 python 인터프리터에서든 돈다.
    ~/robotics/mujoco_stack/.venv/bin/python scripts/run_hello.py

    # 인터랙티브 viewer (macOS는 반드시 mjpython 사용).
    ~/robotics/mujoco_stack/.venv/bin/mjpython scripts/run_hello.py --viewer

옵션:
    --steps N          : 시뮬레이션 step 수 (기본 300)
    --print-every N    : 몇 step마다 상태를 찍을지 (기본 50)
    --viewer           : mujoco.viewer.launch_passive 로 창을 띄움
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import mujoco


# ---------------------------------------------------------------------------
# 경로 유틸
# ---------------------------------------------------------------------------
# 스크립트가 어디서 실행되든 repo 루트 기준의 models/hello_world.xml 을 찾도록
# __file__ 을 기준으로 절대경로를 만든다.
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "hello_world.xml"


def load_model(path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """MJCF 파일을 읽어 MjModel, MjData 페어를 만든다.

    MjModel  : 파싱된 모델(질량, geom, joint, actuator 등) — 시뮬 중 불변.
    MjData   : 시뮬레이션 상태(qpos, qvel, qacc, ctrl, contact 등) — 매 step마다 변함.
    """
    if not path.exists():
        raise FileNotFoundError(f"MJCF not found: {path}")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    return model, data


# ---------------------------------------------------------------------------
# 컨트롤 입력
# ---------------------------------------------------------------------------
def set_control(model: mujoco.MjModel, data: mujoco.MjData, t: float) -> None:
    """시간 t에 따라 actuator ctrl을 준다.

    여기서는 hinge_motor에 작은 sin 파형을 입력해서:
        - qfrc_actuator (actuator가 joint에 발생시키는 일반화 힘)이 0이 아닌 값을 갖고,
        - pendulum이 제자리에 멈춰 있지 않고 흔들리게 한다.
    """
    # nu = actuator 개수. 여기서는 1. ctrl shape == (nu,).
    if model.nu == 0:
        return
    data.ctrl[0] = 0.3 * np.sin(2.0 * np.pi * 1.0 * t)  # 진폭 0.3, 주파수 1 Hz


# ---------------------------------------------------------------------------
# 상태 출력
# ---------------------------------------------------------------------------
def print_state(step: int, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """qpos, qvel, qfrc_actuator, ncon을 한 줄 요약으로 찍는다.

    numpy 배열은 set_printoptions 로 출력 폭을 줄여 한 눈에 비교하기 쉽게 한다.
    """
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    print(
        f"[step {step:>4d}  t={data.time:6.3f}] "
        f"qpos={data.qpos}  "
        f"qvel={data.qvel}  "
        f"qfrc_act={data.qfrc_actuator}  "
        f"ncon={data.ncon}"
    )


def print_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """현재 활성화된 접촉을 모두 나열한다.

    data.contact[i] 에는:
      geom1, geom2        — 접촉한 두 geom의 id
      dist                — 침투 깊이(음수면 서로 겹침)
      pos                 — 접촉점 월드 좌표
      frame               — 접촉 좌표계(R^3x3 flattened)
    가 들어있다. 이름을 보기 위해 mj_id2name 을 쓴다.
    """
    print(f"  contacts: {data.ncon}")
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}"
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}"
        print(f"    #{i}: {g1} <-> {g2}  dist={c.dist:+.5f}  pos={np.array(c.pos)}")


# ---------------------------------------------------------------------------
# 시뮬 루프 (headless)
# ---------------------------------------------------------------------------
def run_headless(model: mujoco.MjModel, data: mujoco.MjData,
                 steps: int, print_every: int) -> None:
    """창 없이 steps 만큼 mj_step을 돌린다."""
    print(f"[headless] stepping {steps} times, dt={model.opt.timestep}s "
          f"→ sim duration ≈ {steps * model.opt.timestep:.3f}s")
    for i in range(1, steps + 1):
        set_control(model, data, data.time)
        mujoco.mj_step(model, data)
        if i % print_every == 0 or i == 1:
            print_state(i, model, data)

    print("\n[final state]")
    print_state(steps, model, data)
    print_contacts(model, data)


# ---------------------------------------------------------------------------
# 시뮬 루프 (viewer)
# ---------------------------------------------------------------------------
def run_viewer(model: mujoco.MjModel, data: mujoco.MjData,
               steps: int, print_every: int) -> None:
    """passive viewer로 창을 띄우고 사용자가 닫거나 steps 끝날 때까지 스텝.

    macOS에서는 viewer의 메인스레드 요구 때문에 반드시 `mjpython`으로 실행해야 한다.
    그렇지 않으면 mujoco 측에서 에러/경고가 난다.

    감지 방법: `mjpython` 런처는 실행 시 `MJPYTHON_BIN` 환경변수를 세팅한다.
    (참고로 sys.executable 은 mjpython 아래서도 실제 python 바이너리를 가리키므로
     그걸로 체크하면 오판한다.)
    """
    if sys.platform == "darwin" and "MJPYTHON_BIN" not in os.environ:
        print("[viewer] macOS에서는 `mjpython`로 실행해야 한다. "
              "다음처럼 다시 시도: "
              "`~/robotics/mujoco_stack/.venv/bin/mjpython scripts/run_hello.py --viewer`")
        print("[viewer] 일단 headless로 fallback.")
        run_headless(model, data, steps, print_every)
        return

    import mujoco.viewer  # 지연 import: headless 모드에서는 안 건드려도 되게.

    print(f"[viewer] launching passive viewer; will step up to {steps} times.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        for i in range(1, steps + 1):
            if not viewer.is_running():
                print(f"[viewer] window closed at step {i}.")
                break
            set_control(model, data, data.time)
            mujoco.mj_step(model, data)
            viewer.sync()
            if i % print_every == 0 or i == 1:
                print_state(i, model, data)
            # 실시간 재생 비슷하게 맞추기 위한 단순 슬립. 학습 목적이라 생략해도 됨.
            elapsed = time.time() - start
            target = i * model.opt.timestep
            if target > elapsed:
                time.sleep(target - elapsed)

        print("\n[final state]")
        print_state(steps, model, data)
        print_contacts(model, data)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="Load hello_world.xml and step MuJoCo.")
    p.add_argument("--steps", type=int, default=300, help="number of mj_step calls")
    p.add_argument("--print-every", type=int, default=50, help="print state every N steps")
    p.add_argument("--viewer", action="store_true", help="open interactive passive viewer")
    args = p.parse_args()

    print(f"mujoco version : {mujoco.__version__}")
    print(f"model path     : {MODEL_PATH}")

    model, data = load_model(MODEL_PATH)

    # 모델 메타정보: 이후 실험에서 숫자가 어떻게 바뀌는지 비교할 기준.
    print(f"nq (qpos dim)  : {model.nq}")
    print(f"nv (qvel dim)  : {model.nv}")
    print(f"nu (actuators) : {model.nu}")
    print(f"ngeom          : {model.ngeom}")
    print(f"timestep       : {model.opt.timestep}")
    print()

    if args.viewer:
        run_viewer(model, data, args.steps, args.print_every)
    else:
        run_headless(model, data, args.steps, args.print_every)
    return 0


if __name__ == "__main__":
    sys.exit(main())
