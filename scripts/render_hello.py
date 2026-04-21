"""
render_hello.py — hello_world.xml 을 스텝하면서 오프스크린으로 프레임을 렌더하고
mp4로 저장한다. GUI 창을 띄울 수 없는 환경에서도 "3D가 어떻게 움직이는지"를 눈으로
확인하기 위한 스크립트.

핵심 개념:
    - mujoco.Renderer(model, height, width) : 오프스크린 GL 컨텍스트를 만든다.
      macOS에서는 Cocoa 기반이라 mjpython 없이도 동작한다(뷰어와 달리 메인스레드 제약 없음).
    - renderer.update_scene(data, camera=...) : 현재 MjData의 상태로 scene을 갱신.
    - renderer.render() : numpy (H, W, 3) uint8 RGB 배열을 돌려줌.
    - mediapy.write_video(path, frames, fps=...) : 프레임 리스트를 mp4로 인코딩.

사용 예:
    ~/robotics/mujoco_stack/.venv/bin/python scripts/render_hello.py
    ~/robotics/mujoco_stack/.venv/bin/python scripts/render_hello.py \
        --steps 600 --render-every 2 --fps 60 --out logs/hello_world.mp4

출력:
    기본적으로 logs/hello_world.mp4 에 영상 저장.
    스텝 수 / render-every / timestep 조합에 따라 영상 길이가 결정된다.
      video_duration_s ≈ (steps / render_every) / fps
      sim_duration_s   ≈ steps * model.opt.timestep
    두 값을 맞추면 "실시간 속도" 영상, 달리하면 슬로우/패스트 모션.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco
import mediapy


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "hello_world.xml"
DEFAULT_OUT = REPO_ROOT / "logs" / "hello_world.mp4"


def set_control(model: mujoco.MjModel, data: mujoco.MjData, t: float) -> None:
    """run_hello.py와 동일한 sin 입력. 영상에서 pendulum이 흔들리는 게 보이도록."""
    if model.nu == 0:
        return
    data.ctrl[0] = 0.3 * np.sin(2.0 * np.pi * 1.0 * t)


def main() -> int:
    p = argparse.ArgumentParser(description="Offscreen-render hello_world.xml to mp4.")
    p.add_argument("--steps", type=int, default=600, help="total mj_step count")
    p.add_argument("--render-every", type=int, default=4,
                   help="render a frame every N sim steps (keeps file size manageable)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=60, help="playback fps for the mp4")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output mp4 path")
    p.add_argument("--camera", default="-1",
                   help="camera id or name; '-1' = free camera (default MuJoCo view)")
    args = p.parse_args()

    print(f"mujoco version : {mujoco.__version__}")
    print(f"model path     : {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # 카메라 인자: 숫자면 int로(-1은 free camera), 이름 문자열이면 그대로 넘김.
    try:
        camera: int | str = int(args.camera)
    except ValueError:
        camera = args.camera

    # Renderer는 컨텍스트 매니저로 쓰면 리소스 정리를 알아서 해줌.
    # 생성 시점에 GL 컨텍스트가 만들어지므로 약간의 지연이 있다.
    print(f"creating Renderer {args.width}x{args.height} ...")
    frames: list[np.ndarray] = []
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        # 시뮬 + 렌더 루프
        for i in range(1, args.steps + 1):
            set_control(model, data, data.time)
            mujoco.mj_step(model, data)

            # 매 스텝마다 렌더하면 너무 느리고 파일도 커지므로 주기 조절.
            if i % args.render_every == 0:
                renderer.update_scene(data, camera=camera)
                frame = renderer.render()  # (H, W, 3) uint8
                frames.append(frame)

    # 저장 디렉터리 보장.
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # 영상/시뮬 시간 안내.
    sim_dur = args.steps * model.opt.timestep
    vid_dur = len(frames) / args.fps if frames else 0.0
    speed = vid_dur / sim_dur if sim_dur > 0 else 0.0
    print(f"captured frames : {len(frames)}")
    print(f"sim duration    : {sim_dur:.3f}s")
    print(f"video duration  : {vid_dur:.3f}s  ({args.fps} fps)")
    print(f"playback speed  : x{1.0/speed:.2f} real-time" if speed > 0 else "(empty)")

    if not frames:
        print("no frames captured; increase --steps or decrease --render-every.")
        return 1

    mediapy.write_video(str(args.out), frames, fps=args.fps)
    print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
