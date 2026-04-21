# Project purpose

This repository is for learning MuJoCo from first principles.

Current phase:
- Learn model loading, stepping, state inspection, kinematics/dynamics basics, and contact observation.
- Prefer tiny self-contained MJCF examples.
- Keep code minimal and heavily commented.
- Do not introduce RL frameworks, dm_control, MJX, or large external repos unless explicitly requested.

Working style:
- Make one small runnable example at a time.
- After each example, suggest 3 focused modifications:
  1. one modeling change,
  2. one physics parameter change,
  3. one observation/logging change.
- Always explain what to observe before and after the change.

## Learning curriculum

학습 진행은 `LEARNING_PLAN.md` (동일 내용: `~/.claude/plans/majestic-hatching-starlight.md`) 참고.

사용자가 "나 어디까지 했어? 다음 진행해줘" 류의 발화를 하면:
1. `LEARNING_PLAN.md` 읽기.
2. `chapters/` 아래 각 폴더 스캔 — `done.md` 존재 여부로 완료/미완료 판별.
3. 현재 위치 3줄 요약 + 다음 챕터의 "복사-붙여넣기 프롬프트"를 코드펜스로 제공.

챕터 산출물 규약: `chapters/s<stage>_<##>_<slug>/{prompt.md, notes.md, *.xml, *.py, done.md}`.

공용 venv 인터프리터:
- `PYBIN=~/robotics/mujoco_stack/.venv/bin/python`
- `MJPY=~/robotics/mujoco_stack/.venv/bin/mjpython` (macOS viewer 필수)
