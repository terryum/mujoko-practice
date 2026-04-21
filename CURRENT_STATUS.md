# Current Status — MuJoCo 학습 진행판

> 이 파일은 "나 어디까지 했어? 남은 공부가 뭐야?" 질문에 즉답하기 위한 **동적 진행판**이다.
> 커리큘럼 상세는 [`LEARNING_PLAN.md`](./LEARNING_PLAN.md).
> Claude Code가 각 챕터 시작/완료 시점에 이 파일의 "진행 현황" 섹션을 업데이트한다.

**Last updated**: 2026-04-21 (워크스페이스 셋업 완료, 학습 착수 전)

---

## 한눈 요약

- **현재 Stage**: 1 (MuJoCo 몸에 익히기)
- **착수한 챕터**: 없음
- **다음 챕터**: **s1_01 — MJCF 해부**
- **완료율**: 0 / 21 챕터 (스켈레톤 포함 대략)

---

## 진행 현황

### Stage 1 — MuJoCo 몸에 익히기
| 챕터 | 상태 |
|---|---|
| s1_01 MJCF 해부 | ⬜ 다음 진행 |
| s1_02 상태 벡터 해부 | ⬜ |
| s1_03 mass & inertia | ⬜ |
| s1_04 damping · stiffness · armature 스윕 | ⬜ |
| s1_05 actuator 4종 비교 | ⬜ |
| s1_06 camera · site · sensor | ⬜ |
| s1_07 FK · IK · ID 수치 검증 | ⬜ |
| s1_08 sjchoi86 tutorial | ⬜ |

### Stage 2 — 접촉과 closed-chain (스켈레톤)
| 챕터 | 상태 |
|---|---|
| s2_01 접촉 anatomy 심화 | ⬜ |
| s2_02 friction/solref/solimp 스윕 | ⬜ |
| s2_03 equality constraint → pseudo closed-chain | ⬜ |
| s2_04 Menagerie 손 모델 tour | ⬜ |
| s2_05 grasp stability 비교 | ⬜ |

### Stage 3 — Planning / Control (스켈레톤)
| 챕터 | 상태 |
|---|---|
| s3_01 PID | ⬜ |
| s3_02 LQR | ⬜ |
| s3_03 MJPC CLI 데모 | ⬜ |
| s3_04 MJPC 커스텀 task | ⬜ |
| s3_05 contact-rich cost 실험 | ⬜ |

### Stage 4 — RL / Imitation Learning (설치 대부분 완료)
| 챕터 | 상태 |
|---|---|
| s4_00 설치 (SB3, imitation, d3rlpy) | 🟨 부분 완료 — SB3/sb3-contrib 설치됨. imitation/d3rlpy은 gymnasium 핀 충돌로 chapter 진입 시 결정 |
| s4_01 dm_control env 작성 | ⬜ |
| s4_02 SB3 PPO 짧은 학습 | ⬜ |
| s4_03 Behavior Cloning | ⬜ |
| s4_04 SAC 비교 | ⬜ |
| s4_05 Offline RL | ⬜ |
| s4_06 Model-based RL (개념) | ⬜ |

### Stage 5 — VLA (설치 대기)
| 챕터 | 상태 |
|---|---|
| s5_00 LeRobot / OpenVLA 개요 + 진입 판단 | ⬜ |
| s5_01 LeRobot quickstart | 🟩 설치 완료 — `~/robotics/external/lerobot/.venv` (Python 3.13) |
| s5_02 OpenVLA inference demo | 🟨 clone만 — install은 CUDA/flash-attn 의존이라 필요 시점으로 보류 |

---

## 환경 / 설치 현황 (2026-04-21)

- **공유 venv**: `~/robotics/mujoco_stack/.venv` (Python 3.12.13 via symlink → `.venv-312`). 이전 3.11은 `.venv.bak.py311`에 백업.
- **LeRobot 전용 venv**: `~/robotics/external/lerobot/.venv` (Python 3.13). torch<2.11 핀 충돌 때문에 공유 venv 통합 불가.
- **설치된 학습용 패키지**: mujoco 3.7.0, mjx 3.7.0, mjlab 1.3.0, warp 3.7.0.1, jax 0.10.0 (CPU), brax 0.14.2, dm-control, mink, pin, casadi, osqp, gymnasium 1.2.3, gymnasium-robotics 1.4.2, stable-baselines3 2.8.0, sb3-contrib 2.8.0, torch 2.11.0, mediapy, rerun-sdk, jupyterlab.
- **클론된 외부 레포**: `~/robotics/external/yet-another-mujoco-tutorial-v3`, `lerobot`, `openvla`.
- **Jupyter kernel**: `MuJoCo Stack (Py 3.12)` 등록.
- **GitHub repo**: https://github.com/terryum/mujoko-practice (public, main).
- **MuJoCo MPC**: 이미 네이티브 빌드됨 (`~/robotics/mujoco_stack/external/mujoco_mpc/build/bin/mjpc`). Stage 3에서 활용.

---

## "나 어디까지 했어?" 복귀 시 Claude Code가 하는 일

1. 이 파일(`CURRENT_STATUS.md`) 읽기.
2. `chapters/` 디렉토리 스캔 — 각 폴더의 `done.md` 유무 교차 검증.
3. 불일치 있으면 이 파일을 실제 상태에 맞춰 업데이트.
4. "다음 챕터" 항목의 복사-붙여넣기 프롬프트를 [`LEARNING_PLAN.md`](./LEARNING_PLAN.md)에서 찾아 코드펜스로 제공.

챕터 완료 시 Claude Code가 추가로 하는 일:
- 해당 챕터 폴더에 `done.md` (요약 + 핵심 배움 3줄) 생성.
- 이 파일의 해당 챕터 행을 `✅`로 변경 + `Last updated` 갱신.
- 필요 시 "한눈 요약" 섹션의 다음 챕터/완료율 업데이트.
