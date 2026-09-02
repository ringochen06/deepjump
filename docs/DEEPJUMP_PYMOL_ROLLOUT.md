# DeepJump rollout 动画：checkpoint、初始构象与 PyMOL 命令

> 只需要最简命令时，请看 [`DEEPJUMP_PYMOL_QUICKSTART.md`](DEEPJUMP_PYMOL_QUICKSTART.md)。正式 500k checkpoint 已下载到本机并完成 SHA256 校验。

这份文档把一个 DeepJump checkpoint 和一个明确的 mdCATH 初始帧重新跑成多状态 PDB，再用 PyMOL 播放或导出 MP4。

> **科学含义：**这里生成的是 DeepJump 自回归 `rollout`，不是逐飞秒积分的分子动力学轨迹，也不是已经验证正确的折叠路径。正式 500k 模型目前不能支持“可稳定地从伸展态折叠到天然态”这一结论。旧的 64×3000 压力测试没有保存中间坐标，所以动画必须从 checkpoint 重新推理。

## 1. 动画里每一帧是什么

- state 1：指定的真实 mdCATH 初始帧。
- state 2：模型从 state 1 预测的下一个粗时间步。
- state 3：把 state 2 重新输入模型所得的下一步，之后依此类推。
- `--delta-ns 1` 表示每个相邻 state 的**名义条件间隔**为 1 ns；这不等于做了 1 ns 的原子级力场积分。
- `--ode-steps 20` 是每次 jump 内部的数值积分步数。导出的 PDB 只保存每次 jump 的端点，不保存这 20 个内部采样子步。
- 模型状态为 `P ∈ R^(N×3)` 的 Cα 坐标，以及 `V ∈ R^(N×13×3)` 的非 Cα 重原子相对偏移；导出脚本将二者还原为 PyMOL 可读的重原子 PDB。

`--mode ode` 使用随机源和 ODE sampler，`--seed` 决定随机重复性。`--mode mean` 是确定性的单次条件均值预测，通常更平滑、更保守，但不是同一种采样分布。

## 2. 导出脚本的产物

运行 [`scripts/export_rollout_pdb.py`](../scripts/export_rollout_pdb.py) 后会生成：

| 文件 | 内容 |
|---|---|
| `rollout.pdb` | 多状态 PDB；state 1 是初始值 |
| `rollout.json` | checkpoint SHA256、初始帧、参数、保存状态数、gate 接受率等 |
| `rollout.pml` | PyMOL 交互播放脚本 |
| `rollout_render.pml` | 无界面逐帧渲染脚本 |
| `rollout_frames/` | PNG 帧目录，执行 render PML 后填充 |

如果轨迹出现 `NaN/Inf`，或坐标绝对值超过 `--max-abs-coordinate`，PDB 会在第一个坏状态前截断，并把原因写入 JSON；这避免把数值崩溃伪装成可播放轨迹。

## 3. 本机已有 40k 模型：先做可运行演示

以下 checkpoint 和输入文件已在本机确认存在：

- checkpoint：`runs/faithful_scaled/last.ckpt`
- step：40,000
- SHA256：`53665fdd9d73deb95bdc374a6f6e96a780755baa513cca61d314a37463e8f715`
- 初始值：`1a92A00`，320 K，replica 0，frame 0，共 50 个残基

从仓库根目录运行：

```bash
cd /Users/ringochen/hkucds/deepjump

export PYTHONPATH="$PWD/src"
export DEEPJUMP_CKPT="$PWD/runs/faithful_scaled/last.ckpt"
export DEEPJUMP_INPUT="/Users/ringochen/hkucds/data/mdcath/data/mdcath_dataset_1a92A00.h5"
export DEEPJUMP_OUT="$PWD/runs/visualization/local40k_1a92A00/rollout.pdb"

python scripts/export_rollout_pdb.py \
  --ckpt "$DEEPJUMP_CKPT" \
  --input-h5 "$DEEPJUMP_INPUT" \
  --temperature 320 \
  --replica 0 \
  --frame 0 \
  --steps 20 \
  --delta-ns 1 \
  --mode ode \
  --ode-steps 20 \
  --integrator euler \
  --tau-max 1.0 \
  --drift-anchor state \
  --seed 0 \
  --device auto \
  --out "$DEEPJUMP_OUT"
```

先检查元数据，尤其是 `saved_rollout_steps` 和 `stop_reason`：

```bash
python -m json.tool "${DEEPJUMP_OUT%.pdb}.json"
```

若只想先确认全链路工作，可改成 `--steps 2 --mode mean --ode-steps 1 --device cpu`。这一短命令已经实际验证，生成的 PDB 被 PyMOL 3.1.8 读回为 1 个对象、3 个 states、每个 state 417 个重原子。

## 4. 正式 500k 模型：可复现路径

正式模型记录为：

- run ID：`20260726T164217Z`
- 代码 commit：`d469bfebc55a087725527721d1798b7f592fb5bb`
- checkpoint：`ckpt_500000.pt`
- checkpoint SHA256：`d0e7ae08f1a9e4f3ae11fa73c45f4e6005e9eac66754070b5b92fcaab91348e6`
- OBS：`obs://deepjump-mdcath-cn4-ringochen/deepjump-formal500k/seed0/20260726T164217Z`
- 证据分类：`closer-to-paper_not_exact_reproduction`

### 4.1 下载并校验 checkpoint

如果 checkpoint 尚未在本机：

```bash
cd /Users/ringochen/hkucds/deepjump

export DEEPJUMP_FORMAL_DIR="$PWD/artifacts/formal500k/20260726T164217Z"
export DEEPJUMP_FORMAL_CKPT="$DEEPJUMP_FORMAL_DIR/ckpt_500000.pt"
export DEEPJUMP_FORMAL_SHA="d0e7ae08f1a9e4f3ae11fa73c45f4e6005e9eac66754070b5b92fcaab91348e6"

mkdir -p "$DEEPJUMP_FORMAL_DIR"
obsutil cp \
  "obs://deepjump-mdcath-cn4-ringochen/deepjump-formal500k/seed0/20260726T164217Z/ckpt_500000.pt" \
  "$DEEPJUMP_FORMAL_CKPT"

export DEEPJUMP_ACTUAL_SHA="$(shasum -a 256 "$DEEPJUMP_FORMAL_CKPT" | awk '{print $1}')"
test "$DEEPJUMP_ACTUAL_SHA" = "$DEEPJUMP_FORMAL_SHA"
```

`test` 无输出且退出码为 0 才表示哈希一致。

### 4.2 用训练时的精确代码版本推理

不要直接切换当前脏工作树。建立 detached worktree，并让导出脚本从该 worktree 导入 DeepJump：

```bash
cd /Users/ringochen/hkucds/deepjump

export DEEPJUMP_ROOT="$PWD"
export DEEPJUMP_FORMAL_SRC="/tmp/deepjump-formal-d469bfeb"
export DEEPJUMP_FORMAL_COMMIT="d469bfebc55a087725527721d1798b7f592fb5bb"

git worktree add --detach "$DEEPJUMP_FORMAL_SRC" "$DEEPJUMP_FORMAL_COMMIT"
```

然后先跑一条不带 gate 的原始 20-step 轨迹。GPU 机器用 `--device cuda`；本机可改为 `auto`，但 H128 正式模型在 CPU/MPS 上可能很慢或内存不足。

```bash
export DEEPJUMP_FORMAL_INPUT="/Users/ringochen/hkucds/data/mdcath/data/mdcath_dataset_1a92A00.h5"
export DEEPJUMP_FORMAL_OUT="$DEEPJUMP_ROOT/runs/visualization/formal500k_1a92A00/rollout.pdb"

PYTHONPATH="$DEEPJUMP_FORMAL_SRC/src" \
python "$DEEPJUMP_ROOT/scripts/export_rollout_pdb.py" \
  --ckpt "$DEEPJUMP_FORMAL_CKPT" \
  --input-h5 "$DEEPJUMP_FORMAL_INPUT" \
  --temperature 320 \
  --replica 0 \
  --frame 0 \
  --steps 20 \
  --delta-ns 1 \
  --mode ode \
  --ode-steps 20 \
  --integrator euler \
  --tau-max 1.0 \
  --drift-anchor state \
  --seed 0 \
  --device cuda \
  --out "$DEEPJUMP_FORMAL_OUT"
```

云上已有 checkpoint 路径时，可直接把 `--ckpt` 换为：

```text
/data/deepjump-formal500k-runs/20260726T164217Z/ckpt_500000.pt
```

初始 HDF5 也应换成云上实际 `/data/mdcath/.../mdcath_dataset_*.h5` 路径。

### 4.3 仅为观看而使用 geometry gate

如果原始 rollout 很快崩溃，可添加 `--gate` 并增加 `--steps`。例如把上面的命令改为：

```text
--steps 100 --gate
```

gate 会检查 Cα–Cα 几何，拒绝时重复上一帧。因此：

- 它能让文件保持可视化，但不是力场、不是能量守恒，也不是物理采样器。
- 重复帧代表拒绝，不代表蛋白“停在稳定中间态”。
- 必须同时报告 JSON 中的 `acceptance_rate`；不能只展示看起来稳定的动画。

## 5. PyMOL 交互播放

本机 PyMOL 3.1.8 的命令行程序位于：

```bash
export DEEPJUMP_PYMOL="/Applications/PyMOL.app/Contents/bin/pymol"
```

导出脚本已经生成交互 PML，直接打开即可自动循环：

```bash
"$DEEPJUMP_PYMOL" "${DEEPJUMP_OUT%.pdb}.pml"
```

正式模型则把 `DEEPJUMP_OUT` 换成 `DEEPJUMP_FORMAL_OUT`。

在 PyMOL 控制台可使用：

```pml
count_states rollout
frame 1
frame 10
mplay
mstop
set movie_loop, on
set movie_fps, 20
```

若只想比较内部构象变化、去掉每帧的整体旋转和平移，可运行：

```pml
intra_fit rollout and name CA, 1
```

这会改变显示坐标，只适合视觉比较；不要把对齐后的坐标再当作原始输出分析。

不使用自动 PML 时，最小 PyMOL 命令是：

```pml
load /absolute/path/to/rollout.pdb, rollout
hide everything, all
show cartoon, rollout
show sticks, rollout and not hydro
set all_states, off
mset 1 -21
set movie_loop, on
set movie_fps, 20
mplay
```

`21` 必须等于 JSON 的 `saved_states_including_initial`，即 20 个 rollout steps 加 1 个初始 state。

## 6. 无界面渲染并导出 MP4

自动生成的 render PML 已包含真实 state 数，不需要手工填写：

```bash
"$DEEPJUMP_PYMOL" -cq "${DEEPJUMP_OUT%.pdb}_render.pml"

ffmpeg -y \
  -framerate 20 \
  -i "${DEEPJUMP_OUT%.pdb}_frames/frame_%04d.png" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -crf 18 \
  "${DEEPJUMP_OUT%.pdb}.mp4"
```

本流程已经以 3 个 states 实测：PyMOL 生成 3 张 1280×720 PNG，FFmpeg 成功编码为 H.264、`yuv420p` 的 3-frame MP4。

## 7. 怎样更换初始值

只需要修改四个参数：

```text
--input-h5 /path/to/mdcath_dataset_DOMAIN.h5
--temperature 320
--replica 0
--frame 0
```

脚本会从 HDF5/PSF 读取序列、原子名、残基编号、`atom_mask` 和 `bond_mask`。如果蛋白比 checkpoint 的训练 crop 更长，默认取中间连续 crop；可用 `--crop-length N` 显式指定，且 JSON 会记录实际 `[start, stop]`。

当前脚本**不直接接受任意 PDB 或 1ENH 的伸展链**。任意 PDB 要先构造与训练完全一致的 `(P,V)`、残基类型、原子掩码和拓扑掩码；否则即使能画出来，也不能声称输入与模型训练接口一致。

## 8. 解读动画时至少检查什么

1. `rollout.json` 的 checkpoint SHA、初始帧和 `stop_reason`。
2. gate 是否开启，以及 `acceptance_rate`；低接受率动画会有大量重复帧。
3. 是否出现链断裂、原子爆炸、侧链异常或整体快速坍缩。
4. 动画只说明模型生成了某条路径。要讨论路径是否“对”，仍需与真实轨迹或实验可观测量比较 RMSD、接触形成顺序、二级结构、TICA/JSD 和多 seed 分布，不能依靠单条 PyMOL 电影判断。
