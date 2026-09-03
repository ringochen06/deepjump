# PyMOL 可视化命令

本文档的每条命令都在 **PyMOL 3.1.8** 上 headless 跑通验证过，不是凭记忆写的。

相关结论见 [`FINDING_no_sequence_specific_folding_20260902.md`](FINDING_no_sequence_specific_folding_20260902.md)。

## 0. 一个必须知道的命名陷阱

`split_states` 用 **state 编号**命名对象，**不加下划线**：

```pml
split_states rollout, 601, 601, prefix=step
# 生成的对象叫 step0601
# 不是 step_0601，也不是按顺序编号的 step0001
```

写错会报 `Invalid selection name`。验证方法：

```pml
print(cmd.get_names('objects'))
```

## 1. 收缩与折叠的对比（主图）

```bash
cd /Users/ringochen/hkucds/deepjump
/Applications/PyMOL.app/Contents/bin/pymol docs/pymol/compaction_vs_folding.pml
```

蓝色是 `1hw7A02` 的真实天然态，红色是模型从完全伸展链跑 600 跳后的终态，叠合
RMSD **13.05 Å**。两者体积相同、结构无关。

## 2. 播放收缩动画

```bash
/Applications/PyMOL.app/Contents/bin/pymol \
  runs/visualization/folding_probe_20260902/1hw7A02_from_extended.pml
```

601 帧，自动循环。Rg 轨迹 51.95 → 11.62 → 9.83 → 10.96 Å。

```pml
mstop          # 暂停
mplay          # 继续
frame 1        # 伸展起点
frame 601      # 终态
set movie_fps, 10
```

## 3. 手动做叠合对比

```pml
load runs/visualization/folding_probe_20260902/1hw7A02_native.pdb, native
split_states rollout, 601, 601, prefix=final
delete rollout
hide everything
show cartoon, native final0601
color marine, native
color firebrick, final0601
align final0601, native
set cartoon_transparency, 0.2, native
orient native
```

## 4. 收缩过程三帧

`grid_mode` 共用一个相机，会被 52 Å 的伸展链撑爆比例，后面几格缩到看不见。要么
接受，要么逐帧单独渲染（见第 6 节）。

```pml
split_states rollout, 1, 1, prefix=step
split_states rollout, 76, 76, prefix=step
split_states rollout, 601, 601, prefix=step
hide everything
show cartoon, step0001 step0076 step0601
color grey60, step0001
color orange, step0076
color firebrick, step0601
set grid_mode, 1
zoom step0076
```

## 5. 读出定量值

```pml
print('states:', cmd.count_states('rollout'))
print('RMSD:', cmd.align('final0601','native')[0])
print('objects:', cmd.get_names('objects'))
```

回转半径随帧变化：

```pml
python
from pymol import cmd
import numpy as np
for s in (1, 76, 151, 301, 601):
    xyz = np.array(cmd.get_coords('rollout and name CA', state=s))
    c = xyz - xyz.mean(0)
    print(s, round(float(np.sqrt((c**2).sum(1).mean())), 2))
python end
```

## 6. Headless 渲图

命令行直接出 PNG，不开界面：

```bash
/Applications/PyMOL.app/Contents/bin/pymol -cq docs/pymol/compaction_vs_folding.pml -d "
png /tmp/overlay.png, width=1500, height=950, dpi=150, ray=1
"
```

逐结构单独定标（`grid_mode` 比例问题的解法）：

```bash
/Applications/PyMOL.app/Contents/bin/pymol -cq docs/pymol/compaction_vs_folding.pml -d "
python
from pymol import cmd
for obj in ('step0001','step0076','step0601','native'):
    cmd.disable('all'); cmd.enable(obj)
    cmd.orient(obj); cmd.zoom(obj, 3)
    cmd.png(f'/tmp/panel_{obj}.png', width=800, height=800, dpi=150, ray=1)
python end
"
```

`-c` 无界面、`-q` 静默、`-d` 执行命令。

## 7. 生成新的 rollout

```bash
python scripts/export_rollout_pdb.py \
  --ckpt artifacts/formal500k/20260726T164217Z/ckpt_500000.pt \
  --input-h5 /Users/ringochen/hkucds/data/mdcath/data/mdcath_dataset_1hw7A02.h5 \
  --initial-pdb runs/visualization/folding_probe_20260902/1hw7A02_extended.pdb \
  --steps 600 --ode-steps 20 --mode ode \
  --out runs/visualization/<name>/from_extended.pdb
```

会同时产出 `.pdb`、`.pml`、`_render.pml` 和一份记录采样器设置的 `.json`。
**打开 `.pml` 而不是 `.pdb`** —— PML 会强制单 state 显示并设好相机与配色。

确认用的是修复后的采样器：

```bash
grep project_v_atom_mask runs/visualization/<name>/from_extended.json   # 应为 true
```

## 8. 已有的其他产物

| 路径 | 内容 |
|---|---|
| `runs/visualization/formal500k_1a92A00/rollout.pml` | 天然态起步，`1a92A00` |
| `runs/visualization/formal500k_1a92A00_hot_unfolded/` | 450 K 去折叠态起步 |
| `runs/visualization/folding_probe_20260902/from_extended.pml` | `1a92A00` 伸展起步 |

**`1a92A00` 那条伸展起步的不要用来下结论**：Rg 走向是 49.8 → 18.1 → 22.9 → 64.6 Å，
先收缩后散开，351 步触发安全阈值退出，且该域不在探针实测的三个域内。要看用
`1hw7A02_from_extended.pml`。

## 9. 注意

`runs/` 在 `.gitignore` 里，轨迹 PDB（单个约 20 MB）不入库。图存在
`docs/figures/`，脚本存在 `docs/pymol/`，都可由上面的命令重新生成。
