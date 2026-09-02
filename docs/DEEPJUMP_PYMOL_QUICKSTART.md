# DeepJump 正式 500k 模型动画：本机最简版

模型已下载并校验，本机路径为：

```text
/Users/ringochen/hkucds/deepjump/artifacts/formal500k/20260726T164217Z/ckpt_500000.pt
```

SHA256：

```text
d0e7ae08f1a9e4f3ae11fa73c45f4e6005e9eac66754070b5b92fcaab91348e6
```

## 1. 生成 20-step 动画

```bash
cd /Users/ringochen/hkucds/deepjump
bash scripts/run_formal500k_animation_local.sh
```

默认使用 `mean` 模式，CPU 上实测约数秒完成。初始值是：

```text
mdCATH domain 1a92A00 / 320 K / replica 0 / frame 0
```

## 2. 用 PyMOL 播放

请打开 `.pml`，不要直接打开 `.pdb`；PML 会强制一次只显示一个 state，并设置 cartoon、侧链棒状、颜色、相机和循环播放。

```bash
/Applications/PyMOL.app/Contents/bin/pymol \
  /Users/ringochen/hkucds/deepjump/runs/visualization/formal500k_1a92A00/rollout.pml
```

打开后会自动循环播放。暂停与继续：

```pml
mstop
mplay
```

如果当前窗口已经把全部 states 叠在一起，先运行：

```pml
set all_states, off
hide everything, all
dss all, state=1
show cartoon, all
frame 1
mplay
```

## 3. 现成文件

- PyMOL 轨迹：`runs/visualization/formal500k_1a92A00/rollout.pdb`
- PyMOL 播放脚本：`runs/visualization/formal500k_1a92A00/rollout.pml`
- MP4：`runs/visualization/formal500k_1a92A00/rollout.mp4`
- 参数与校验记录：`runs/visualization/formal500k_1a92A00/rollout.json`

当前文件包含初始帧加 20 个模型 jump，共 21 states。这里是模型生成的 rollout 动画，不是力场 MD，也不能直接视为已经验证正确的折叠路径。

## 可选：随机 ODE 版本

随机 ODE 版本更接近模型的采样方式，但更慢，也更可能在长 rollout 中发散：

```bash
cd /Users/ringochen/hkucds/deepjump
bash scripts/run_formal500k_animation_local.sh 20 ode
```

第二个参数只能是 `mean` 或 `ode`；第一个参数是 rollout steps。

## 4. 从展开态开始

这里提供两个起点。更推荐先看实际高温展开帧：

```bash
cd /Users/ringochen/hkucds/deepjump
bash scripts/run_formal500k_animation_local.sh 100 mean unfolded

/Applications/PyMOL.app/Contents/bin/pymol \
  /Users/ringochen/hkucds/deepjump/runs/visualization/formal500k_1a92A00_hot_unfolded/from_unfolded.pml
```

这个起点是 `1a92A00 / 450 K / replica 2 / frame 211`，相对 320 K native-frame 的初始 Cα RMSD 约 19.83 Å，来自真实 mdCATH 轨迹。

也可以从 PyMOL 构造的完全伸展链开始：

```bash
cd /Users/ringochen/hkucds/deepjump
bash scripts/run_formal500k_animation_local.sh 100 mean extended

/Applications/PyMOL.app/Contents/bin/pymol \
  /Users/ringochen/hkucds/deepjump/runs/visualization/formal500k_1a92A00_extended/from_extended.pml
```

这两条都是能力边界测试，不是折叠成功结果。实测两者都会很快破坏 Cα 几何并发散：高温起点在 step 1 的平均 Cα 键长已经变成约 2.34 Å；完全伸展起点在 step 1 约为 2.96 Å，step 5 后发生明显几何爆炸。因此它们说明正式 500k 模型目前不能可靠地做从头折叠。
