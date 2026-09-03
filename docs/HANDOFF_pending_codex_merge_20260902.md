# 待合并：`pending-codex-merge` 上的三个提交

- 日期：2026-09-02
- 分支：`pending-codex-merge`（本地，指向 `1897f3a`）
- 状态：**未合并，需要熟悉 endpoint/cloud 设计的人处置**

## 为什么单独放着

本地 `cloud-fullscale` 曾与 `origin/cloud-fullscale` 分叉：双方各有 14 个提交，
共同祖先是 `fd2eec3`。远端那 14 个是 Codex 的 discriminator 系列。

我这边的 14 个里，**11 个与 Codex 的改动零文件重叠**，已 cherry-pick 到
`sampler-measurement-fixes` 并推送，`578 passed`。

剩下 3 个有实质冲突，按 `COLLABORATION_CONTRACT.md`「A review is independent only
when the reviewer did not implement the reviewed change」的精神，不该由我单方面
决定如何合并 Codex 领域的代码。

| 提交 | 内容 | 冲突面 |
|---|---|---|
| `d7ef80b` | fast-dev 门控确定化 + 梯度健康度 | `tests/test_training_gates.py` |
| `4825b2e` | external endpoint panel 门控与裁决加固 | `scripts/external_endpoint_identity.py`、`tests/test_external_endpoint_panel_adjudication.py` |
| `92f173f` | seed-1 续训与单批次 sanity 云配置 | `tests/test_cloud_configs.py` |

## 三者互相耦合，不能单独摘

`d7ef80b` 单独 cherry-pick 到 `origin/cloud-fullscale` **无文本冲突但测试失败**：
它带进的 `tests/test_training_gates.py` 含有另外两个提交的断言——
`EXPECTED_TRAIN_SEED=${EXPECTED_TRAIN_SEED:-0}`（属于 `4825b2e`）和
`cloud/huawei/run_full_tensor_seed1_2000.sh`（属于 `92f173f`）。

所以要么三个一起合，要么先把 `test_training_gates.py` 的断言按提交拆开。

## 冲突的实质

`4825b2e` 与 Codex 的改动都扩展了 `verify_multidomain_checkpoint()` 的签名：

- 我方新增 `expected_data_config` / `expected_model_config` / `expected_train_config`
- Codex 新增 `expected_train_seed`

两者不互斥，合并大概率是把四个参数并入同一签名，但**具体语义应由设计者判断**。

## 建议处置

1. 由 Codex 或了解 endpoint 门控设计的人合并这三个提交；
2. 或者判定它们已被 Codex 的等价工作取代，直接作废；
3. 无论哪种，处置后删除 `pending-codex-merge`。

## 已完成的清理

- 本地 `cloud-fullscale` 已重置为 `origin/cloud-fullscale`，不再显示分叉；
- 两个临时备份分支（`backup-before-rebase`、`backup-before-trailer-strip`）与
  `cloud-fullscale` 逐字节相同，已删除。
