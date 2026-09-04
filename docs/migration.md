# 迁移到另一台服务器

代码、文档和实验结果都在 git 里（约 2MB），数据集和 venv 不在。所以搬迁 = 搬仓库 + 重建这两样。

## 1. 搬仓库

**方式 A：单文件 bundle**（不需要任何远端和凭据）

```bash
# 旧机器
git bundle create msa-$(git rev-parse --short HEAD).bundle --all   # 约 280KB
scp msa-*.bundle 新机器:~/

# 新机器
git clone msa-<hash>.bundle msa && cd msa
git remote remove origin        # 解除对 bundle 文件的引用
```

bundle 带全部历史与结果产物，克隆后可直接 `scripts/verify_runs.py` 重新审计历史数字。

**方式 B：推到远端**

```bash
git remote add origin <你的仓库地址>
git push -u origin main
```

两种方式都可以，B 更适合长期多机协作。

## 2. 数据集

都不入库：MOSI 879MB、**MOSEI 18GB**（aligned 4.7G + unaligned 13.7G）。**一条命令下载并校验**：

```bash
bash scripts/fetch_dataset.sh            # mosi
bash scripts/fetch_dataset.sh mosei      # 需要约 22G 余量
bash scripts/fetch_dataset.sh all
```

它幂等（已就位且哈希对得上就直接退出），并以 sha256 校验收尾。手动放置也可以，目录结构：

```
datasets/CMU-MOSI/Processed/aligned_50.pkl
datasets/CMU-MOSI/Processed/unaligned_50.pkl
datasets/CMU-MOSI/label.csv
```

来源是 MMSA（THUIAR）的预处理发布，MOSI 文件夹：

- Google Drive: <https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk>
- 百度网盘: <https://pan.baidu.com/s/1a1bDX5htPsZjsRyHcvCKHw?pwd=qq0b>（提取码 qq0b）

**注意文本特征是 BERT 的，不是 CMU-Multimodal-SDK 原版的 GloVe。**

**MOSEI 的两个大文件走不同的路径。** `aligned_50.pkl`（4.7GB）gdown 能下；`unaligned_50.pkl`（13.7GB）不能——Drive 对这个尺寸的文件返回"无法病毒扫描"中间页，gdown 在上面失败并报 `Cannot retrieve the public link... but Gdown can't`，读起来像权限或配额问题，两者都不是。那个页面带一个 `uuid` 确认令牌，所以脚本对这类文件改用 curl 打 `drive.usercontent.google.com` 并附上令牌，且带 `-C -` 断点续传（13.7GB 断了不必从头再来）。

**不要对着那个 Drive 目录直接 `gdown --folder`。** 它会递归整个发布（CH-SIMS 与 MOSI 的原始视频，几千个 .mp4），枚举到一半被 Drive 返回 500 掐断，结果是一个文件都没下到。`fetch_dataset.sh` 因此逐层解析、只取 MOSI 的 `Processed/` 与 `label.csv`，folder ID 直接记在脚本里——发布若被重传会 404，是响亮的失败而不是悄悄下错文件。

若两台机器网络互通，直接 `rsync -a 旧机器:~/msa/datasets/ datasets/` 更快。

三个文件的 sha256 记录在 `src/msa/config.py` 的 `DatasetSpec.file_sha256` 里，`scripts/setup.sh` 会自动校验；也可单独查：

```bash
.venv/bin/python scripts/check_data.py --verify-files
```

**这一步不能省。** 一个截断的下载或一份重新抽取的特征会被静默地训练下去，而所有已记录的数字都建立在这份特定数据上。

## 3. 环境

**先确认解释器是 3.12。** `requirements-lock.txt` 在 3.11 以下根本解析不了（`networkx==3.6.1` 没有轮子），而 305 次已入库运行记录的全是 3.12.x。系统 python 更老时：

```bash
uv python install 3.12.3
PYTHON=$(uv python find 3.12.3) bash scripts/setup.sh
```

`setup.sh` 会自己拦截并打印这段提示；`scripts/check_env.py` 是对应的闸门，它同时核对 lock 里每一条钉死的版本是否装上且一致。

```bash
bash scripts/setup.sh
```

它会建 venv、按平台选 PyTorch 轮子（有 NVIDIA GPU → cu128；macOS → 默认轮子带 MPS；其余 → cpu）、装 `requirements-lock.txt` 的固定版本、校验数据集哈希，最后跑一遍全部闸门。

强制指定通道用 `TORCH_INDEX=cpu bash scripts/setup.sh`。

**PyTorch 不写进 lock 文件**，因为轮子与平台绑定。开发机用的是 torch 2.11.0+cu128——Blackwell（sm_120）显卡必须 cu128 及以上，PyPI 默认轮子没有 sm_120 kernel，会在第一次 CUDA 运算时报错。

## 4. 确认落地

```bash
bash scripts/check_all.sh
```

闸门全绿说明**代码、数据、已存结果三者自洽**。注意它**没有**说明的那件事：

> **全绿 ≠ 旧数字在新机器上重现得出来。** `check_all.sh` 校验的是 committed 的 `result.json`（指标能否由 committed 的预测重算、验收判据是否成立），它**不重新训练**。真正重训并比对的是 `bash scripts/reproduce_all.sh`（约 25 分钟）。

预期内的差异有三点，第三点是 2026-08-01 换机器时实测订正的：

- **跨设备类型的数字不会逐比特相同**（CUDA / MPS / CPU 归约顺序不同），各设备内部可复现。
- **CPU 上还须钉住线程数**（`--num-threads N`），否则线程数不同结果就不同。
- **换机器后数字会变，即使设备类型相同、线程数钉住、库版本完全一致。** 换 GPU 型号会变；**换 CPU 型号同样会变**（很可能是 torch CPU kernel 按指令集分派所致）。而且**不是"细微出入"**：LF-LSTM seed 42 从 5070 Ti 换到 5080，best_epoch 从 13 变成 23、MAE 从 0.9543 变成 0.9318——微小数值差被 early stopping 的模型选择放大了。差值仍在该模型的 seed 间标准差（0.039）之内，所以不影响多 seed 结论，但单 seed 的哈希与数字都对不上。

**因此换机器后哈希对不上时，不要直接归因于硬件。** 先用 2×2 交叉排除代码：把产生旧哈希的那个 commit 用 `git worktree` 签出，在**新机器上**同时跑新旧两份代码 × 两个设备。同机器上新旧代码逐比特相同，才能说差异来自机器。完整方法、命令与那次的实测数据见 [`investigations.md#cross-machine-hash`](investigations.md#cross-machine-hash)。

同理，CLAUDE.md 约定 6 的 LF-LSTM 锚点哈希 `172967c7dced83b2` **绑定产生它的那台机器**，换机器后需要在新机器上重立基线，并标注它绑定哪台。

还有一道闸门在新机器上会静默失效：`check_almt_equivalence.py` 在没有 MMSA 检出时 SKIP 但记 PASS（有意为之，让 fresh clone 走绿）。它是约定 5 里最关键的那道数值等价检查。**换机器后跑一条命令把它接回来**：

```bash
bash scripts/setup_mmsa_reference.sh     # clone MMSA + 建参照 venv + 建 shim + 验证
```

默认装到仓库内的 `.mmsa-reference/`（已 gitignore），不再是 `/workspace`——路径写死在仓库外正是这道闸门两次静默失效的病根（一次是换机器后检出没了，一次更早，shim 路径指向某次会话的临时目录）。解析顺序统一在 `scripts/_reference_paths.py`：`$MMSA_DIR`/`$MMSA_SHIM` → `.mmsa-reference/` → `/workspace`（仅当真的存在）。**不需要 export 任何变量**，裸跑 `check_almt_equivalence.py` 就能找到它。

参照 venv 用 `.venv` 的同一个解释器建，不是系统 `python3`：shim 里 `tokenizers`/`safetensors`/`sentencepiece` 都是按 CPython ABI 编译的，版本对不上会在 import 时炸。前两台机器的系统 python 恰好同版本，所以这个坑没暴露。

脚本自身以那道等价检查收尾，且**不看退出码看结论**（SKIP 时退出码也是 0），拿不到 `EQUIVALENT` 就报错退出。跑完 `check_all.sh` 里的 `almt equivalence` 才是真在检查，而不是在跳过。

为什么要单独一个环境：MMSA 的模型在 transformers 5 下根本 import 不了，而本项目用的是 5.x，所以给它单独装一棵 transformers 4.44.2，只在跑 MMSA 代码的进程里插到 `sys.path` 前面。**torch 与 numpy 故意不在那棵树里**——两份实现必须跑在同一个张量库上，否则比较的就不只是实现。shim 因此是"挑出来的包的符号链接"，不是整个 site-packages。

`best.pt` 默认训练结束即删除，也不入库。需要某个 checkpoint 时用 `--keep-checkpoint` 重跑——按 `result.json` 里记录的命令，结果逐比特一致。
