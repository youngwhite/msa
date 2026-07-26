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

不入库（879MB）。放到 `datasets/CMU-MOSI/` 下，目录结构：

```
datasets/CMU-MOSI/Processed/aligned_50.pkl
datasets/CMU-MOSI/Processed/unaligned_50.pkl
datasets/CMU-MOSI/label.csv
```

来源是 MMSA（THUIAR）的预处理发布，MOSI 文件夹：

- Google Drive: <https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk>
- 百度网盘: <https://pan.baidu.com/s/1a1bDX5htPsZjsRyHcvCKHw?pwd=qq0b>（提取码 qq0b）

**注意文本特征是 BERT 的，不是 CMU-Multimodal-SDK 原版的 GloVe。**

若两台机器网络互通，直接 `rsync -a 旧机器:~/msa/datasets/ datasets/` 更快。

三个文件的 sha256 记录在 `src/msa/config.py` 的 `DatasetSpec.file_sha256` 里，`scripts/setup.sh` 会自动校验；也可单独查：

```bash
.venv/bin/python scripts/check_data.py --verify-files
```

**这一步不能省。** 一个截断的下载或一份重新抽取的特征会被静默地训练下去，而所有已记录的数字都建立在这份特定数据上。

## 3. 环境

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

六道闸门全绿即代表新机器与旧机器状态一致。注意两点预期内的差异：

- **跨设备类型的数字不会逐比特相同**（CUDA / MPS / CPU 归约顺序不同），各设备内部可复现。新机器若换了 GPU 型号，重跑的数字可能与 `docs/experiments.md` 有细微出入——这是硬件事实，不是回归。
- **CPU 上还须钉住线程数**（`--num-threads N`），否则线程数不同结果就不同。

`outputs/` 里的 `best.pt` 不入库（584MB）。需要某个 checkpoint 就按 `result.json` 里记录的命令重跑，运行是逐比特可复现的。
