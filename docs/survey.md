# 调研台账

这份文件回答三个问题，每一栏都必须可复查：**扫了哪里**、**每个方法的证据强度有多高**、**什么被排除了以及为什么**。

正面清单说明做了什么；**排除清单说明没做什么**——后者才是防"以为做全了"的东西。年份、复现判定与逐模型解读在 [`storyline.md`](storyline.md)，此处不重复。

## 核实规矩

**代码可用性只认两样证据：论文全文解析 + 仓库实地 HTTP 请求。**

会议页面的元数据不作数。ACL Anthology 与 CVF 的论文页**都不显示**正文里的代码链接，据此判断在 2026-08-02 一天内连续误判四篇：ConFEDE（差点降到理解档）、EBMC / MoLAN / MMA（一度记为"无代码"，实际三个仓库都存在）。

操作顺序：

```bash
curl -sL -A "<浏览器 UA>" -o paper.pdf <PDF URL>      # CVF 拒绝脚本式请求，要带 UA
python -c "from pypdf import PdfReader; ..."          # 抽全文
# 正则找 github|gitlab|gitee|bitbucket|anonymous
# 注意：URL 会被 PDF 换行折断，去空白后再匹配（MoLAN 就是这样才抓全的）
curl -s -o /dev/null -w "%{http_code}" <repo URL>     # 每个链接实地请求，记状态码
```

三条已踩实的教训：

- **承诺 ≠ 交付**：P-RMF（ACL 2025 长文）正文写 "Code will be available at …"，实测 **404**。
- **抓取失败 ≠ 不存在**：OpenReview 有人机验证墙、CVF 对无 UA 请求返回 403、部分 PDF 是压缩流。**换权威副本再判**，不要把工具失败写成结论。
- **有代码 ≠ 可用**：FeaDA 的仓库存在但无 README、无 LICENSE。无许可证在法律上不可复用。

## 扫描规程

- **出处**：ACL / NAACL / EMNLP / Findings / IJCNLP-AACL、CVPR / ICCV / WACV、AAAI、ICLR、ACM MM
- **年份**：滚动近三年（当前 2024–2026）+ 故事线所需的奠基之作
- **周期**：每季度一次，或投稿前一次。**每次重扫记日期**——没有日期的"我搜过了"不可复查
- **不收** arXiv 预印本，除非同一篇已有同行评审版本

## 实现出处

每个进入对照表的模型必须带这一列。它决定结论能写成什么样。

| 标记 | 含义 | 可以怎么写 |
|---|---|---|
| `ported` | 移植自官方或参照实现 | "复现该方法" |
| `reimpl+equiv` | 自己实现，且通过权重复制的数值等价测试 | "复现该方法" |
| `reimpl` | 自己实现，无参照可验证，消融方向一致 | "重新实现；与报告值相差 X SE，**实现忠实性未经验证**" |
| `reimpl-partial` | 论文细节不全，含我们的补充设定 | "含我们的补充设定，**不作为该方法的成绩**" |

两条铁律：**比参照『更好』先当 bug 信号排查**（`#almt-better-than-reference` 的教训）；**无参照时**用消融方向一致性 + 参数量对表 + 逐公式退化测试替代等价测试——其中消融方向最有力，因为它是内部对照，不依赖绝对标定。

## 台账 A：已在对照表中

| 模型 | 参照来源 | 实现出处 | 备注 |
|---|---|---|---|
| ef_lstm / lf_dnn / tfn / lmf / mfn / graph_mfn / mult / misa / self_mm / cenet / tetfn | MMSA 公开表 + 我们跑它的代码 | `ported` | 11 个有表内参照值 |
| bert_mag | 我们跑 MMSA 的代码 | `reimpl` | 直接对着 transformers 写，未转抄；原作者实现 `WasifurRahman/BERT_multimodal_transformer` |
| mmim | 我们跑 MMSA 的代码 | `ported` | 已对照原作者 `declare-lab/Multimodal-Infomax`，MMILB 与 CPC 逐行一致 |
| almt | 我们跑 MMSA 的代码 | `reimpl+equiv` | 权重复制等价测试，最大绝对差 `0.000e+00` |
| text_bert | 无（对照组） | 自建 | 故事线的关键对照 |
| lf_lstm | 无（自建基线） | 自建 | 方差标尺（seed 间 MAE sd 0.039） |

## 台账 B：第 1 阶段收口的两个

两者 MMSA 的 MOSI 表都没有行，参照只能用它自己的代码（10 seed 已于 2026-08-02 在本机跑齐）。**原作者实现也都在**，故可按约定 5 做双向核对。

| | MCTN | MFM |
|---|---|---|
| 标题 | Found in Translation: Learning Robust Joint Representations by Cyclic Translations Between Modalities | Learning Factorized Multimodal Representations |
| 作者 | Hai Pham\*, Paul Pu Liang\*, Thomas Manzini, Louis-Philippe Morency, Barnabás Póczos | Yao-Hung Hubert Tsai\*, Paul Pu Liang\*, Amir Zadeh, Louis-Philippe Morency, Ruslan Salakhutdinov |
| 出处 | **AAAI 2019**, vol. 33, pp. 6892-6899 | **ICLR 2019**（PDF 页眉印 "Published as a conference paper at ICLR 2019"） |
| 权威 PDF | [arXiv:1812.07809](https://arxiv.org/pdf/1812.07809) | [arXiv:1806.06176](https://arxiv.org/pdf/1806.06176) |
| 官方代码 | [hainow/MCTN](https://github.com/hainow/MCTN)（正文脚注）→ **200** | [pliang279/factorized](https://github.com/pliang279/factorized/)（正文）→ **200** |
| 已知 MMSA 缺陷 | teacher forcing 是空操作：`dec_input = trg[t] if teacher_force else top1`，两分支同值，`random.random()` 只消耗 RNG；配置里 `update_epochs` / `contrast` / `mem_size` / `add_va` 其 trainer 一个都没读 | `learning_rate: 0.002` 从未生效——trainer 的 `optim.Adam(...)` 没传 `lr`，实际用 Adam 默认 1e-3 |

**MFM 原论文用 CMU-MultimodalSDK 特征（GloVe 系），不是 MMSA 的 BERT 特征**，且写明"所有基线都做了大范围超参搜索后重训"。故其数字与我们不可直接比——这正是验收标准取 MMSA 表而非原论文的原因，在 MFM 上再次成立。原论文超参查阅记录进 docstring，不作达标目标。

两者的 `KeyEval` 都是 `Loss`，即按**复合损失**选模型（MFM 是 disc+gen+mmd，MCTN 是 0.1×三路重构+L1），不是按 MAE。与 ALMT 需要 `--select-on mse` 同类，移植时要专门处理。

## 台账 C：近三年调研（核实于 2026-08-02）

| 方法 | 出处 | 数据集 | 官方代码 | 实测 | 档位 |
|---|---|---|---|---|---|
| EBMC | CVPR 2026, pp.30183-30193 | MOSI, MOSEI, IEMOCAP | [kangverse/EBMC](https://github.com/kangverse/EBMC) | 200 | 验收档 |
| MoLAN | Findings of ACL 2026 | MOSI, MOSEI, SIMS, IEMOCAP | [betterfly123/MoLAN-Framework](https://github.com/betterfly123/MoLAN-Framework) | 200 | 验收档 |
| DPDF-LQ | EMNLP 2025 主会 | MOSI, MOSEI | [ZhouMiaoGX/DPDF-LQ](https://github.com/ZhouMiaoGX/DPDF-LQ) | 200 | **优先**——带预训练权重 + MIT，可做等价测试 |
| MMA | NAACL 2025 长文 | MOSI, MOSEI（摘要未提，正文有） | [MMA4MSA/MMA](https://github.com/MMA4MSA/MMA) | 200 | 验收档 |
| ConFEDE | ACL 2023 长文, pp.7617-7630 | MOSI, MOSEI, CH-SIMS | [XpastaX/ConFEDE](https://github.com/XpastaX/ConFEDE/) | 200 | 验收档；对比学习方向的对口起点 |
| DMD | CVPR 2023 (highlight), pp.6631-6640 | 待核 | [mdswyz/DMD](https://github.com/mdswyz/DMD)，MIT | 200 | 验收档；提供预训练权重 |
| CMAD | ICCV 2025, pp.4626-4636 | MOSEI 等五个，**无 MOSI** | [YetZzzzzz/CMAD](https://github.com/YetZzzzzz/CMAD) | 200 | 排除（见下） |
| FeaDA | IJCNLP-AACL 2025 | MOSI, MOSEI, SIMS | [PowerLittleYin/FeaDA-main](https://github.com/PowerLittleYin/FeaDA-main) | 200 | **卡住**——无 README 无 LICENSE |
| DEAR | Findings of ACL 2026 | MOSI, MOSEI, SIMS | 全文 18 页无链接 | — | `reimpl`，先做完整性自测 |
| MER-CLIP | WACV 2025, pp.6115-6124 | MOSI, MOSEI, IEMOCAP | 全文 10 页无链接 | — | `reimpl`；与第 2 阶段假设最相关 |
| P-RMF | ACL 2025 长文 | 摘要未点名 | 承诺 aoqzhu/P-RMF | **404** | 先写信要代码 |

**ConFEDE 的脚注 1 值得单记**：它的对照基线正是 MMSA 的 `result-stat.md`——而 `#mmsa-all-eleven` 已证那张表**它自己的代码在 9/11 个模型上都够不到**。这不说明 ConFEDE 错了，而是说它报告的相对提升需要在同口径下重新测量。

**分布本身是个结果**：11 篇权威论文里 8 篇有可运行官方代码（1 篇无许可证要卡），2 篇没提供，1 篇承诺了但仓库不存在。

## 排除清单

排除必须带理由，否则下一个人会重新纠结一遍。

- **CMAD**（ICCV 2025）：不报 MOSI，只在 MOSEI 等五个数据集上评测。**我们目前只有 MOSI** → MOSEI 到位后重新评估。这也是 MOSEI 的一条新必要性理由。
- **mlf_dnn / mlmf / mtfn**（MMSA multiTask）：只有 sims/simsv2 配置，需要逐模态标注，**MOSI 不提供**。不是漏做，是数据集不支持。
- **TFR-NET**（MMSA missingTask）：缺失模态鲁棒性，**另一个任务设定**，与当前回归对照不可直接比。属第 3 阶段范围。
- **arXiv 预印本**（QA-MoE、DecAlign、PaSE、GSIFN、AlignMamba-2 等）：未见同行评审版本，按规程不收。
- **MMSA 之外的历史长尾**（ICCN、MAG-XLNet、HyCon 等）：**状态是"未调研"，不是"已排除"**——两者不可混。

## 本次调研推翻的旧结论

- `roadmap.md` 第 3 阶段称缺失/带噪模态鲁棒性"远未饱和"——**已过时**。本次核实的 11 篇里 5 篇做这个（DEAR / P-RMF / CMAD / EBMC / MoLAN），横跨 ACL / CVPR / ICCV，是过去两年的主战场。
- `CLAUDE.md` 称 `reproduce_all.sh` "约 25 分钟"——本机实测 38 组 **5 小时**。

## 待核实

**不许当已知事实引用。**

- DMD 使用谁家的预处理特征（决定其数字能否与我们直接比；README 未写）
- DEAR / MER-CLIP 的论文完整性：能否只凭正文写出全部公式、超参、初始化与调度 → 决定 `reimpl` 还是 `reimpl-partial`
- EBMC / MoLAN / MMA / DPDF-LQ 各自的特征与划分是否与本仓库一致
