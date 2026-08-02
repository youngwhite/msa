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

### 方法：逐卷枚举，不是关键词搜索

**关键词搜索无法保证不漏**——换一种措辞就漏一篇，而且漏了不会有任何征兆。可靠做法是把每一卷的**全部标题**取下来再筛：

```bash
curl -sL "https://aclanthology.org/volumes/<卷号>/" -o vol.html   # 每卷一个索引页
# 抽全部标题 → 关键词筛 → 分档
```

**提取必须处理嵌套标签。** ACL 的标题里缩写被包在 `<span class=acl-fixed-case>DEAR</span>` 中，用 `[^<]+` 取标题会在第一个内嵌标签处截断，**凡标题含缩写的论文全部漏掉**。首版就踩了这个坑：筛出 15 篇、看起来完全正常，实际漏了 DEAR 与 MoLAN——**若非恰好知道这两篇存在，这个损耗永远不会暴露**。正确做法是非贪婪匹配到 `</a>` 后再剥标签。

**自检**：每次扫描都要报"枚举总数"。总数明显偏低即提取器坏了，而不是那一年没论文。

### 2026-08-02 的扫描结果

12 卷（ACL/EMNLP/NAACL/Findings 的 2024–2026），**枚举 15834 篇，关键词命中 95 篇**，分档后 **14 篇与本项目同任务**（多模态情感回归）。完整结果入库为 [`sweep_acl_2024_2026.tsv`](sweep_acl_2024_2026.tsv)，含全部 95 篇及其分档，可复查。

分档规则：`IN_SCOPE` = 多模态情感（MOSI/MOSEI 系）；`other_task` = 面向方面的情感分析（ABSA，纯文本）与对话情绪识别（ERC，IEMOCAP/MELD 系）——**同名不同任务，不可混入同一张对照表**。

**尚未枚举**：CVF（CVPR/ICCV/WACV）、AAAI、ACM MM、IJCNLP-AACL。它们各有自己的索引页格式，需分别写提取器。**在完成之前，本台账的覆盖是"ACL 系已枚举、其余靠关键词"，不得声称已扫全。**

## 实现出处

每个进入对照表的模型必须带这一列。它决定结论能写成什么样。

| 标记 | 含义 | 可以怎么写 |
|---|---|---|
| `ported` | 移植自官方或参照实现 | "复现该方法" |
| `reimpl+equiv` | 自己实现，且通过权重复制的数值等价测试 | "复现该方法" |
| `reimpl` | 自己实现，无参照可验证，消融方向一致 | "重新实现；与报告值相差 X SE，**实现忠实性未经验证**" |
| `reimpl-partial` | 论文细节不全，含我们的补充设定 | "含我们的补充设定，**不作为该方法的成绩**" |

两条铁律：**比参照『更好』先当 bug 信号排查**（`#almt-better-than-reference` 的教训）；**无参照时**用消融方向一致性 + 参数量对表 + 逐公式退化测试替代等价测试——其中消融方向最有力，因为它是内部对照，不依赖绝对标定。

## 台账 A0：原始论文出处与作者代码（核实于 2026-08-02）

仓库各处（模型 docstring、storyline）声称的引用，逐条对权威出处核实。**代码列一律取自论文正文**，不是搜索猜测——猜出来的 URL 即使返回 200 也不能证明归属。

| 模型 | 论文与出处（已核） | 作者代码（论文正文所载） | 实测 |
|---|---|---|---|
| tfn | Tensor Fusion Network for Multimodal Sentiment Analysis；Zadeh, Chen, Poria, Cambria, Morency；[EMNLP 2017](https://aclanthology.org/D17-1115/) pp.1103-1114 | **正文无代码链接** | — |
| lmf | Efficient Low-rank Multimodal Fusion With Modality-Specific Factors；Liu, Shen, Lakshminarasimhan, Liang, Bagher Zadeh, Morency；[ACL 2018](https://aclanthology.org/P18-1209/) pp.2247-2256 | [Justin1904/Low-rank-Multimodal-Fusion](https://github.com/Justin1904/Low-rank-Multimodal-Fusion) | 200 |
| mfn | Memory Fusion Network for Multi-view Sequential Learning；Zadeh, Liang, Mazumder, Poria, Cambria, Morency；[AAAI-18](https://ojs.aaai.org/index.php/AAAI/article/view/12021) | 正文给 `A2Zadeh/MFN` | **404，已失效** |
| graph_mfn | Multimodal Language Analysis in the Wild: **CMU-MOSEI Dataset** and Interpretable Dynamic Fusion Graph；Bagher Zadeh, Liang, Poria, Cambria, Morency；[ACL 2018](https://aclanthology.org/P18-1208/) pp.2236-2246 | 正文只给数据 SDK（`A2Zadeh/CMU-MultimodalDataSDK`，**404**），无模型代码 | — |
| mult | Multimodal Transformer for Unaligned Multimodal Language Sequences；Tsai, Bai, Liang, Kolter, Morency, Salakhutdinov；[ACL 2019](https://aclanthology.org/P19-1656/) pp.6558-6569 | [yaohungt/Multimodal-Transformer](https://github.com/yaohungt/Multimodal-Transformer) | 200 |
| misa | MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis；Hazarika, Zimmermann, Poria；**ACM MM '20** pp.1122-1131（[arXiv:2005.03545](https://arxiv.org/pdf/2005.03545)） | [declare-lab/MISA](https://github.com/declare-lab/MISA) | 200 |
| self_mm | Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis；Yu, Xu, Yuan, Wu；[AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17289) vol.35 pp.10790-10797 | [thuiar/Self-MM](https://github.com/thuiar/Self-MM) | 200 |
| bert_mag | Integrating Multimodal Information in Large Pretrained Transformers；Rahman, Hasan, Lee, Bagher Zadeh, Mao, Morency, Hoque；[ACL 2020](https://aclanthology.org/2020.acl-main.214/) pp.2359-2369 | [WasifurRahman/BERT_multimodal_transformer](https://github.com/WasifurRahman/BERT_multimodal_transformer) | 200 |
| mctn | Found in Translation: Learning Robust Joint Representations by Cyclic Translations Between Modalities；Pham, Liang, Manzini, Morency, Póczos；**AAAI 2019** vol.33 pp.6892-6899 | [hainow/MCTN](https://github.com/hainow/MCTN) | 200 |
| mfm | Learning Factorized Multimodal Representations；Tsai, Liang, Zadeh, Morency, Salakhutdinov；**ICLR 2019** | [pliang279/factorized](https://github.com/pliang279/factorized/) | 200 |
| mmim | Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis；Wei Han, Hui Chen, Soujanya Poria；[EMNLP 2021](https://aclanthology.org/2021.emnlp-main.723/) pp.9180-9192 | [declare-lab/Multimodal-Infomax](https://github.com/declare-lab/Multimodal-Infomax)（论文正文） | 200 |
| almt | Learning Language-guided Adaptive Hyper-modality Representation for Multimodal Sentiment Analysis；Haoyu Zhang, Yu Wang, Guanghao Yin, Kejun Liu, Yuanyuan Liu, Tianshu Yu；[EMNLP 2023](https://aclanthology.org/2023.emnlp-main.49/) pp.756-767 | **论文正文与 ACL 页面均无代码链接**；`Haoyu-ha/ALMT`（200）出自 **MMSA 的 docstring**，非论文 | 见左 |
| cenet | Cross-modal enhancement network for multimodal sentiment analysis；Di Wang, Shuai Liu, Quan Wang, Yumin Tian, Lihuo He, Xinbo Gao；**IEEE TMM 2022** pp.4909-4921 | **未核**（IEEE 全文需订阅） | — |
| tetfn | TETFN: A text enhanced transformer fusion network for multimodal sentiment analysis；Wang 等；**Pattern Recognition 2023**, vol.136, 109259 | **未核**（ScienceDirect 返回 403） | — |

**四条值得单记的事实：**

1. **TFN 原论文没有放代码。** 领域内普遍使用的 `Justin1904/TensorFusionNetworks` 出现在 **MISA 论文的引用里**（作为其复现 TFN 所用的实现），不是 TFN 作者发布的。所以 TFN 这一支的"参照实现"从一开始就是第三方的。
2. **MFN 与 Graph-MFN 的作者代码链接都已失效**（`A2Zadeh/MFN`、`A2Zadeh/CMU-MultimodalDataSDK` 均 404）。MFN 目前只能以 MMSA 为参照，**这解释了为什么 `#mfn` 那轮"对原作者实现核对"只能止于无发现**。数据 SDK 已迁至 [CMU-MultiComp-Lab/CMU-MultimodalSDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)（200）。
3. **Graph-MFN 的原论文同时是 CMU-MOSEI 的发布论文。** 路线图里"主数据集换 MOSEI"要引的，和 Graph-MFN 是同一篇。
4. **MISA 论文引用了 `pliang279/MFN` 与 `pliang279/factorized`** 作为其基线来源——后者正是 MFM 的官方实现，两条线在这里交汇。
5. **ALMT 的论文里唯一的 GitHub 链接是 MMSA 的 `result-stat.md`**——和 ConFEDE 一样，**它的对照基线就是那张我们已证"MMSA 自己的代码在 9/11 个模型上都够不到"的表**。已发现两篇顶会论文以该表为基线，这不再是孤例。
6. **两篇期刊论文（CENET / TETFN）的全文拿不到**：ScienceDirect 返回 403，IEEE 需订阅。出处以出版商元数据为准，**代码链接一栏记"未核"而不是"无"**——这两者不能混。

### 分档汇总（12 个已核模型）

- **原作者代码可用**（论文正文所载且实测 200）：lmf、mult、misa、self_mm、bert_mag、mmim、mctn、mfm —— **8 个**
- **原论文未给代码**：tfn、almt —— **2 个**
- **作者链接已失效**：mfn、graph_mfn —— **2 个**
- **全文不可获取，未核**：cenet、tetfn —— **2 个**

也就是说，**我们对照表里 12 个已核模型中，有 6 个拿不到可验证归属的作者实现**。这与近三年新论文的比例（11 篇中 8 篇有代码）方向一致：**领域整体的可复现基础设施，比论文数量增长得慢。**

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

- **cenet / tetfn 的论文全文**：两家出版商都挡住了（403 / 需订阅）。需要机构访问权限才能核代码链接与实验设定。**在拿到之前，这两条的代码栏保持"未核"。**
- **tetfn 的完整作者名单**：只确认到 "Wang 等"，出版商元数据未给全。
- ef_lstm / lf_dnn 在 MMSA 中是通用基线，**是否有可指认的原始论文尚未确认**——若无，应在故事线里明说"出自 MMSA 的基线实现"，不要虚构出处
- DMD 使用谁家的预处理特征（决定其数字能否与我们直接比；README 未写）
- DEAR / MER-CLIP 的论文完整性：能否只凭正文写出全部公式、超参、初始化与调度 → 决定 `reimpl` 还是 `reimpl-partial`
- EBMC / MoLAN / MMA / DPDF-LQ 各自的特征与划分是否与本仓库一致
