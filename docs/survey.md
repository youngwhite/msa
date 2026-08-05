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

### 2026-08-02 的扫描结果（全域，2017–2026）

固化为 **`scripts/sweep_venues.py`**（`--offline` 用缓存重跑，`--from-year` 控制年份）。结果入库为 [`sweep_venues.tsv`](sweep_venues.tsv)。

**DBLP 为主源**，一个解析器覆盖 ACL / EMNLP / NAACL / CVPR / ICCV / ECCV / WACV / AAAI / IJCAI / ACM MM / ICML / ICLR / NeurIPS。走各家自己的站点则每家都要写一个提取器（AAAI 要抓 97 个 OJS 分册页、CVF 要处理 gzip 与 `?day=all`），而 **ACM 的 DL 对脚本一律 403**——正是 DBLP 让 ACM MM 从"抓不到"变成免费可得。

**但 DBLP 有滞后，不能作为唯一源**：CVPR 2026 已在 CVF 发布而 DBLP 尚未收录，其 8 篇同任务论文会在纯 DBLP 扫描中消失。故 CVF 保留为直连回退。**权限或收录障碍是换路径的理由，不是放弃覆盖的理由。**

| | 数量 |
|---|---|
| DBLP 枚举 | **114,689** |
| CVF 直连补充 | 7,600 |
| 命中（宽词） | 598 |
| **同任务（2017–2026）** | **101** |

逐年：2017: 1 | 2018: 2 | 2019: 6 | 2020: 4 | 2021: 6 | 2022: 11 | 2023: 15 | 2024: 13 | 2025: 20 | 2026: 23

**2017–2021 五年合计 19 篇，2025–2026 两年 43 篇。** 领域产出在 2023 年后翻了两三倍。

### 卷号必须发现，不能猜

首版按格式猜 key（`emnlp{y}`、`acl{y}-1`），**59 个 venue-year 落空**——包括 emnlp2019 至 emnlp2022、acl2020 这些主力年份，因为主会卷在不同年份分别键为 `emnlp2019-1` 与 `emnlp2023`。**猜错的 key 404 之后，长得和"那年没有论文"一模一样**：TFN（EMNLP 2017）、Graph-MFN（ACL 2018）、MISA（MM 2020）这些奠基之作当时全部缺席，而扫描结果看不出任何异常。

改为**从每个会议的 DBLP 索引页读出真实卷号**后，缺失从 59 降到 3，命中从 92 升到 101。那 3 个空卷（`emnlp2020-t`、`emnlp2021-1`、`emnlp2021-d`）由脚本单独列出提醒复查，不静默吞掉。

### 这套方法的三个坑，都是静默失败

1. **嵌套标签截断标题**：ACL 把缩写包在 `<span class=acl-fixed-case>` 里，用 `[^<]+` 取标题会在第一个内嵌标签处断，**凡标题含缩写的论文全部漏掉**。首版筛出 15 篇、看起来正常，实际丢了 DEAR 与 MoLAN。
2. **词边界**：`mosi` 不加 `\b` 会匹配到 **MoSiC**（ICCV 2025，自监督运动轨迹，与本任务无关）。
3. **标题定不了任务归属**：D2R（EMNLP 2024）与 Beyond Static Alignment（ACL 2026）标题都写 "Multimodal Sentiment"，正文用的却是 MVSA/HFM——**图文社交媒体情感**。**枚举解决"漏"，不解决"错收"**，数据集提取是必需的第二步。

**自检**：每次扫描都报枚举总数，脚本在总数低于 500 时会自己喊。总数掉了是提取器坏了，不是那年没论文。

### 仍未枚举

- **ACM MM**：`dl.acm.org` 对脚本请求一律 **403**（proceedings 与 conference 两个入口都试过）。**需要人工或机构访问**——与两篇付费期刊同类，不是"没有论文"。
- **IJCNLP-AACL**：在 ACL Anthology 内，可按同法补。
- **历史长尾**（ICCN、MAG-XLNet、HyCon、UniMSE 等）：状态仍是**"未调研"**，不是"已排除"。

## 实现出处

每个进入对照表的模型必须带这一列。它决定结论能写成什么样。

| 标记 | 含义 | 可以怎么写 |
|---|---|---|
| `ported` | 移植自官方或参照实现 | "复现该方法" |
| `reimpl+equiv` | 自己实现，且通过权重复制的数值等价测试 | "复现该方法" |
| `reimpl` | 自己实现，无参照可验证，消融方向一致 | "重新实现；与报告值相差 X SE，**实现忠实性未经验证**" |
| `reimpl-partial` | 论文细节不全，含我们的补充设定 | "含我们的补充设定，**不作为该方法的成绩**" |

两条铁律：**比参照『更好』先当 bug 信号排查**（`#almt-better-than-reference` 的教训）；**无参照时**用消融方向一致性 + 参数量对表 + 逐公式退化测试替代等价测试——其中消融方向最有力，因为它是内部对照，不依赖绝对标定。

### 参照优先级（2026-08-02 定）

论文没给代码不等于没有参照。按下列顺序取，**取到哪一级就在文档里注明哪一级**：

1. **作者实现**（论文正文所载，或作者名单里任一人的仓库——MFN 的可用参照就在合著者 `pliang279` 名下，论文印的那个已 404）
2. **权威第三方开源实现**，本项目即 MMSA（MIT，THUIAR）。TFN / ALMT / TETFN 三个原论文未发代码的模型走的都是这一级
3. **都没有 → 自己实现**，用消融方向一致性等替代检验，标 `reimpl`

**取第 2 级时必须同时声明"复现的是谁的版本"。** 这不是形式主义：MMSA 与论文的偏离已经抓到多起——MulT 的位置编码被默认关闭（原作者无条件启用）、CENET 的核心量化机制被换成原始连续特征、MFM 的 `learning_rate` 配置从未生效、MCTN 的 teacher forcing 是空操作。**照抄 MMSA 就会忠实复现这些偏离**，所以 storyline 里 CENET 那句"我们复现的是 MMSA 的 CENET，不是论文的 CENET"是这一级的标准写法，其余模型同理。

第 2 级还有一层风险要记住：**MMSA 的公开表已被本项目证明其自身代码在 9/11 个模型上都达不到**。它作为**可运行参照**是合格的（能跑、带方差、同数据），作为**目标值**不合格。两种用途不可混。

> **清单已移到 [`survey_checklist.md`](survey_checklist.md)**，由
> `python scripts/survey_checklist.py` 生成，状态只在 `docs/papers.tsv` 里声明。
> 下面的台账保留为 2026-08-02 的核实记录（含每条的出处考据），**不再手工更新状态**——
> 同一件事有多个出处就必然对不上，storyline 的进度表已经这么漂过一次。
>
> 生成器还会做一致性检查：声明「已复现」却没有对应的 `outputs/` 组、或枚举为 `IN_SCOPE`
> 却从未进 `papers.tsv`，都会报出来。**首次运行即报出约 70 篇枚举后从未分诊的论文**——
> 这正是它存在的理由。

## 台账 A0：原始论文出处与作者代码（核实于 2026-08-02）

仓库各处（模型 docstring、storyline）声称的引用，逐条对权威出处核实。**代码列一律取自论文正文**，不是搜索猜测——猜出来的 URL 即使返回 200 也不能证明归属。

| 模型 | 论文与出处（已核） | 作者代码（论文正文所载） | 实测 |
|---|---|---|---|
| tfn | Tensor Fusion Network for Multimodal Sentiment Analysis；Zadeh, Chen, Poria, Cambria, Morency；[EMNLP 2017](https://aclanthology.org/D17-1115/) pp.1103-1114 | **正文无代码链接** | — |
| lmf | Efficient Low-rank Multimodal Fusion With Modality-Specific Factors；Liu, Shen, Lakshminarasimhan, Liang, Bagher Zadeh, Morency；[ACL 2018](https://aclanthology.org/P18-1209/) pp.2247-2256 | [Justin1904/Low-rank-Multimodal-Fusion](https://github.com/Justin1904/Low-rank-Multimodal-Fusion) | 200 |
| mfn | Memory Fusion Network for Multi-view Sequential Learning；Zadeh, Liang, Mazumder, Poria, Cambria, Morency；[AAAI-18](https://ojs.aaai.org/index.php/AAAI/article/view/12021) | 正文给 `A2Zadeh/MFN`（**404**）；实际可用的是合著者仓库 [pliang279/MFN](https://github.com/pliang279/MFN) | 200 |
| graph_mfn | Multimodal Language Analysis in the Wild: **CMU-MOSEI Dataset** and Interpretable Dynamic Fusion Graph；Bagher Zadeh, Liang, Poria, Cambria, Morency；[ACL 2018](https://aclanthology.org/P18-1208/) pp.2236-2246 | 正文只给数据 SDK（`A2Zadeh/CMU-MultimodalDataSDK`，**404**），无模型代码 | — |
| mult | Multimodal Transformer for Unaligned Multimodal Language Sequences；Tsai, Bai, Liang, Kolter, Morency, Salakhutdinov；[ACL 2019](https://aclanthology.org/P19-1656/) pp.6558-6569 | [yaohungt/Multimodal-Transformer](https://github.com/yaohungt/Multimodal-Transformer) | 200 |
| misa | MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis；Hazarika, Zimmermann, Poria；**ACM MM '20** pp.1122-1131（[arXiv:2005.03545](https://arxiv.org/pdf/2005.03545)） | [declare-lab/MISA](https://github.com/declare-lab/MISA) | 200 |
| self_mm | Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis；Yu, Xu, Yuan, Wu；[AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17289) vol.35 pp.10790-10797 | [thuiar/Self-MM](https://github.com/thuiar/Self-MM) | 200 |
| bert_mag | Integrating Multimodal Information in Large Pretrained Transformers；Rahman, Hasan, Lee, Bagher Zadeh, Mao, Morency, Hoque；[ACL 2020](https://aclanthology.org/2020.acl-main.214/) pp.2359-2369 | [WasifurRahman/BERT_multimodal_transformer](https://github.com/WasifurRahman/BERT_multimodal_transformer) | 200 |
| mctn | Found in Translation: Learning Robust Joint Representations by Cyclic Translations Between Modalities；Pham, Liang, Manzini, Morency, Póczos；**AAAI 2019** vol.33 pp.6892-6899 | [hainow/MCTN](https://github.com/hainow/MCTN) | 200 |
| mfm | Learning Factorized Multimodal Representations；Tsai, Liang, Zadeh, Morency, Salakhutdinov；**ICLR 2019** | [pliang279/factorized](https://github.com/pliang279/factorized/) | 200 |
| mmim | Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis；Wei Han, Hui Chen, Soujanya Poria；[EMNLP 2021](https://aclanthology.org/2021.emnlp-main.723/) pp.9180-9192 | [declare-lab/Multimodal-Infomax](https://github.com/declare-lab/Multimodal-Infomax)（论文正文） | 200 |
| almt | Learning Language-guided Adaptive Hyper-modality Representation for Multimodal Sentiment Analysis；Haoyu Zhang, Yu Wang, Guanghao Yin, Kejun Liu, Yuanyuan Liu, Tianshu Yu；[EMNLP 2023](https://aclanthology.org/2023.emnlp-main.49/) pp.756-767 | **论文正文与 ACL 页面均无代码链接**；`Haoyu-ha/ALMT`（200）出自 **MMSA 的 docstring**，非论文 | 见左 |
| cenet | Cross-Modal Enhancement Network for Multimodal Sentiment Analysis；Di Wang, Shuai Liu, Quan Wang, Yumin Tian, Lihuo He, Xinbo Gao；**IEEE TMM vol.25, 2023**, p.4909 起（**不是 2022**，见下） | [Say2L/**CENet**](https://github.com/Say2L/CENet)（论文正文；注意拼写是 CENet） | 200 |
| tetfn | TETFN: A text enhanced transformer fusion network for multimodal sentiment analysis；**Di Wang, Xutong Guo, Yumin Tian, Jinhui Liu, LiHuo He, Xuemei Luo**（西安电子科技大学）；**Pattern Recognition 136 (2023) 109259** | **正文无代码链接** | — |

**四条值得单记的事实：**

1. **TFN 原论文没有放代码。** 领域内普遍使用的 `Justin1904/TensorFusionNetworks` 出现在 **MISA 论文的引用里**（作为其复现 TFN 所用的实现），不是 TFN 作者发布的。所以 TFN 这一支的"参照实现"从一开始就是第三方的。
2. **论文里印的作者链接失效，不等于没有作者实现。** `A2Zadeh/MFN` 与 `A2Zadeh/CMU-MultimodalDataSDK` 均 404，但 MFN 的逐行核对（`investigations.md#mfn`，2026-07-31）用的是 **[pliang279/MFN](https://github.com/pliang279/MFN)（200）——Paul Pu Liang 是 MFN 论文的合著者**，那是一份合著者实现。**本台账初版据此把 MFN 记成"无作者实现可核"，是错的，已订正。** 教训：论文正文的链接是起点不是终点，作者名单里每个人的仓库都该查一遍。数据 SDK 已迁至 [CMU-MultiComp-Lab/CMU-MultimodalSDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)（200）。
3. **Graph-MFN 的原论文同时是 CMU-MOSEI 的发布论文。** 路线图里"主数据集换 MOSEI"要引的，和 Graph-MFN 是同一篇。
4. **MISA 论文引用了 `pliang279/MFN` 与 `pliang279/factorized`** 作为其基线来源——后者正是 MFM 的官方实现，两条线在这里交汇。
5. **ALMT 的论文里唯一的 GitHub 链接是 MMSA 的 `result-stat.md`**——和 ConFEDE 一样，**它的对照基线就是那张我们已证"MMSA 自己的代码在 9/11 个模型上都够不到"的表**。已发现两篇顶会论文以该表为基线，这不再是孤例。
6. **CENET 的年份此前记错了。** 仓库 docstring 与检索摘要都写 "TMM 2022"，但正式版页眉是 **IEEE TRANSACTIONS ON MULTIMEDIA, VOL. 25, 2023, p.4909**——2022 是 early access 的年份，**卷期年是 2023**。这正是当初标记"期刊出处比会议更易记错"的那一类错误，也是**只有拿到全文才能发现**的一类：出版商元数据与检索结果都在传播 2022。
7. **TETFN 的论文正文没有代码链接**，且第一作者与 CENET 同为 Di Wang（西安电子科技大学）。两篇同组工作，一篇给了代码一篇没给。
8. 两篇期刊全文由用户下载提供（ScienceDirect 403 / IEEE 需订阅，脚本取不到）。**这条留作规程的一部分：付费墙论文需要人工介入，不是"核不到"就记"无"。**

### 分档汇总（14 个已核模型，全部已核）

- **作者代码可用**：lmf、mult、misa、self_mm、bert_mag、mmim、mctn、mfm、cenet（论文正文所载）+ **mfn**（合著者仓库，论文链接已失效）—— **10 个**
- **原论文未给代码**：tfn、almt、tetfn —— **3 个**
- **仅数据 SDK，无模型代码**：graph_mfn —— **1 个**

也就是说，**14 个已核模型中有 4 个拿不到可验证归属的作者实现**。这与近三年新论文的比例（11 篇中 8 篇有代码）方向一致：**领域整体的可复现基础设施，比论文数量增长得慢。**

## 台账 A：已在对照表中

| 模型 | 参照来源 | 实现出处 | 备注 |
|---|---|---|---|
| ef_lstm / lf_dnn / tfn / lmf / mfn / graph_mfn / mult / misa / self_mm / cenet / tetfn | MMSA 公开表 + 我们跑它的代码 | `ported` | 11 个有表内参照值 |
| bert_mag | 我们跑 MMSA 的代码 | `reimpl` | 直接对着 transformers 写，未转抄；原作者实现 `WasifurRahman/BERT_multimodal_transformer` |
| mmim | 我们跑 MMSA 的代码 | `ported` | 已对照原作者 `declare-lab/Multimodal-Infomax`，MMILB 与 CPC 逐行一致 |
| almt | 我们跑 MMSA 的代码 | `reimpl+equiv` | 权重复制等价测试，最大绝对差 `0.000e+00` |
| text_bert | 无（对照组） | 自建 | 故事线的关键对照 |
| lf_lstm | 无（自建基线） | 自建 | 方差标尺（seed 间 MAE sd 0.039） |
| **dpdf_lq** | 我们跑**作者代码** 10 seed（`docs/author_code_runs_mosi.json`），按验证集选轮 | `reimpl+equiv` | 权重复制等价测试 `0.000e+00`；结构与协议均取自作者实现，超参取自论文 Table 3 |
| **dlf** | 我们跑**作者代码** 10 seed（`docs/author_code_runs_mosi.json`），按验证集选轮 | `reimpl+equiv` | 权重复制等价测试 `2.384e-07`，**五个监督头全部比较**；结构、协议、超参均取自作者实现。论文未写的两处（四条通路构造不接线、语言头权重 3）只在代码里 |
| **dmd** | 我们跑**作者代码 + 已验证的兼容性重建** 10 seed（`docs/author_code_runs_mosi.json`），按验证集选轮 | `reimpl+equiv` | 权重复制等价测试 `9.537e-07`，**14 个输出全部比较**；release 在当前 PyTorch 上无法启动，参照因此低一档，补丁分档见 `investigations.md#dmd-reference-patches` |
| **confede** | 我们跑**作者代码** 10 seed（`docs/author_code_runs_mosi.json`），其自身即按验证集 MAE 选轮、测试集只碰一次 | `reimpl+equiv` | 权重复制等价测试 `0.000e+00`，**两个编码器 + 六个视图全部比对**；三处补丁均为管道问题（不进梯度/选轮），故参照不降级 |

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

## 台账 D：逐卷枚举补核（2026-08-02）

ACL 系逐卷枚举筛出的同任务论文，按核实规矩逐条核（全文解析 + 仓库实地请求）。

| 方法 | 出处 | 数据集（正文实测） | 官方代码 | 实测 |
|---|---|---|---|---|
| **KuDA** Knowledge-Guided Dynamic Modality Attention Fusion | Findings of EMNLP 2024 | MOSI, MOSEI, CH-SIMS | [MKMaS-GUET/KuDA](https://github.com/MKMaS-GUET/KuDA) | 200 |
| **TF-Mamba** Text-enhanced Fusion Mamba, Missing Modalities | Findings of EMNLP 2025 | MOSI, MOSEI, CH-SIMS | [codemous/TF-Mamba](https://github.com/codemous/TF-Mamba) | 200 |
| **MulCoT-RD** Resource-Limited Joint MSA Reasoning（CoT 蒸馏） | Findings of ACL 2026 | 待核 | [123sghn/MulCoT-RD](https://github.com/123sghn/MulCoT-RD) | 200 |
| **LFD-RT** Cross-lingual MSA, Language Family Disentanglement | Findings of ACL 2025 | MOSEI, SIMS, MELD（**无 MOSI**） | [ShuoyuGuan/LFD-RT](https://github.com/ShuoyuGuan/LFD-RT) | 200 |
| **QA-MoE** Quality-Aware Mixture of Experts, Robust MSA | ACL 2026 长文 | CMU-MOSI, CMU-MOSEI, IEMOCAP | **正文无代码链接** | — |
| **Fast Retrieval and Slow Reasoning** Explainable MSA | Findings of ACL 2026 | MOSI, MOSEI, CH-SIMS | **正文无代码链接** | — |
| **Uncertainty-Calibrated Elastic Alignment**, Missing Modalities | Findings of ACL 2026 | MOSI, MOSEI, CH-SIMS | **正文无代码链接** | — |

### 枚举筛出但**任务不同**，排除

- **D2R**（EMNLP 2024）与 **Beyond Static Alignment**（ACL 2026 长文）：标题都写 "Multimodal Sentiment"，但正文用的是 **MVSA-Single / MVSA-Multiple / HFM**——**图文社交媒体情感**，不是视频三模态情感回归。**同名不同任务，不可混入同一张对照表。**
  - 这两条是标题关键词筛的固有漏洞：任务归属只能由正文的数据集判定，**标题不可信**。逐卷枚举解决"漏"，不解决"错收"。
- **Emosical**（Findings of EMNLP 2024）：情绪标注的音乐剧数据集，非方法论文。

### 覆盖现状

近三年同任务、且**有可运行官方代码**的合计 **12 篇**（台账 C 的 8 篇 + 本节 4 篇）；**无代码需自行实现**的 6 篇（DEAR、MER-CLIP、P-RMF、QA-MoE、Fast Retrieval、Uncertainty-Calibrated）。

**仍未枚举**：CVF、AAAI、ACM MM、IJCNLP-AACL。历史长尾（ICCN、MAG-XLNet、HyCon、UniMSE 等）状态仍是**"未调研"**。

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

- ef_lstm / lf_dnn 在 MMSA 中是通用基线，**是否有可指认的原始论文尚未确认**——若无，应在故事线里明说"出自 MMSA 的基线实现"，不要虚构出处
- DMD 使用谁家的预处理特征（决定其数字能否与我们直接比；README 未写）
- DEAR / MER-CLIP 的论文完整性：能否只凭正文写出全部公式、超参、初始化与调度 → 决定 `reimpl` 还是 `reimpl-partial`
- EBMC / MoLAN / MMA / DPDF-LQ 各自的特征与划分是否与本仓库一致
