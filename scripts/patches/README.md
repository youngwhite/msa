# 对论文原作者代码的源码级改动

**这里的东西与 `.mmsa-reference/compat/` 是两件不同的事，不要混。**

- `compat/`：torch / transformers 的 **API 变更**兼容层，论文源码逐字节不改。目的是
  让"我们跑的是发布版"这句话成立。
- 这里：**故意改变实验的**源码改动，每一条都是一个有名字的对照臂。以 `.patch` 形式
  入库，diff 可审；as-released 的数字与打了补丁的数字**绝不混在一张无标注的表里**。

## `hscl_clean_selection.patch`

HSCL 发布版的模型选择读了测试集——上报的那一轮必须同时"验证变好"且"测试变好"
（原因、以及能看到它发生的那次运行，见 `docs/investigations.md#hscl-test-leak`）。

这个补丁把内层那个测试集门槛去掉，只按验证集选。**一行。**

有意保留的两点：

- `best_mae = test_loss` 留着不删。它不再参与选择，只是记账；删掉会让 diff 变大，
  而 diff 越小越好审。
- `patience` 的重置在外层 `if val_loss < best_valid` 里，补丁没碰它，**所以早停行为
  与原版逐轮一致**。唯一变的是"哪一轮被上报"。这是把这条改动当成可解释对照臂的前提：
  两臂的训练轨迹相同，只有选择规则不同。

怎么用（HSCL 那份是 git clone，跑完 `git checkout` 就能回到发布版）：

```bash
cd .mmsa-reference/papers/HSCL
git apply ../../../scripts/patches/hscl_clean_selection.patch
# ...跑 clean 臂...
git checkout src/solver.py          # 回到 as-released
```

`scripts/run_hscl_arms.sh` 把这套流程连打补丁、跑两臂、还原一起做了，别手工来。
