"""`gate_strength=0.0` 的带门控 ALMT，与发布版 ALMT 是否逐比特相同。

这是 QKV 门控消融的前提。消融要成立，两臂之间必须**只有一个变量**——而"门控在
alpha=0 时退化为原版"这句话，读代码是读不出来的：`to_gate` 那两列参数照样存在、
照样被初始化、照样出现在 state_dict 里，只是够不到输出。**够不到**这件事必须测。

做法沿用 `check_almt_equivalence.py`：直接构造两个模型、把 BERT 换成透传桩、权重
逐张量复制、同一批输入跑前向、比最大绝对差。用桩是为了不依赖 BERT 权重，也让这道
检查几秒钟跑完、能进 `check_all.sh`。

**还反着测一次**：`gate_strength=0.5` 必须真的改变输出。少了这一条，"等价"可能是
因为门控压根没接上——那样消融两臂之差恒为零，会以「无效应」的面目通过，而它其实
什么都没测。本仓库有过一个指标全面优于参照却是错的 ALMT 版本
（`investigations.md#almt-better-than-reference`），正是这类数值检查抓出来的。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from msa.models.almt import ALMT  # noqa: E402

BATCH = 4
LEN_TEXT, LEN_AUDIO, LEN_VISION = 50, 375, 500
AUDIO_DIM, VISION_DIM = 5, 20
TOLERANCE = 0.0        # 逐比特：退化是恒等式，不是近似
LIVE_STRENGTH = 0.5


class _StubBert(nn.Module):
    """透传，与 check_almt_equivalence.py 的同名桩一致。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _build(gate_strength: float) -> ALMT:
    torch.manual_seed(0)
    model = ALMT(text_dim=768, audio_dim=AUDIO_DIM, vision_dim=VISION_DIM,
                 gate_strength=gate_strength)
    model.encoder = _StubBert()
    return model.eval()


def main() -> int:
    published = _build(0.0)                      # gate_strength 的默认值就是 0.0
    published_state = {k: v for k, v in published.state_dict().items()
                       if not k.startswith("encoder.")}

    gate_keys = [k for k in published_state if ".to_gate." in k]
    if not gate_keys:
        print("FAIL: 模型里没有 to_gate 参数——门控没被构造出来", file=sys.stderr)
        return 1
    print(f"{len(published_state)} 个参数张量，其中 {len(gate_keys)} 个属于门控")

    batch = {
        "text_bert": torch.randn(BATCH, LEN_TEXT, 768),
        "audio": torch.randn(BATCH, LEN_AUDIO, AUDIO_DIM),
        "vision": torch.randn(BATCH, LEN_VISION, VISION_DIM),
    }

    def run(gate_strength: float) -> torch.Tensor:
        model = _build(gate_strength)
        model.load_state_dict(published_state, strict=False)
        with torch.no_grad():
            return model(batch)["M"]

    baseline = run(0.0)

    # 关键的一步：**扰动门控参数**再跑一次 alpha=0。若 to_gate 真的够不到输出，
    # 改它一个字节也不该动结果。不扰动的话，两次 alpha=0 相同只能说明构造可复现,
    # 说明不了门控被旁路——那是个弱得多的命题。
    perturbed = _build(0.0)
    perturbed.load_state_dict(published_state, strict=False)
    with torch.no_grad():
        for name, parameter in perturbed.named_parameters():
            if ".to_gate." in name:
                parameter.add_(torch.randn_like(parameter) * 10.0)
        off_perturbed = perturbed(batch)["M"]

    live = run(LIVE_STRENGTH)

    off_difference = float((baseline - off_perturbed).abs().max())
    live_difference = float((baseline - live).abs().max())
    print(f"alpha=0，门控参数被大幅扰动   最大绝对差 {off_difference:.3e}  （须为 0）")
    print(f"alpha={LIVE_STRENGTH}                        "
          f"最大绝对差 {live_difference:.3e}  （须非 0）")

    if off_difference > TOLERANCE:
        print("FAIL: alpha=0 时门控参数仍能影响输出，退化不成立", file=sys.stderr)
        return 1
    if live_difference == 0.0:
        print(f"FAIL: alpha={LIVE_STRENGTH} 没有改变输出——门控没接上，"
              f"消融会测出一个假的零", file=sys.stderr)
        return 1
    print("EQUIVALENT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
