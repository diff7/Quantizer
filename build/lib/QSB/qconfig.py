from typing import List
from dataclasses import dataclass, field


@dataclass
class QConfig:
    bits: List[int] = field(default_factory=lambda: [2, 4, 8])
    act_quantizer: str = "LSQ"
    weight_quantizer: str = "LSQ"
    noise_search: bool = False


if __name__ == "__main__":
    qconfig = QConfig(bits=[2, 8])
    print(qconfig)

    qconfig = QConfig(noise_search=True)
    print(qconfig)

    qconfig = QConfig(act_quantizer="HWGQ", weight_quantizer="HWGQ")
    print(qconfig)
