from dataclasses import dataclass, field

from torchtitan.config.job_config import JobConfig as BaseJobConfig


@dataclass
class Data:
    img_size: int = 256


@dataclass
class JobConfig(BaseJobConfig):
    data: Data = field(default_factory=Data)
