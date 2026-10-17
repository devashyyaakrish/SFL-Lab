from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Operation(StrictModel):
    op: str
    params: dict[str, float] = Field(default_factory=dict)


class CandidateConfig(StrictModel):
    name: str = "Gradient clipping"
    mode: Literal["formula", "algorithm"] = "formula"
    hook: str = "H6"
    expression: str = "clip(G, C_G)"
    parameters: dict[str, float] = Field(default_factory=lambda: {"C_G": 1.0})
    algorithm: list[Operation] = Field(default_factory=list)
    activation_clip: float | None = Field(default=None, gt=0)
    activation_noise: float = Field(default=0, ge=0)
    gradient_clip: float | None = Field(default=None, gt=0)
    gradient_noise: float = Field(default=0, ge=0)
    fedprox_mu: float = Field(default=0, ge=0)
    aggregation: Literal["weighted", "equal"] = "weighted"


class NetworkConfig(StrictModel):
    bandwidth_mbps: float = Field(default=100, gt=0)
    latency_ms: float = Field(default=10, ge=0)
    packet_loss: float = Field(default=0, ge=0, lt=1)
    jitter_ms: float = Field(default=0, ge=0)
    retransmission: bool = True


class EnergyConfig(StrictModel):
    enabled: bool = True
    frequency_ghz: float = Field(default=2, gt=0)
    kappa: float = Field(default=1e-28, gt=0)
    cycles_per_second: float = Field(default=2e9, gt=0)
    tx_power_w: float = Field(default=1.4, ge=0)
    rx_power_w: float = Field(default=0.9, ge=0)


class PrivacyConfig(StrictModel):
    delta: float = Field(default=1e-5, gt=0, lt=1)
    epsilon_max: float = Field(default=10, gt=0)
    warning_fraction: float = Field(default=.8, gt=0, lt=1)
    stop_on_budget: bool = False
    threat_model: Literal["honest_but_curious_server", "external_eavesdropper", "malicious_server", "malicious_client", "parameter_server"] = "honest_but_curious_server"


class AcceptanceConfig(StrictModel):
    max_accuracy_drop_pp: float = Field(default=1, ge=0)
    max_communication_increase_pct: float = Field(default=5, ge=0)
    max_runtime_increase_pct: float = Field(default=10, ge=0)
    required_privacy_improvement_pct: float = Field(default=0, ge=0, le=100)


class EvaluationConfig(StrictModel):
    network: bool = True
    non_iid: bool = True
    attacks: bool = True
    ablation: bool = True
    sweep: bool = True
    scalability: bool = False
    split_profile: bool = False
    attack_steps: int = Field(default=12, ge=1, le=500)
    sweep_parameter: Literal["C_G", "sigma_G", "mu"] = "C_G"
    sweep_values: list[float] = Field(default_factory=lambda: [.5, 1, 2], min_length=1, max_length=20)
    sweep_secondary_parameter: Literal["C_G", "sigma_G", "mu"] | None = None
    sweep_secondary_values: list[float] = Field(default_factory=lambda: [0, .5, 1], min_length=1, max_length=10)
    search: Literal["grid", "random"] = "grid"
    client_counts: list[int] = Field(default_factory=lambda: [5, 10, 20, 50], min_length=1, max_length=8)
    server_counts: list[int] = Field(default_factory=lambda: [1, 2, 4], min_length=1, max_length=8)


class ExperimentConfig(StrictModel):
    name: str = Field(default="Gradient protection study", min_length=1, max_length=120)
    hypothesis: str = Field(default="Gradient clipping preserves validation accuracy within one percentage point while preserving SFL architecture.", min_length=10, max_length=2000)
    mode: Literal["research", "paper_reproduction"] = "research"
    dataset: Literal["digits", "cifar10", "imagefolder"] = "digits"
    dataset_path: str | None = None
    samples: int = Field(default=240, ge=40, le=1000000)
    model: Literal["resnet18", "tiny_cnn"] = "resnet18"
    split: list[int] = Field(default_factory=lambda: [2, 4, 2], min_length=3, max_length=3)
    clients: int = Field(default=3, ge=2, le=100)
    servers: int = Field(default=1, ge=1, le=16)
    rounds: int = Field(default=2, ge=1, le=1000)
    local_epochs: int = Field(default=1, ge=1, le=100)
    batch_size: int = Field(default=16, ge=2, le=512)
    learning_rate: float = Field(default=.01, gt=0, le=10)
    optimizer: Literal["sgd", "adam"] = "sgd"
    seeds: list[int] = Field(default_factory=lambda: [1, 2, 3], min_length=1, max_length=30)
    partition: Literal["iid", "dirichlet", "label_skew", "quantity"] = "iid"
    dirichlet_alpha: float = Field(default=.5, gt=0)
    participation: float = Field(default=1, gt=0, le=1)
    target_loss: float = Field(default=1, gt=0)
    candidate: CandidateConfig = Field(default_factory=CandidateConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    energy: EnergyConfig = Field(default_factory=EnergyConfig)
    acceptance: AcceptanceConfig = Field(default_factory=AcceptanceConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def check_experiment(self):
        if sum(self.split) != 8 or min(self.split) < 1:
            raise ValueError("The front, middle and back must each contain at least one of the eight model blocks; their sizes must sum to 8.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Random seeds must be distinct.")
        if any(s < 0 or s > 2**32 - 1 for s in self.seeds):
            raise ValueError("Seeds must be integers between 0 and 2**32 - 1.")
        if self.clients > int(self.samples * .7):
            raise ValueError("The training set must contain at least one sample per client.")
        if self.dataset == "imagefolder" and not self.dataset_path:
            raise ValueError("A local dataset_path with one subdirectory per class is required for imagefolder/SeaShips.")
        if self.mode == "paper_reproduction" and (self.dataset != "imagefolder" or self.model != "resnet18" or self.split != [2,4,2]):
            raise ValueError("Paper reproduction requires a local SeaShips imagefolder, ResNet18 and the 2-4-2 split. Source paper details must be independently verified.")
        if any(k < 2 or k > 100 for k in self.evaluation.client_counts):
            raise ValueError("Scalability client counts must be 2–100.")
        if any(k < 1 or k > 16 for k in self.evaluation.server_counts):
            raise ValueError("Scalability server counts must be 1–16.")
        if self.evaluation.sweep_secondary_parameter == self.evaluation.sweep_parameter:
            raise ValueError("The two sweep dimensions must use different parameters.")
        return self
