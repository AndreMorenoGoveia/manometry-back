"""Response shapes of the public API.

They are declared rather than returned as loose dicts so that `/docs` documents
the contract the frontend is written against, and so a field renamed here breaks
loudly instead of quietly turning into `undefined` in the interface.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------- vocabulary

SWALLOW_PATTERNS: dict[str, dict[str, str]] = {
    "failed": {
        "label": "Falha",
        "description": "Sem contração mensurável, com DCI abaixo de 100 mmHg·s·cm.",
    },
    "weak": {
        "label": "Fraca",
        "description": "Contração ineficaz, com DCI entre 100 e 450 mmHg·s·cm.",
    },
    "fragmented": {
        "label": "Fragmentada",
        "description": "Contração vigorosa, porém com quebra maior que 5 cm no contorno de 20 mmHg.",
    },
    "normal": {
        "label": "Normal",
        "description": "Contração efetiva, contínua e com latência preservada.",
    },
    "hypercontractile": {
        "label": "Hipercontrátil",
        "description": "Vigor excessivo, com DCI acima de 8.000 mmHg·s·cm.",
    },
    "premature": {
        "label": "Prematura",
        "description": "Latência distal menor que 4,5 s: a contração chega cedo demais ao esôfago distal.",
    },
    "panesophageal": {
        "label": "Pressurização panesofágica",
        "description": "Coluna pressurizada de forma uniforme entre o esfíncter superior e a JEG.",
    },
}

METRIC_LABELS: dict[str, dict[str, str]] = {
    "dci": {"label": "DCI", "long": "Integral contrátil distal", "unit": "mmHg·s·cm"},
    "irp4": {"label": "IRP 4s", "long": "Pressão de relaxamento integrada", "unit": "mmHg"},
    "distal_latency": {"label": "Latência distal", "long": "Latência distal (DL)", "unit": "s"},
    "front_arrival": {"label": "Chegada da frente", "long": "Chegada da frente contrátil", "unit": "s"},
    "largest_break_cm": {"label": "Maior quebra", "long": "Maior quebra no contorno de 20 mmHg", "unit": "cm"},
    "egj_basal": {"label": "JEG basal", "long": "Pressão basal da junção esofagogástrica", "unit": "mmHg"},
    "peak_distal_pressure": {"label": "Pico distal", "long": "Pico de pressão no esôfago distal", "unit": "mmHg"},
    "pressurization_fraction": {"label": "Corpo pressurizado", "long": "Maior fração do corpo acima de 30 mmHg", "unit": "fração"},
    "pan_time_fraction": {"label": "Tempo panesofágico", "long": "Fração do tempo com o corpo pressurizado", "unit": "fração"},
    "body_length_cm": {"label": "Extensão do corpo", "long": "Extensão do corpo esofágico avaliada", "unit": "cm"},
}

# ------------------------------------------------------------------ payloads


class ClassScore(BaseModel):
    slug: str
    name: str
    short_name: str
    accent: str
    family: str
    probability: float


class Diagnosis(ClassScore):
    tagline: str
    summary: str


class SwallowMetric(BaseModel):
    key: str
    label: str
    long: str
    unit: str
    value: float


class SwallowResult(BaseModel):
    index: int
    used: bool = Field(description="False for swallows beyond the ten slots the network reads")
    pattern: str
    pattern_label: str
    pattern_description: str
    colour_confidence: float = Field(
        description="Fraction of pixels the jet inversion could map back to a pressure"
    )
    metrics: list[SwallowMetric]
    top_class: ClassScore | None = None
    thumbnail: str = ""


class StudyGate(BaseModel):
    key: str
    label: str
    open: bool
    detail: str


class StudySummary(BaseModel):
    swallows: int
    median_irp4: float
    median_dci: float
    max_dci: float
    median_distal_latency: float
    min_distal_latency: float
    max_break_cm: float
    median_egj_basal: float
    failed_fraction: float
    ineffective_fraction: float
    premature_fraction: float
    hypercontractile_fraction: float
    panesophageal_fraction: float
    gates: list[StudyGate]


class PanelAudit(BaseModel):
    total: int
    accepted: int
    off_spec: int
    logo: int
    placeholder: int
    not_localizable: int
    unreadable: int


class ModelCard(BaseModel):
    encoder: str | None = None
    fusion: str | None = None
    seed: int | None = None
    classes: list[str] = []
    max_swallows: int = 10
    device: str = "cpu"
    selected_epoch: int | None = None
    selected_weights: str | None = None
    tta: bool = False
    test_metrics: dict[str, float] | None = None
    val_metrics: dict[str, float] | None = None
    exam_counts: dict[str, int] | None = None
    per_class: list[dict] | None = None
    calibration: dict | None = None


class AnalysisResponse(BaseModel):
    # `model_card` would otherwise collide with pydantic's reserved `model_` prefix.
    model_config = ConfigDict(protected_namespaces=())

    filename: str
    diagnosis: Diagnosis
    ranking: list[ClassScore]
    swallows: list[SwallowResult]
    study: StudySummary
    panels: PanelAudit
    warnings: list[str] = []
    elapsed_ms: float
    model_card: ModelCard


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    version: str
    model_available: bool
    model_loaded: bool
    device: str
    checkpoint: str
