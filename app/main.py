"""HTTP surface: upload a laudo, get a Chicago v4.0 study-level diagnosis.

The interesting route is `POST /api/analyze`.  Everything else exists so the
frontend can render the result without hard-coding clinical facts of its own:
`/api/diseases` is the reference text, `/api/model` is the provenance and the
held-out performance of the checkpoint answering, and
`/api/diseases/{slug}/reference.png` is a real example topography for each class.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .config import LAZY_MODEL, SETTINGS
from .diseases import DISEASES, get as get_disease
from .inference import CLASSIFIER, ModelUnavailable
from .references import reference_png
from .report import ReportError, read_swallows
from .sample import build_docx_bytes, laudo_from_exam
from .schemas import (
    METRIC_LABELS,
    SWALLOW_PATTERNS,
    AnalysisResponse,
    ClassScore,
    Diagnosis,
    HealthResponse,
    ModelCard,
    PanelAudit,
    StudyGate,
    StudySummary,
    SwallowMetric,
    SwallowResult,
)

LOGGER = logging.getLogger("manometry")

ACCEPTED_SUFFIXES = (".docx", ".rtf")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not LAZY_MODEL and CLASSIFIER.available:
        # Paying the ~10 s checkpoint load at boot rather than on the first
        # upload is what a container behind a readiness probe wants.
        CLASSIFIER.load()
        LOGGER.info("checkpoint carregado em %s", CLASSIFIER.device)
    yield


app = FastAPI(
    title="Manometry API",
    version=__version__,
    summary="Classificação Chicago v4.0 de manometria esofágica de alta resolução a partir do laudo",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS.allowed_origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------- helpers


def _class_score(slug: str, probability: float) -> ClassScore:
    disease = get_disease(slug)
    return ClassScore(
        slug=slug,
        name=disease.name,
        short_name=disease.short_name,
        accent=disease.accent,
        family=disease.family,
        probability=probability,
    )


def _gates(summary: dict) -> list[StudyGate]:
    """The CC v4.0 decision gates, each with the number that opened or closed it."""
    thresholds = summary["thresholds"]
    return [
        StudyGate(
            key="irp_elevated",
            label="IRP mediana elevada",
            open=bool(summary["irp_elevated"]),
            detail=f"{summary['median_irp4']:.1f} mmHg (corte {thresholds['irp_upper']:.0f} mmHg)",
        ),
        StudyGate(
            key="absent_contractility_gate",
            label="100% de deglutições falhas",
            open=bool(summary["absent_contractility_gate"]),
            detail=f"{summary['failed_fraction'] * 100:.0f}% com DCI < {thresholds['dci_failed']:.0f}",
        ),
        StudyGate(
            key="ineffective_gate",
            label="Mais de 70% ineficazes",
            open=bool(summary["ineffective_gate"]),
            detail=f"{summary['ineffective_fraction'] * 100:.0f}% com DCI < {thresholds['dci_weak']:.0f}",
        ),
        StudyGate(
            key="premature_gate",
            label="≥ 20% prematuras",
            open=bool(summary["premature_gate"]),
            detail=(
                f"{summary['premature_fraction'] * 100:.0f}% com DL < {thresholds['dl_premature']:.1f} s "
                f"(menor DL {summary['min_distal_latency']:.1f} s)"
            ),
        ),
        StudyGate(
            key="hypercontractile_gate",
            label="≥ 20% hipercontráteis",
            open=bool(summary["hypercontractile_gate"]),
            detail=(
                f"{summary['hypercontractile_fraction'] * 100:.0f}% com DCI > {thresholds['dci_hyper']:.0f} "
                f"(máximo {summary['max_dci']:.0f})"
            ),
        ),
        StudyGate(
            key="panesophageal_gate",
            label="≥ 20% com pressurização panesofágica",
            open=bool(summary["panesophageal_gate"]),
            detail=f"{summary['panesophageal_fraction'] * 100:.0f}% das deglutições",
        ),
    ]


def _swallow_metrics(metrics: dict[str, float]) -> list[SwallowMetric]:
    return [
        SwallowMetric(
            key=key,
            label=METRIC_LABELS[key]["label"],
            long=METRIC_LABELS[key]["long"],
            unit=METRIC_LABELS[key]["unit"],
            value=float(value),
        )
        for key, value in metrics.items()
        if key in METRIC_LABELS
    ]


def _panel_audit(panels) -> PanelAudit:
    counts = {status: 0 for status in
              ("accepted", "accepted_off_spec", "logo", "placeholder", "not_localizable", "unreadable")}
    for panel in panels:
        counts[panel.status] = counts.get(panel.status, 0) + 1
    return PanelAudit(
        total=len(panels),
        accepted=counts["accepted"] + counts["accepted_off_spec"],
        off_spec=counts["accepted_off_spec"],
        logo=counts["logo"],
        placeholder=counts["placeholder"],
        not_localizable=counts["not_localizable"],
        unreadable=counts["unreadable"],
    )


def _plural(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def _warnings(audit: PanelAudit, reading) -> list[str]:
    notes: list[str] = []
    used = sum(1 for swallow in reading.swallows if swallow.used)
    if used < 10:
        notes.append(
            f"O laudo trouxe {_plural(used, 'deglutição válida', 'deglutições válidas')}; o modelo lê até 10. "
            "Com menos deglutições a estatística do estudo fica mais ruidosa."
        )
    if len(reading.swallows) > used:
        extra = len(reading.swallows) - used
        notes.append(
            f"{_plural(extra, 'deglutição', 'deglutições')} além da décima "
            f"{'foi medida e exibida' if extra == 1 else 'foram medidas e exibidas'}, "
            "mas não entraram na classificação."
        )
    if audit.placeholder:
        notes.append(
            f"{_plural(audit.placeholder, 'painel', 'painéis')} “sem dados” do software de aquisição "
            f"{'foi descartado' if audit.placeholder == 1 else 'foram descartados'}."
        )
    if audit.off_spec:
        notes.append(
            f"{_plural(audit.off_spec, 'painel', 'painéis')} fora do formato de 420 px do export Multiplex "
            f"{'foi aceito' if audit.off_spec == 1 else 'foram aceitos'} após localização do gráfico; "
            "confira as miniaturas."
        )
    unusable = audit.not_localizable + audit.unreadable
    if unusable:
        notes.append(
            f"{_plural(unusable, 'imagem não pôde', 'imagens não puderam')} ser interpretada"
            f"{'' if unusable == 1 else 's'} como topografia de pressão."
        )
    low_confidence = [s.index for s in reading.swallows if s.colour_confidence < 0.6]
    if low_confidence:
        notes.append(
            "Inversão de cor com baixa cobertura "
            f"{'na deglutição' if len(low_confidence) == 1 else 'nas deglutições'} "
            f"{', '.join(str(index) for index in low_confidence)}: as métricas de Chicago "
            "dessas imagens são menos confiáveis."
        )
    return notes


def _model_card() -> ModelCard:
    return ModelCard(**CLASSIFIER.card) if CLASSIFIER.card else ModelCard()


# -------------------------------------------------------------------- routes


@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        model_available=CLASSIFIER.available,
        model_loaded=CLASSIFIER.loaded,
        device=str(CLASSIFIER.device or SETTINGS.device),
        checkpoint=str(SETTINGS.checkpoint_path),
    )


@app.get("/api/model", response_model=ModelCard, tags=["meta"])
def model_card() -> ModelCard:
    try:
        CLASSIFIER.load()
    except ModelUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return _model_card()


@app.get("/api/vocabulary", tags=["meta"])
def vocabulary() -> dict:
    """Labels the interface uses for swallow patterns and Chicago metrics."""
    return {"patterns": SWALLOW_PATTERNS, "metrics": METRIC_LABELS}


@app.get("/api/diseases", tags=["reference"])
def diseases() -> dict:
    return {"diseases": [disease.to_dict() for disease in DISEASES]}


@app.get("/api/diseases/{slug}", tags=["reference"])
def disease(slug: str) -> dict:
    try:
        return get_disease(slug).to_dict()
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/diseases/{slug}/reference.png", tags=["reference"])
def disease_reference(slug: str) -> Response:
    """A real swallow topography of this diagnosis, for the reference gallery."""
    try:
        get_disease(slug)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    payload = reference_png(slug)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Nenhuma imagem de referência disponível para {slug}. "
                "Gere-as com scripts/export_reference_images.py."
            ),
        )
    return Response(
        content=payload,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _sample_exams(slug: str) -> list:
    return sorted((SETTINGS.dataset_root / "test" / slug).glob("exam_*"))


@app.get("/api/samples", tags=["reference"])
def samples() -> dict:
    """Which diagnoses a demo laudo can be generated for on this machine."""
    return {
        "available": [
            disease.slug for disease in DISEASES if len(_sample_exams(disease.slug)) > 0
        ]
    }


@app.get("/api/sample-report", tags=["reference"])
def sample_report(diagnosis: str, seed: int = 0) -> Response:
    """A real examination of ``diagnosis``, re-wrapped as a laudo `.docx`.

    The corpus is stored already cropped, so the panels are re-framed with the
    white UI chrome the Multiplex export carries. Uploading the result therefore
    exercises the whole path — reader, cropper, colour inversion, classifier —
    exactly as a laudo from the hospital would.
    """
    try:
        get_disease(diagnosis)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    exams = _sample_exams(diagnosis)
    if not exams:
        raise HTTPException(
            status_code=404,
            detail=f"O corpus de exemplo nao esta disponivel em {SETTINGS.dataset_root}.",
        )
    exam = exams[seed % len(exams)]
    payload = build_docx_bytes(
        laudo_from_exam(exam),
        text=[
            "SERVICO DE MOTILIDADE DIGESTIVA",
            "Manometria esofagica de alta resolucao - 10 degluticoes de agua",
            f"Exame: {exam.name}",
        ],
    )
    return Response(
        content=payload,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="laudo_{diagnosis}.docx"'},
    )


@app.post("/api/analyze", response_model=AnalysisResponse, tags=["inference"])
async def analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    filename = file.filename or "laudo.docx"
    if not filename.lower().endswith(ACCEPTED_SUFFIXES):
        raise HTTPException(
            status_code=415,
            detail="Envie o laudo de manometria em .docx (ou .rtf, o formato original do Multiplex).",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")
    if len(data) > SETTINGS.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo acima do limite de {SETTINGS.max_upload_bytes // (1024 * 1024)} MB.",
        )

    try:
        swallows, panels = read_swallows(data, filename)
    except ReportError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    try:
        reading = CLASSIFIER.analyse(swallows)
    except ModelUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    order = np.argsort(reading.probabilities)[::-1]
    ranking = [
        _class_score(reading.classes[int(position)], float(reading.probabilities[int(position)]))
        for position in order
    ]
    winner = get_disease(ranking[0].slug)
    diagnosis = Diagnosis(
        **ranking[0].model_dump(), tagline=winner.tagline, summary=winner.summary
    )

    results: list[SwallowResult] = []
    for swallow in reading.swallows:
        pattern = SWALLOW_PATTERNS[swallow.pattern]
        top = None
        if swallow.used and swallow.probabilities.size:
            best = int(np.argmax(swallow.probabilities))
            top = _class_score(reading.classes[best], float(swallow.probabilities[best]))
        results.append(
            SwallowResult(
                index=swallow.index,
                used=swallow.used,
                pattern=swallow.pattern,
                pattern_label=pattern["label"],
                pattern_description=pattern["description"],
                colour_confidence=swallow.colour_confidence,
                metrics=_swallow_metrics(swallow.metrics),
                top_class=top,
                thumbnail=swallow.thumbnail,
            )
        )

    audit = _panel_audit(panels)
    summary = reading.study_summary
    study = StudySummary(
        **{key: value for key, value in summary.items() if key in StudySummary.model_fields and key != "gates"},
        gates=_gates(summary),
    )

    return AnalysisResponse(
        filename=filename,
        diagnosis=diagnosis,
        ranking=ranking,
        swallows=results,
        study=study,
        panels=audit,
        warnings=_warnings(audit, reading),
        elapsed_ms=reading.elapsed_ms,
        model_card=_model_card(),
    )
