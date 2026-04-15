"""Modelos de datos estandarizados para resultados numericos."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RootIteration:
    iteracion: int
    aproximacion: float
    error_absoluto: float
    error_relativo: Optional[float]
    error_local: Optional[float] = None
    error_global: Optional[float] = None
    detalles: Dict[str, float] = field(default_factory=dict)


@dataclass
class RootResult:
    metodo: str
    convergio: bool
    aproximacion: float
    iteraciones: List[RootIteration]
    mensaje: str
    metadatos: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterpolationStep:
    iteracion: int
    h: Optional[float]
    x: float
    P_x: float
    y_exacto: Optional[float] = None
    error_local: Optional[float] = None
    error_global: Optional[float] = None
    error_relativo: Optional[float] = None

@dataclass
class InterpolationResult:
    metodo: str
    valor_interpolado: float
    x_eval: float
    puntos: List[tuple[float, float]]
    mensaje: str
    pasos: List[InterpolationStep] = field(default_factory=list)
    metadatos: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationSample:
    indice: int
    x: float
    fx: float
    peso: float
    aporte: float


@dataclass
class IntegrationResult:
    metodo: str
    valor_aproximado: float
    valor_exacto: Optional[float]
    error_absoluto: Optional[float]
    error_relativo: Optional[float]
    muestras: List[IntegrationSample]
    mensaje: str
    metadatos: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ODEStep:
    paso: int
    t: float
    y: float
    y_exacto: Optional[float] = None
    error_absoluto: Optional[float] = None
    error_local: Optional[float] = None
    error_global: Optional[float] = None


@dataclass
class ODEResult:
    metodo: str
    pasos: List[ODEStep]
    mensaje: str
    metadatos: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonteCarloPoint:
    indice: int
    x: float
    fx: float
    aporte: float


@dataclass
class MonteCarloResult:
    metodo: str
    estimacion: float
    desvio_muestral: float
    error_estandar: float
    ic_bajo: float
    ic_alto: float
    confianza: float
    puntos: List[MonteCarloPoint]
    mensaje: str
    metadatos: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LinearSystemResult:
    metodo: str
    solucion: List[float]
    residuo: List[float]
    mensaje: str
    metadatos: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SequenceAccelerationResult:
    metodo: str
    convergio: bool
    aproximacion: float
    iteraciones: List[RootIteration]
    mensaje: str
    metadatos: Dict[str, Any] = field(default_factory=dict)
