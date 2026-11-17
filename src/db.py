"""Relational schema. PostgreSQL in Docker, SQLite for local runs and tests.

observations / photos : the training data, with license and attribution kept per photo
predictions           : every served prediction, with its embedding (input for drift detection)
feedback              : later ground truth (user correction or iNat community ID)
"""
from datetime import datetime, timezone

from sqlalchemy import (Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text,
                        create_engine)
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src import config


def utcnow():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Observation(Base):
    __tablename__ = "observations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)  # iNaturalist observation id
    taxon_id: Mapped[int] = mapped_column(Integer, index=True)
    taxon_name: Mapped[str] = mapped_column(String(200))
    quality_grade: Mapped[str] = mapped_column(String(20))
    observed_on: Mapped[str | None] = mapped_column(String(20), nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    user_login: Mapped[str | None] = mapped_column(String(100), nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Photo(Base):
    __tablename__ = "photos"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)  # iNaturalist photo id
    observation_id: Mapped[int] = mapped_column(ForeignKey("observations.id"), index=True)
    license_code: Mapped[str] = mapped_column(String(20))
    attribution: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    local_path: Mapped[str | None] = mapped_column(Text, nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    model_version: Mapped[str] = mapped_column(String(50))
    predicted_taxon: Mapped[str] = mapped_column(String(200))
    confidence: Mapped[float] = mapped_column(Float)
    prediction_set: Mapped[str] = mapped_column(Text)  # JSON list of taxon names
    needs_review: Mapped[bool] = mapped_column(Boolean)
    latency_ms: Mapped[float] = mapped_column(Float)
    image_sha256: Mapped[str] = mapped_column(String(64))
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)  # float16 bytes


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), index=True)
    true_taxon: Mapped[str] = mapped_column(String(200))
    source: Mapped[str] = mapped_column(String(20), default="user")  # user | expert | inat
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


_engine = None
_Session = None


def get_engine(url: str | None = None):
    global _engine, _Session
    if url is not None or _engine is None:
        url = url or config.DATABASE_URL
        if url.startswith("sqlite:///"):
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        if url == "sqlite://":  # in-memory (tests): one shared connection, else each one sees an empty DB
            _engine = create_engine(url, poolclass=StaticPool, connect_args={"check_same_thread": False})
        else:
            _engine = create_engine(url, pool_pre_ping=True)
        _Session = sessionmaker(_engine, expire_on_commit=False)
        Base.metadata.create_all(_engine)
    return _engine


def session():
    get_engine()
    return _Session()
