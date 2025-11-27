
import os
from pathlib import Path

basedir = Path(__file__).resolve().parent

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "sqlite:///" + str(basedir / "app.db"),
    )
    # Heroku may provide postgres://, SQLAlchemy expects postgresql://
    if SQLALCHEMY_DATABASE_URI.startswith("postgres://"):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace("postgres://", "postgresql://", 1)

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    WTF_CSRF_ENABLED = True
