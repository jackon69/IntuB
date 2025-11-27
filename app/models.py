
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, login_manager

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    records = db.relationship("IntubationRecord", backref="operator", lazy="dynamic")

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def __repr__(self) -> str:
        return f"<User {self.email}>"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class IntubationRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    # Demographics
    age = db.Column(db.Integer, nullable=False)
    weight = db.Column(db.Float, nullable=False)

    # Airway scores / distances
    dtm = db.Column(db.Float, nullable=True)  # thyromental distance (cm)
    dii = db.Column(db.Float, nullable=True)  # interincisor distance (cm)
    mallampati = db.Column(db.Integer, nullable=True)
    stop_bang = db.Column(db.Integer, nullable=True)
    alganzouri = db.Column(db.Integer, nullable=True)

    # Technique
    drug_used = db.Column(db.String(120), nullable=True)
    technique = db.Column(db.String(120), nullable=True)

    # Outcome
    success = db.Column(db.Boolean, nullable=False)
    cormack = db.Column(db.Integer, nullable=True)

    difficult_binary = db.Column(db.Boolean, nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    operator_id = db.Column(db.Integer, db.ForeignKey("user.id"))

    def __repr__(self) -> str:
        return f"<IntubationRecord {self.id}>"
