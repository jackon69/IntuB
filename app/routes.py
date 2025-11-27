
from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from sqlalchemy import func

from app import db
from app.models import User, IntubationRecord
from app.forms import LoginForm, RegisterForm, IntubationForm
from app.ml import train_logistic_model

bp = Blueprint("main", __name__)

@bp.route("/")
def landing():
    return render_template("index.html")

@bp.route("/dashboard")
@login_required
def dashboard():
    total = IntubationRecord.query.count()
    user_total = IntubationRecord.query.filter_by(operator_id=current_user.id).count()
    difficult = IntubationRecord.query.filter_by(difficult_binary=True).count()
    return render_template(
        "dashboard.html",
        total=total,
        user_total=user_total,
        difficult=difficult,
    )

@bp.route("/blog")
def blog():
    return render_template("blog.html")

@bp.route("/analytics")
@login_required
def analytics():
    total = IntubationRecord.query.count()
    diff_count = IntubationRecord.query.filter_by(difficult_binary=True).count()
    easy_count = total - diff_count

    # simple distributions
    age_stats = db.session.query(
        func.count(IntubationRecord.id),
        func.avg(IntubationRecord.age),
        func.min(IntubationRecord.age),
        func.max(IntubationRecord.age),
    ).one()

    return render_template(
        "analytics.html",
        total=total,
        easy_count=easy_count,
        diff_count=diff_count,
        age_stats=age_stats,
    )

@bp.route("/analytics/json")
@login_required
def analytics_json():
    try:
        _, metrics = train_logistic_model()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(metrics)

@bp.route("/new", methods=["GET", "POST"])
@login_required
def new_record():
    form = IntubationForm()
    if form.validate_on_submit():
        difficult = False
        if form.cormack.data is not None and form.cormack.data >= 3:
            difficult = True
        if not form.success.data:
            difficult = True

        record = IntubationRecord(
            age=form.age.data,
            weight=form.weight.data,
            dtm=form.dtm.data,
            dii=form.dii.data,
            mallampati=form.mallampati.data,
            stop_bang=form.stop_bang.data,
            alganzouri=form.alganzouri.data,
            drug_used=form.drug_used.data,
            technique=form.technique.data,
            success=form.success.data,
            cormack=form.cormack.data,
            difficult_binary=difficult,
            operator=current_user,
        )
        db.session.add(record)
        db.session.commit()
        flash("Record saved.")
        return redirect(url_for("main.dashboard"))
    return render_template("new_record.html", form=form)

@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data.lower()).first()
        if user is None or not user.check_password(form.password.data):
            flash("Invalid email or password.")
            return redirect(url_for("main.login"))
        login_user(user)
        return redirect(url_for("main.dashboard"))
    return render_template("login.html", form=form)

@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    form = RegisterForm()
    if form.validate_on_submit():
        if User.query.filter_by(email=form.email.data.lower()).first():
            flash("Email already registered.")
            return redirect(url_for("main.register"))
        user = User(
            name=form.name.data,
            email=form.email.data.lower(),
        )
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please log in.")
        return redirect(url_for("main.login"))
    return render_template("register.html", form=form)

@bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("main.landing"))
