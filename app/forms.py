
from flask_wtf import FlaskForm
from wtforms import (
    StringField,
    PasswordField,
    SubmitField,
    IntegerField,
    FloatField,
    BooleanField,
    SelectField,
)
from wtforms.validators import DataRequired, Email, Length, NumberRange, Optional

class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Log in")


class RegisterForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired(), Length(max=120)])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=6)])
    submit = SubmitField("Register")


class IntubationForm(FlaskForm):
    age = IntegerField("Age", validators=[DataRequired(), NumberRange(min=0, max=120)])
    weight = FloatField("Weight (kg)", validators=[DataRequired(), NumberRange(min=1, max=300)])

    dtm = FloatField("Thyromental distance (cm)", validators=[Optional(), NumberRange(min=0, max=20)])
    dii = FloatField("Interincisor distance (cm)", validators=[Optional(), NumberRange(min=0, max=10)])
    mallampati = IntegerField("Mallampati class (1-4)", validators=[Optional(), NumberRange(min=1, max=4)])
    stop_bang = IntegerField("STOP-BANG score (0-8)", validators=[Optional(), NumberRange(min=0, max=8)])
    alganzouri = IntegerField("Al-Ganzouri score", validators=[Optional(), NumberRange(min=0, max=12)])

    drug_used = StringField("Drug(s) used", validators=[Optional(), Length(max=120)])
    technique = StringField("Technique", validators=[Optional(), Length(max=120)])

    success = BooleanField("Intubation successful?")
    cormack = IntegerField("Cormack-Lehane grade (1-4)", validators=[Optional(), NumberRange(min=1, max=4)])

    submit = SubmitField("Save record")
