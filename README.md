
# intuB – Prototype difficult intubation logging app

**Important:** This is a research / educational prototype, not a medical device.
Do not use it for real‑time clinical decision making.

## Features

- Flask web app with login / registration
- PostgreSQL / SQLAlchemy backend (works with Heroku Postgres via `DATABASE_URL`)
- Form for recording intubation episodes (demographics, airway scores, technique, outcome)
- Simple logistic regression model (scikit‑learn) to correlate predictors with a binary
  difficult / easy outcome and Cormack–Lehane grade
- Analytics page with model metrics and coefficient bar chart (Chart.js)
- Blog‑style background page

## Local setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# set up the database (SQLite by default)
export FLASK_APP=wsgi.py
flask db init
flask db migrate -m "Initial tables"
flask db upgrade

flask run
```

## Heroku deployment (outline)

1. Create a GitHub repo and push this project.
2. In Heroku, create a new app and connect it to the GitHub repo.
3. Add the Heroku Postgres add‑on.
4. Ensure the `DATABASE_URL` config var is set automatically by Heroku.
5. Deploy from the main branch.
6. Run database migrations from the Heroku dashboard (or CLI):

```bash
heroku run flask db upgrade
```

After that, you can register a user, start entering records, and view analytics.
