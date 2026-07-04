"""Flask web application for Credit Card Approval Prediction."""

from __future__ import annotations

import csv
import base64
import json
import io
import logging
import os
import re
import sqlite3
import secrets
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import pandas as pd
from flask import (
    Flask,
    Response,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is installed from requirements.txt
    def load_dotenv(*args, **kwargs):
        return False

from predict import CreditApprovalPredictor
from preprocessing import TARGET_COLUMN, build_single_record


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
HISTORY_PATH = BASE_DIR / "prediction_history.csv"
HISTORY_DB_PATH = BASE_DIR / "prediction_history.db"
ALLOWED_EXTENSIONS = {"csv"}
BATCH_REQUIRED_COLUMNS = [
    "ID",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
]

load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key-change-before-production")
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
UPLOAD_DIR.mkdir(exist_ok=True)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


def login_required(view):
    def wrapped_view(*args, **kwargs):
        if not session.get("logged_in"):
            flash("Please log in to use that page.", "warning")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    wrapped_view.__name__ = view.__name__
    return wrapped_view


def history_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(HISTORY_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_history_db() -> None:
    with history_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                first_name TEXT,
                last_name TEXT,
                phone_number TEXT,
                gender TEXT,
                age TEXT,
                income TEXT,
                education TEXT,
                occupation TEXT,
                prediction TEXT,
                approval_probability REAL,
                risk_level TEXT
            )
            """
        )
        existing_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(prediction_history)").fetchall()
        }
        for column in ["first_name", "last_name", "phone_number"]:
            if column not in existing_columns:
                connection.execute(f"ALTER TABLE prediction_history ADD COLUMN {column} TEXT")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                email TEXT UNIQUE,
                phone_number TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        count = connection.execute("SELECT COUNT(*) FROM prediction_history").fetchone()[0]
        if count == 0 and HISTORY_PATH.exists():
            rows = pd.read_csv(HISTORY_PATH).to_dict(orient="records")
            connection.executemany(
                """
                INSERT INTO prediction_history (
                    timestamp, gender, age, income, education, occupation,
                    prediction, approval_probability, risk_level
                )
                VALUES (
                    :timestamp, :gender, :age, :income, :education, :occupation,
                    :prediction, :approval_probability, :risk_level
                )
                """,
                rows,
            )


init_history_db()


def get_predictor() -> CreditApprovalPredictor | None:
    """Load predictor lazily so the site can show a helpful message if missing."""

    try:
        return CreditApprovalPredictor()
    except Exception as exc:
        LOGGER.warning("%s", exc)
        return None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def missing_batch_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in BATCH_REQUIRED_COLUMNS if column not in df.columns]


def validate_form(form) -> list[str]:
    """Validate prediction form inputs."""

    errors = []
    numeric_rules = {
        "age": (1, 100),
        "income": (1, 10000000),
        "family_members": (1, 20),
        "years_employed": (0, 60),
        "children": (0, 10),
    }
    required_fields = [
        "first_name",
        "last_name",
        "phone_number",
        "gender",
        "age",
        "income",
        "family_members",
        "education",
        "occupation",
        "housing_type",
        "car_ownership",
        "real_estate_ownership",
        "years_employed",
        "children",
        "income_type",
        "marital_status",
    ]
    for field in required_fields:
        if not form.get(field):
            errors.append(f"{field.replace('_', ' ').title()} is required.")
    phone = form.get("phone_number", "")
    if phone and not is_valid_phone(phone):
        errors.append("Phone Number must contain 10 to 15 digits.")
    for field, (minimum, maximum) in numeric_rules.items():
        try:
            value = float(form.get(field, ""))
        except ValueError:
            errors.append(f"{field.replace('_', ' ').title()} must be a number.")
            continue
        if value < minimum or value > maximum:
            errors.append(f"{field.replace('_', ' ').title()} must be between {minimum} and {maximum}.")
    return errors


def is_valid_email(value: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value.strip()))


def is_valid_phone(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return 10 <= len(digits) <= 15


def normalize_phone(value: str) -> str:
    return re.sub(r"\D", "", value)


def signup_errors(form) -> list[str]:
    errors = []
    first_name = form.get("first_name", "").strip()
    last_name = form.get("last_name", "").strip()
    email = form.get("email", "").strip().lower()
    phone = form.get("phone_number", "").strip()
    password = form.get("password", "")
    confirm_password = form.get("confirm_password", "")

    if not first_name:
        errors.append("First name is required.")
    if not last_name:
        errors.append("Last name is required.")
    if not is_valid_email(email):
        errors.append("Enter a valid email address.")
    if not is_valid_phone(phone):
        errors.append("Enter a valid phone number with 10 to 15 digits.")
    if len(password) < 6:
        errors.append("Password must be at least 6 characters.")
    if password != confirm_password:
        errors.append("Passwords do not match.")
    return errors


def find_user(identifier: str) -> sqlite3.Row | None:
    identifier = identifier.strip().lower()
    phone = normalize_phone(identifier)
    with history_connection() as connection:
        return connection.execute(
            "SELECT * FROM users WHERE lower(email) = ? OR phone_number = ?",
            (identifier, phone),
        ).fetchone()


def create_quick_user(identifier: str, password: str) -> sqlite3.Row:
    identifier = identifier.strip().lower()
    phone = normalize_phone(identifier)
    email = identifier if is_valid_email(identifier) else None
    phone_number = phone if is_valid_phone(identifier) else None
    if not email and not phone_number:
        raise ValueError("Enter a valid email address or phone number.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    if email:
        first_name = email.split("@")[0].split(".")[0].title() or "New"
        last_name = "User"
    else:
        first_name = "Phone"
        last_name = "User"

    with history_connection() as connection:
        connection.execute(
            """
            INSERT INTO users (
                first_name, last_name, email, phone_number, password_hash, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                first_name,
                last_name,
                email,
                phone_number,
                generate_password_hash(password),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        return connection.execute(
            "SELECT * FROM users WHERE lower(email) = ? OR phone_number = ?",
            (email or "", phone_number or ""),
        ).fetchone()


def find_or_create_google_user(profile: dict) -> sqlite3.Row:
    email = profile.get("email", "").strip().lower()
    if not is_valid_email(email):
        raise ValueError("Google account did not return a valid email address.")

    first_name = profile.get("given_name") or profile.get("name", "Google").split(" ")[0]
    last_name = profile.get("family_name") or "User"
    with history_connection() as connection:
        user = connection.execute("SELECT * FROM users WHERE lower(email) = ?", (email,)).fetchone()
        if user:
            return user
        connection.execute(
            """
            INSERT INTO users (
                first_name, last_name, email, phone_number, password_hash, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                first_name,
                last_name,
                email,
                None,
                generate_password_hash(secrets.token_urlsafe(24)),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        return connection.execute("SELECT * FROM users WHERE lower(email) = ?", (email,)).fetchone()


def google_oauth_configured() -> bool:
    return bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"))


def is_loopback_host(hostname: str | None) -> bool:
    return hostname in {"localhost", "127.0.0.1", "::1"}


def is_safe_next_url(target: str | None) -> bool:
    if not target:
        return False
    parsed = urllib.parse.urlparse(target)
    return not parsed.scheme and not parsed.netloc and target.startswith("/")


def login_redirect_target(default_endpoint: str = "prediction"):
    next_url = request.args.get("next") or session.pop("next_url", None)
    if is_safe_next_url(next_url):
        return redirect(next_url)
    return redirect(url_for(default_endpoint))


def google_redirect_uri() -> str:
    configured_uri = os.getenv("GOOGLE_REDIRECT_URI", "").strip()
    if not configured_uri:
        return url_for("google_callback", _external=True)

    configured = urllib.parse.urlparse(configured_uri)
    current_host = request.host.split(":", 1)[0]
    if is_loopback_host(configured.hostname) and is_loopback_host(current_host):
        return url_for("google_callback", _external=True)
    return configured_uri


def google_canonical_login_url() -> str | None:
    configured_uri = os.getenv("GOOGLE_REDIRECT_URI", "").strip()
    if not configured_uri:
        return None

    configured = urllib.parse.urlparse(configured_uri)
    current = urllib.parse.urlparse(request.url)
    current_host = request.host.split(":", 1)[0]
    if (
        is_loopback_host(configured.hostname)
        and is_loopback_host(current_host)
        and configured.netloc
        and configured.netloc != current.netloc
    ):
        query = urllib.parse.urlencode(
            {"next": request.args.get("next")}
            if is_safe_next_url(request.args.get("next"))
            else {}
        )
        return urllib.parse.urlunparse(
            (current.scheme, configured.netloc, url_for("google_login"), "", query, "")
        )
    return None


def google_http_error_message(exc: urllib.error.HTTPError) -> str:
    body = exc.read().decode("utf-8", errors="replace")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        payload = {}
    error = payload.get("error") or exc.reason
    description = payload.get("error_description") or body
    return f"{error}: {description}".strip(": ")


def google_profile_from_id_token(id_token: str) -> dict:
    parts = id_token.split(".")
    if len(parts) < 2:
        raise ValueError("Google did not return a readable identity token.")
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    return json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))


def google_login_error_message(exc: Exception) -> str:
    if isinstance(exc, KeyError):
        return f"Google response was missing this field: {exc}"
    if isinstance(exc, urllib.error.URLError):
        return str(exc.reason)
    return str(exc)


def save_history(form_data: dict, result) -> None:
    """Persist one prediction to SQLite history."""

    with history_connection() as connection:
        connection.execute(
            """
            INSERT INTO prediction_history (
                timestamp, first_name, last_name, phone_number, gender, age, income, education, occupation,
                prediction, approval_probability, risk_level
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                form_data.get("first_name"),
                form_data.get("last_name"),
                normalize_phone(form_data.get("phone_number", "")),
                form_data.get("gender"),
                form_data.get("age"),
                form_data.get("income"),
                form_data.get("education"),
                form_data.get("occupation"),
                result.label,
                result.probability,
                result.risk_level,
            ),
        )


def recent_history(limit: int = 8) -> list[dict]:
    with history_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM prediction_history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def history_dataframe() -> pd.DataFrame:
    with history_connection() as connection:
        return pd.read_sql_query("SELECT * FROM prediction_history ORDER BY id DESC", connection)


def history_stats() -> dict:
    rows = history_dataframe()
    if rows.empty:
        return {
            "total": 0,
            "approved": 0,
            "rejected": 0,
            "average_probability": 0,
            "prediction_counts": {"Approved": 0, "Rejected": 0},
            "risk_counts": {"Low Risk": 0, "Moderate Risk": 0, "High Risk": 0},
            "recent": [],
        }
    prediction_counts = rows["prediction"].value_counts().to_dict()
    risk_counts = rows["risk_level"].value_counts().to_dict()
    return {
        "total": int(len(rows)),
        "approved": int(prediction_counts.get("Approved", 0)),
        "rejected": int(prediction_counts.get("Rejected", 0)),
        "average_probability": round(float(rows["approval_probability"].mean()), 2),
        "prediction_counts": {
            "Approved": int(prediction_counts.get("Approved", 0)),
            "Rejected": int(prediction_counts.get("Rejected", 0)),
        },
        "risk_counts": {
            "Low Risk": int(risk_counts.get("Low Risk", 0)),
            "Moderate Risk": int(risk_counts.get("Moderate Risk", 0)),
            "High Risk": int(risk_counts.get("High Risk", 0)),
        },
        "recent": rows.head(12).to_dict(orient="records"),
    }


def prediction_reasons(form_data: dict, result) -> list[str]:
    reasons = []
    income = float(form_data.get("income", 0))
    age = float(form_data.get("age", 0))
    years = float(form_data.get("years_employed", 0))
    family_members = max(float(form_data.get("family_members", 1)), 1)
    children = float(form_data.get("children", 0))
    income_per_member = income / family_members
    income_type = form_data.get("income_type", "")

    if age < 18:
        reasons.append("Applicant is below 18, so the application is rejected by eligibility rules.")
    if income_type in {"Student", "Unemployed"}:
        reasons.append(f"{income_type} income status is not eligible for approval.")
    if income < 100000:
        reasons.append("Very low annual income is below the approval threshold.")
    elif income >= 200000:
        reasons.append("Higher income improved the approval chance.")
    else:
        reasons.append("Lower income reduced the approval confidence.")
    if years >= 3:
        reasons.append("Stable employment history supported the result.")
    else:
        reasons.append("Short employment history increased the risk signal.")
    if form_data.get("real_estate_ownership") == "Y":
        reasons.append("Real estate ownership helped the stability score.")
    if form_data.get("car_ownership") == "Y":
        reasons.append("Car ownership added a small positive stability signal.")
    if income_per_member < 75000:
        reasons.append("Income per family member is low, so dependency risk is higher.")
    if children >= 3:
        reasons.append("More children can increase financial dependency pressure.")
    if result.label == "Rejected" and result.probability < 50:
        reasons.append("The approval probability is below the safer approval range.")
    return reasons[:5]


@app.route("/history/delete/<int:history_id>", methods=["POST"])
@login_required
def delete_history_item(history_id: int):
    with history_connection() as connection:
        cursor = connection.execute("DELETE FROM prediction_history WHERE id = ?", (history_id,))
    if cursor.rowcount == 0:
        flash("That prediction history item could not be found.", "warning")
        return redirect(url_for("home"))
    flash("Prediction history item deleted.", "success")
    return redirect(url_for("home"))


@app.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    with history_connection() as connection:
        connection.execute("DELETE FROM prediction_history")
    flash("All prediction history cleared.", "success")
    return redirect(url_for("home"))


@app.route("/")
def home():
    predictor = get_predictor()
    predictor_ready = predictor is not None
    using_rule_fallback = bool(getattr(predictor, "using_rule_fallback", False))
    stats = history_stats()
    return render_template(
        "index.html",
        predictor_ready=predictor_ready,
        using_rule_fallback=using_rule_fallback,
        history=recent_history(),
        stats=stats,
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    next_url = request.args.get("next") if is_safe_next_url(request.args.get("next")) else None
    if request.method == "GET":
        return render_template("login.html", next_url=next_url)

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    user = find_user(username)
    expected_username = os.getenv("ADMIN_USERNAME", "admin")
    expected_password = os.getenv("ADMIN_PASSWORD", "admin123")
    if (user and check_password_hash(user["password_hash"], password)) or (
        username == expected_username and password == expected_password
    ):
        session["logged_in"] = True
        session["username"] = (user["email"] or user["phone_number"]) if user else username
        flash("Logged in successfully.", "success")
        return login_redirect_target()

    if user is None and (is_valid_email(username) or is_valid_phone(username)):
        try:
            user = create_quick_user(username, password)
        except (ValueError, sqlite3.IntegrityError) as exc:
            flash(str(exc), "danger")
            return render_template("login.html", next_url=next_url)
        session["logged_in"] = True
        session["username"] = user["email"] or user["phone_number"]
        flash("Account created and logged in successfully.", "success")
        return login_redirect_target()

    flash("Invalid username or password.", "danger")
    return render_template("login.html", next_url=next_url)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html", form_data={})

    errors = signup_errors(request.form)
    if errors:
        for error in errors:
            flash(error, "danger")
        return render_template("signup.html", form_data=request.form)

    email = request.form.get("email", "").strip().lower()
    phone = normalize_phone(request.form.get("phone_number", ""))
    try:
        with history_connection() as connection:
            connection.execute(
                """
                INSERT INTO users (
                    first_name, last_name, email, phone_number, password_hash, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    request.form.get("first_name", "").strip(),
                    request.form.get("last_name", "").strip(),
                    email,
                    phone,
                    generate_password_hash(request.form.get("password", "")),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
    except sqlite3.IntegrityError:
        flash("An account already exists with that email or phone number.", "danger")
        return render_template("signup.html", form_data=request.form)

    session["logged_in"] = True
    session["username"] = email
    flash("Account created successfully.", "success")
    return redirect(url_for("prediction"))


@app.route("/login/google")
def google_login():
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    if not google_oauth_configured():
        flash("Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env, then restart the app.", "warning")
        return redirect(url_for("login"))

    canonical_url = google_canonical_login_url()
    if canonical_url:
        return redirect(canonical_url)

    state = secrets.token_urlsafe(24)
    session["google_oauth_state"] = state
    if is_safe_next_url(request.args.get("next")):
        session["next_url"] = request.args.get("next")
    params = {
        "client_id": client_id,
        "redirect_uri": google_redirect_uri(),
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
    }
    return redirect(f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}")


@app.route("/login/google/callback")
def google_callback():
    if request.args.get("error"):
        flash("Google login was cancelled or denied.", "warning")
        return redirect(url_for("login"))

    state = request.args.get("state")
    if not state or state != session.pop("google_oauth_state", None):
        flash("Google login session expired. Please try again.", "danger")
        return redirect(url_for("login"))

    code = request.args.get("code")
    if not code:
        flash("Google did not return a login code.", "danger")
        return redirect(url_for("login"))

    token_data = urllib.parse.urlencode(
        {
            "code": code,
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            "redirect_uri": google_redirect_uri(),
            "grant_type": "authorization_code",
        }
    ).encode("utf-8")
    try:
        token_request = urllib.request.Request(
            GOOGLE_TOKEN_URL,
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(token_request, timeout=10) as response:
            token_response = json.loads(response.read().decode("utf-8"))
        access_token = token_response.get("access_token")
        if access_token:
            try:
                user_request = urllib.request.Request(
                    GOOGLE_USERINFO_URL,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                with urllib.request.urlopen(user_request, timeout=10) as response:
                    profile = json.loads(response.read().decode("utf-8"))
            except urllib.error.URLError:
                if not token_response.get("id_token"):
                    raise
                profile = google_profile_from_id_token(token_response["id_token"])
        elif token_response.get("id_token"):
            profile = google_profile_from_id_token(token_response["id_token"])
        else:
            raise ValueError("Google did not return account details.")
        user = find_or_create_google_user(profile)
    except urllib.error.HTTPError as exc:
        details = google_http_error_message(exc)
        LOGGER.warning("Google login failed: %s", details)
        if "redirect_uri_mismatch" in details:
            flash("Google login failed because the redirect URL does not match Google Console. Add this exact redirect URL there: " + google_redirect_uri(), "danger")
        else:
            flash("Google login failed: " + details, "danger")
        return redirect(url_for("login"))
    except (KeyError, ValueError, urllib.error.URLError) as exc:
        details = google_login_error_message(exc)
        LOGGER.warning("Google login failed: %s", details)
        flash("Google login failed: " + details, "danger")
        return redirect(url_for("login"))

    session["logged_in"] = True
    session["username"] = user["email"]
    flash("Logged in with Google successfully.", "success")
    return login_redirect_target()


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("home"))


@app.route("/predict", methods=["GET", "POST"])
@login_required
def prediction():
    if request.method == "GET":
        predictor = get_predictor()
        return render_template(
            "prediction.html",
            result=None,
            form_data={},
            using_rule_fallback=bool(getattr(predictor, "using_rule_fallback", False)),
        )

    errors = validate_form(request.form)
    if errors:
        for error in errors:
            flash(error, "danger")
        return render_template("prediction.html", result=None, form_data=request.form)

    predictor = get_predictor()
    if predictor is None:
        flash("Model is not trained yet. Run python train.py, then restart the app.", "warning")
        return render_template("prediction.html", result=None, form_data=request.form)

    result = predictor.predict_form(request.form)
    result.reasons = prediction_reasons(request.form, result)
    save_history(request.form, result)
    return render_template(
        "prediction.html",
        result=result,
        form_data=request.form,
        using_rule_fallback=bool(getattr(predictor, "using_rule_fallback", False)),
    )


@app.route("/batch", methods=["POST"])
@login_required
def batch_prediction():
    predictor = get_predictor()
    if predictor is None:
        flash("Model is not trained yet. Run python train.py before batch prediction.", "warning")
        return redirect(url_for("prediction"))

    file = request.files.get("csv_file")
    if not file or file.filename == "":
        flash("Please choose a CSV file.", "danger")
        return redirect(url_for("prediction"))
    if not allowed_file(file.filename):
        flash("Only CSV files are supported.", "danger")
        return redirect(url_for("prediction"))

    filename = secure_filename(file.filename)
    upload_path = UPLOAD_DIR / filename
    file.save(upload_path)
    raw = pd.read_csv(upload_path)
    missing_columns = missing_batch_columns(raw)
    if missing_columns:
        flash(
            "Your CSV is missing required columns: "
            + ", ".join(missing_columns[:6])
            + ("..." if len(missing_columns) > 6 else "")
            + ". Download the sample CSV below and use the same column names.",
            "danger",
        )
        return redirect(url_for("prediction"))
    if TARGET_COLUMN not in raw.columns:
        raw[TARGET_COLUMN] = 1
    predictions = predictor.predict_dataframe(raw)
    output_filename = f"batch_predictions_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    output_path = UPLOAD_DIR / output_filename
    predictions.to_csv(output_path, index=False)
    return render_template(
        "batch_results.html",
        rows=predictions.head(50).to_dict(orient="records"),
        columns=predictions.columns.tolist(),
        total=len(predictions),
        filename=output_filename,
    )


@app.route("/batch/download/<path:filename>")
@login_required
def download_batch_predictions(filename: str):
    safe_filename = secure_filename(filename)
    return send_from_directory(UPLOAD_DIR, safe_filename, as_attachment=True)


@app.route("/sample-batch-csv")
@login_required
def sample_batch_csv():
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=BATCH_REQUIRED_COLUMNS)
    writer.writeheader()
    writer.writerow(
        {
            "ID": 100001,
            "CODE_GENDER": "M",
            "FLAG_OWN_CAR": "Y",
            "FLAG_OWN_REALTY": "Y",
            "CNT_CHILDREN": 1,
            "AMT_INCOME_TOTAL": 250000,
            "NAME_INCOME_TYPE": "Working",
            "NAME_EDUCATION_TYPE": "Higher education",
            "NAME_FAMILY_STATUS": "Married",
            "NAME_HOUSING_TYPE": "House / apartment",
            "DAYS_BIRTH": -12784,
            "DAYS_EMPLOYED": -1826,
            "OCCUPATION_TYPE": "Managers",
            "CNT_FAM_MEMBERS": 3,
        }
    )
    output.seek(0)
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_batch_input.csv"},
    )


@app.route("/download-pdf", methods=["POST"])
@login_required
def download_pdf():
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    result = request.form
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle("Credit Card Approval Prediction")
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(72, 730, "Credit Card Approval Prediction")
    pdf.setFont("Helvetica", 11)
    lines = [
        f"Prediction: {result.get('label', 'N/A')}",
        f"Approval Probability: {result.get('probability', 'N/A')}%",
        f"Confidence: {result.get('confidence', 'N/A')}%",
        f"Risk Level: {result.get('risk_level', 'N/A')}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]
    y = 690
    for line in lines:
        pdf.drawString(72, y, line)
        y -= 24
    pdf.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="prediction_report.pdf", mimetype="application/pdf")


@app.route("/download-excel", methods=["POST"])
@login_required
def download_excel():
    result = request.form.to_dict()
    buffer = io.BytesIO()
    row = {
        "prediction": result.get("label", "N/A"),
        "approval_probability": result.get("probability", "N/A"),
        "confidence": result.get("confidence", "N/A"),
        "risk_level": result.get("risk_level", "N/A"),
        "generated_at": datetime.now().isoformat(timespec="minutes"),
    }
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame([row]).to_excel(writer, index=False, sheet_name="Prediction")
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="prediction_report.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/model")
def model_information():
    metadata_path = BASE_DIR / "models" / "model_metadata.pkl"
    comparison_path = BASE_DIR / "reports" / "model_comparison.csv"
    metadata = None
    comparison = []
    if metadata_path.exists():
        import joblib

        try:
            metadata = joblib.load(metadata_path)
        except Exception as exc:
            LOGGER.warning("Model metadata could not be loaded: %s", exc)
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path).to_dict(orient="records")
    return render_template("model.html", metadata=metadata, comparison=comparison)


@app.route("/admin")
@login_required
def admin_dashboard():
    return render_template("admin.html", stats=history_stats())


@app.route("/reports/plots/<path:filename>")
def report_plot(filename):
    return send_from_directory(BASE_DIR / "reports" / "plots", filename)


@app.route("/features")
def features():
    return render_template("features.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_server_error(error):
    LOGGER.exception("Internal server error: %s", error)
    return render_template("500.html"), 500


if __name__ == "__main__":
    app.run(debug=True)
