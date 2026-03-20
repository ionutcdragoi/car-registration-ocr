"""
Flask web application for car registration OCR extraction.
Upload images -> extract fields via Tesseract -> export to CSV/XML/Google Sheets.
"""

import os
import io
import csv
import json
import uuid
import time
from pathlib import Path
import xml.etree.ElementTree as ET

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, Response
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from ocr_engine import process_image
from sheets_integration import GoogleSheetsClient

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(32).hex())

# ── Version ────────────────────────────────────────────────
APP_VERSION = "5.0.0"

# ── Configuration ──────────────────────────────────────────
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"}
MAX_FILES_PER_REQUEST = 10  # safety cap per upload

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB total

# Tesseract path
TESSERACT_CMD = os.getenv("TESSERACT_CMD", None)

# ── Export cache with TTL eviction ─────────────────────────
EXPORT_TTL = 3600  # 1 hour
EXPORT_CACHE: dict[str, dict] = {}


def _cache_put(export_id: str, payload: dict) -> None:
    """Insert into export cache and evict stale entries."""
    now = time.time()
    stale = [k for k, v in EXPORT_CACHE.items() if now - v.get("_ts", 0) > EXPORT_TTL]
    for k in stale:
        EXPORT_CACHE.pop(k, None)
    payload["_ts"] = now
    EXPORT_CACHE[export_id] = payload


def _cache_get(export_id: str) -> dict | None:
    """Retrieve from export cache if still valid."""
    entry = EXPORT_CACHE.get(export_id)
    if entry and time.time() - entry.get("_ts", 0) <= EXPORT_TTL:
        return entry
    EXPORT_CACHE.pop(export_id, None)
    return None


def get_request_sheets_client(
    sheet_url: str | None = None,
    creds_json: dict | None = None,
) -> GoogleSheetsClient | None:
    """Build a sheets client from user-uploaded credentials + sheet URL.

    Both a sheet URL and credentials are required.  If the user provided a
    sheet URL but no key (or vice-versa), we skip the export entirely.
    """
    sheet_name = os.getenv("GOOGLE_SHEET_NAME", "Car Registrations")

    if creds_json and sheet_url:
        return GoogleSheetsClient(
            credentials_info=creds_json,
            sheet_name=sheet_name,
            spreadsheet_url=sheet_url,
        )

    # Nothing usable — skip sheets export
    return None


def build_xml_export(payload: dict) -> str:
    """Build XML export for single or batch extraction payloads."""
    field_labels = {
        "license_plate": "LicensePlate",
        "vin": "VIN",
        "vehicle_make": "VehicleMake",
        "vehicle_model": "VehicleModel",
        "vehicle_type": "VehicleType",
        "color": "Color",
        "owner_name": "OwnerName",
        "owner_address": "OwnerAddress",
        "registration_date": "RegistrationDate",
        "first_registration_date": "FirstRegistrationDate",
        "engine_capacity_cc": "EngineCapacityCC",
        "engine_power_kw": "EnginePowerKW",
        "engine_power_hp": "EnginePowerHP",
        "fuel_type": "FuelType",
        "num_seats": "Seats",
        "max_weight_kg": "MaxWeightKG",
        "co2_emissions": "CO2Emissions",
        "mileage": "Mileage",
    }

    results = payload.get("results", [])
    root = ET.Element("CarRegistrations")
    root.set("count", str(len(results)))

    for item in results:
        entry = ET.SubElement(root, "CarRegistration")
        entry.set("source", item.get("filename", ""))
        for key, tag in field_labels.items():
            val = item.get("fields", {}).get(key, "")
            if val:
                el = ET.SubElement(entry, tag)
                el.text = str(val)

    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")


# CSV column order matching the Google Sheets headers
CSV_FIELDS = [
    ("license_plate", "License Plate"),
    ("owner_name", "Owner Name"),
    ("owner_address", "Owner Address"),
    ("vehicle_make", "Vehicle Make"),
    ("vehicle_model", "Vehicle Model"),
    ("vehicle_type", "Vehicle Type"),
    ("vin", "VIN"),
    ("registration_date", "Registration Date"),
    ("first_registration_date", "First Registration Date"),
    ("engine_capacity_cc", "Engine Capacity (cc)"),
    ("engine_power_kw", "Engine Power (kW)"),
    ("engine_power_hp", "Engine Power (HP)"),
    ("fuel_type", "Fuel Type"),
    ("color", "Color"),
    ("num_seats", "Number of Seats"),
    ("max_weight_kg", "Max Weight (kg)"),
    ("co2_emissions", "CO2 Emissions"),
    ("mileage", "Mileage"),
]


def build_csv_export(payload: dict) -> str:
    """Build a CSV string from cached extraction results."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    headers = ["Filename"] + [label for _, label in CSV_FIELDS]
    writer.writerow(headers)
    for item in payload.get("results", []):
        row = [item.get("filename", "")]
        for key, _ in CSV_FIELDS:
            row.append(item.get("fields", {}).get(key, ""))
        writer.writerow(row)
    return buf.getvalue()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle uploads that exceed MAX_CONTENT_LENGTH."""
    if request.accept_mimetypes.best == "application/json":
        return jsonify({"error": "File(s) too large. Maximum total is 64 MB."}), 413
    flash("File(s) too large. Maximum total upload size is 64 MB.", "error")
    return redirect(url_for("index"))


@app.route("/update/<export_id>", methods=["POST"])
def update_result(export_id: str):
    """Allow client to save edited field values back into the export cache."""
    payload = _cache_get(export_id)
    if not payload:
        return jsonify({"error": "Not found or expired"}), 404
    updated = request.get_json(force=True, silent=True)
    if not updated:
        return jsonify({"error": "No data"}), 400
    allowed_keys = {
        "license_plate", "owner_name", "owner_address", "vehicle_make",
        "vehicle_model", "vehicle_type", "vin", "registration_date",
        "first_registration_date", "engine_capacity_cc", "engine_power_kw",
        "engine_power_hp", "fuel_type", "color", "num_seats",
        "max_weight_kg", "co2_emissions", "mileage",
    }
    # Single-result payload stores fields directly; batch stores under results[]
    if payload.get("results"):
        for item in payload["results"]:
            for k, v in updated.items():
                if k in allowed_keys:
                    item["fields"][k] = v.strip() if v else ""
    _cache_put(export_id, payload)
    return jsonify({"ok": True})


@app.route("/health")
def health():
    """Health-check endpoint for load balancers / container orchestrators."""
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    """Main page with upload form."""
    return render_template("index.html", max_files=MAX_FILES_PER_REQUEST, app_version=APP_VERSION)


@app.route("/upload", methods=["POST"])
def upload():
    """Handle one or many uploads, then optionally dump all rows to Google Sheets."""
    uploaded_files = request.files.getlist("files") or request.files.getlist("file")
    uploaded_files = [f for f in uploaded_files if f and f.filename]
    if not uploaded_files:
        flash("No files selected.", "error")
        return redirect(url_for("index"))

    if len(uploaded_files) > MAX_FILES_PER_REQUEST:
        flash(f"Too many files. Maximum is {MAX_FILES_PER_REQUEST} per upload.", "error")
        return redirect(url_for("index"))

    invalid = [f.filename for f in uploaded_files if not allowed_file(f.filename)]
    if invalid:
        flash(f"Invalid file type: {', '.join(invalid)}", "error")
        return redirect(url_for("index"))

    sheet_url_input = request.form.get("sheet_url", "").strip()

    # Parse user-uploaded credentials JSON (if provided)
    user_creds_json = None
    creds_upload = request.files.get("credentials")
    if creds_upload and creds_upload.filename:
        try:
            user_creds_json = json.load(creds_upload.stream)
        except (json.JSONDecodeError, UnicodeDecodeError):
            flash("Invalid credentials JSON file.", "error")
            return redirect(url_for("index"))

    saved_paths: list[str] = []
    results: list[dict] = []

    try:
        for file in uploaded_files:
            original_name = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex}_{original_name}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            file.save(filepath)
            saved_paths.append(filepath)

            fields = process_image(filepath, TESSERACT_CMD)
            results.append({
                "filename": original_name,
                "fields": fields,
            })

        # Save all rows to Google Sheets at the end if configured
        sheet_url = None
        row_num = None
        saved_count = 0
        sheets_client = get_request_sheets_client(
            sheet_url=sheet_url_input or None,
            creds_json=user_creds_json,
        )
        if sheets_client:
            try:
                last_row = None
                for item in results:
                    last_row = sheets_client.append_record(item["fields"], item["filename"])
                    saved_count += 1
                row_num = last_row
                sheet_url = sheets_client.spreadsheet_url
            except Exception as e:
                flash(f"Google Sheets error: {e}", "warning")

        export_id = uuid.uuid4().hex
        _cache_put(export_id, {
            "results": results,
            "sheet_url": sheet_url,
        })

        if len(results) == 1:
            return render_template(
                "result.html",
                fields=results[0]["fields"],
                filename=results[0]["filename"],
                sheet_url=sheet_url,
                row_num=row_num,
                export_id=export_id,
            )

        return render_template(
            "batch_result.html",
            results=results,
            total_files=len(results),
            saved_count=saved_count,
            detected_count=sum(1 for item in results if item["fields"].get("license_plate")),
            sheet_url=sheet_url,
            row_num=row_num,
            export_id=export_id,
        )

    except Exception as e:
        flash(f"OCR processing error: {e}", "error")
        return redirect(url_for("index"))

    finally:
        for filepath in saved_paths:
            try:
                os.remove(filepath)
            except OSError:
                pass


@app.route("/export/xml/<export_id>")
def export_xml(export_id: str):
    """Export a cached single or batch extraction result as XML download."""
    payload = _cache_get(export_id)
    if not payload:
        flash("No extraction data to export. Upload again.", "error")
        return redirect(url_for("index"))

    xml_str = build_xml_export(payload)
    first_name = payload["results"][0]["filename"] if payload.get("results") else "extraction"
    safe_name = first_name.rsplit(".", 1)[0] if "." in first_name else first_name
    if len(payload.get("results", [])) > 1:
        safe_name = f"batch_{len(payload['results'])}_files"
    return Response(
        xml_str,
        mimetype="application/xml",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.xml"'},
    )


@app.route("/export/csv/<export_id>")
def export_csv(export_id: str):
    """Export cached results as a CSV file for Excel."""
    payload = _cache_get(export_id)
    if not payload:
        flash("No extraction data to export. Upload again.", "error")
        return redirect(url_for("index"))

    csv_str = build_csv_export(payload)
    first_name = payload["results"][0]["filename"] if payload.get("results") else "extraction"
    safe_name = first_name.rsplit(".", 1)[0] if "." in first_name else first_name
    if len(payload.get("results", [])) > 1:
        safe_name = f"batch_{len(payload['results'])}_files"
    # Use UTF-8 BOM so Excel auto-detects encoding correctly
    bom = b'\xef\xbb\xbf'
    return Response(
        bom + csv_str.encode("utf-8"),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.csv"'},
    )


@app.route("/api/extract", methods=["POST"])
def api_extract():
    """
    REST API endpoint for programmatic access.
    POST a file and get JSON back with extracted fields.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    original_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    try:
        fields = process_image(filepath, TESSERACT_CMD)
        return jsonify({
            "success": True,
            "filename": original_name,
            "fields": fields,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            os.remove(filepath)
        except OSError:
            pass


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") in ("1", "true", "True")

    print("=" * 60)
    print("  Car Registration OCR")
    print("=" * 60)
    print(f"  Port:       {port}")
    print(f"  Debug:      {debug}")
    print(f"  Max files:  {MAX_FILES_PER_REQUEST}")

    if TESSERACT_CMD:
        print(f"  Tesseract:  {TESSERACT_CMD}")
    else:
        print("  Tesseract:  system PATH")

    print("=" * 60)
    app.run(debug=debug, host="0.0.0.0", port=port)
