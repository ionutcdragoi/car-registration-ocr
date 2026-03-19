"""
Google Sheets integration for storing extracted car registration data.
Uses gspread with a Service Account for authentication.
"""

import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from typing import Optional

# Google Sheets API scopes
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Column headers matching the extracted fields
HEADERS = [
    "Timestamp",
    "License Plate",
    "Owner Name",
    "Owner Address",
    "Vehicle Make",
    "Vehicle Model",
    "Vehicle Type",
    "VIN",
    "Registration Date",
    "First Registration Date",
    "Engine Capacity (cc)",
    "Engine Power (kW)",
    "Engine Power (HP)",
    "Fuel Type",
    "Color",
    "Number of Seats",
    "Max Weight (kg)",
    "CO2 Emissions",
    "Mileage",
    "Image Filename",
]

# Map from extracted field keys to header positions
FIELD_KEY_ORDER = [
    "license_plate",
    "owner_name",
    "owner_address",
    "vehicle_make",
    "vehicle_model",
    "vehicle_type",
    "vin",
    "registration_date",
    "first_registration_date",
    "engine_capacity_cc",
    "engine_power_kw",
    "engine_power_hp",
    "fuel_type",
    "color",
    "num_seats",
    "max_weight_kg",
    "co2_emissions",
    "mileage",
]


class GoogleSheetsClient:
    """Manages connection and operations with Google Sheets."""

    def __init__(
        self,
        credentials_file: Optional[str] = None,
        sheet_name: str = "Car Registrations",
        spreadsheet_url: Optional[str] = None,
        credentials_info: Optional[dict] = None,
    ):
        """
        Initialize the client.

        Provide EITHER ``credentials_file`` (path) OR ``credentials_info`` (dict
        parsed from the service-account JSON).  ``credentials_info`` takes
        priority so that user-uploaded keys never touch disk.

        Args:
            credentials_file: Path to the Google service account JSON key file.
            sheet_name: Name of the Google Sheet to use.
            spreadsheet_url: Existing Google Sheets URL to open directly.
            credentials_info: Parsed service-account JSON dict (in-memory).
        """
        self.credentials_file = credentials_file
        self._credentials_info = credentials_info
        self.sheet_name = sheet_name
        self._target_url = spreadsheet_url
        self._client = None
        self._spreadsheet = None
        self._worksheet = None

    def _connect(self):
        """Authenticate and open (or create) the spreadsheet."""
        if self._client is None:
            if self._credentials_info:
                creds = Credentials.from_service_account_info(
                    self._credentials_info, scopes=SCOPES
                )
            else:
                creds = Credentials.from_service_account_file(
                    self.credentials_file, scopes=SCOPES
                )
            self._client = gspread.authorize(creds)

        if self._target_url:
            self._spreadsheet = self._client.open_by_url(self._target_url)
        else:
            # Try to open existing sheet, create if not found
            try:
                self._spreadsheet = self._client.open(self.sheet_name)
            except gspread.SpreadsheetNotFound:
                self._spreadsheet = self._client.create(self.sheet_name)
                print(f"Created new spreadsheet: '{self.sheet_name}'")

        self._worksheet = self._spreadsheet.sheet1

        # Ensure headers are present
        existing = self._worksheet.row_values(1)
        if not existing or existing[0] != HEADERS[0]:
            self._worksheet.update("A1", [HEADERS])
            # Format header row (bold)
            self._worksheet.format("A1:T1", {
                "textFormat": {"bold": True},
                "backgroundColor": {"red": 0.2, "green": 0.5, "blue": 0.8},
                "horizontalAlignment": "CENTER",
            })
            print("Initialized sheet with headers.")

    @property
    def spreadsheet_url(self) -> str:
        """Return the URL of the spreadsheet."""
        if self._spreadsheet is None:
            self._connect()
        return self._spreadsheet.url

    def append_record(self, fields: dict, filename: str = "") -> int:
        """
        Append a row of extracted car registration data to the sheet.

        Args:
            fields: Dictionary of extracted fields from ocr_engine.
            filename: Original image filename for reference.

        Returns:
            The row number where the data was inserted.
        """
        if self._worksheet is None:
            self._connect()

        row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]

        for key in FIELD_KEY_ORDER:
            row.append(fields.get(key, ""))

        row.append(filename)  # Image filename

        self._worksheet.append_row(row, value_input_option="USER_ENTERED")

        # Get the row number
        all_values = self._worksheet.get_all_values()
        return len(all_values)

    def get_all_records(self) -> list[dict]:
        """Retrieve all records from the sheet."""
        if self._worksheet is None:
            self._connect()
        return self._worksheet.get_all_records()

    def get_sheet_info(self) -> dict:
        """Return basic info about the sheet."""
        if self._worksheet is None:
            self._connect()

        all_values = self._worksheet.get_all_values()
        return {
            "url": self._spreadsheet.url,
            "title": self._spreadsheet.title,
            "row_count": len(all_values) - 1 if len(all_values) > 0 else 0,  # Exclude header
        }
