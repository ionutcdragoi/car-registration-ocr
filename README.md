# Car Registration OCR

A web application that extracts information from car registration document photos using **Tesseract OCR** and saves the data to **Google Sheets**.

## Extracted Fields

| Field | Description |
|---|---|
| License Plate | Vehicle registration number |
| Owner Name | Registered owner |
| Owner Address | Owner's address |
| Vehicle Make | Brand (Toyota, BMW, etc.) |
| Vehicle Model | Model name |
| Vehicle Type | Body type |
| VIN | 17-character chassis number |
| Registration Date | Date of registration |
| First Registration | Original registration date |
| Engine Capacity | Displacement in cc |
| Engine Power | kW and HP |
| Fuel Type | Petrol, Diesel, Electric, etc. |
| Color | Vehicle color |
| Seats | Number of seats |
| Max Weight | Maximum weight in kg |
| CO2 Emissions | g/km |

## Prerequisites

### 1. Install Tesseract OCR

**Windows:**
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (default path: `C:\Program Files\Tesseract-OCR\`)
3. During installation, select additional languages you need (German, French, Spanish, Italian, Romanian, etc.)

**macOS:**
```bash
brew install tesseract
brew install tesseract-lang   # additional languages
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-deu tesseract-ocr-fra tesseract-ocr-spa tesseract-ocr-ita tesseract-ocr-ron
```

### 2. Install Python Dependencies

```bash
cd car-registration-ocr
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment

```bash
copy .env.example .env
```

Edit `.env` and set:
- `TESSERACT_CMD` — path to tesseract executable (Windows only, leave empty if it's in PATH)

### 4. Set Up Google Sheets (Optional)

The app works without Google Sheets — it will show extracted data on screen. To enable saving to Sheets:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or use existing)
3. Enable **Google Sheets API** and **Google Drive API**
4. Go to **Credentials** → **Create Credentials** → **Service Account**
5. Create a key for the service account (JSON format)
6. Download the JSON key file and save it as `credentials.json` in the project root
7. In `.env`, set `GOOGLE_SHEET_NAME` to your preferred sheet name

The app will automatically create the spreadsheet. To access it from your Google account, share the sheet with the service account email (found in `credentials.json` → `client_email`).

## Usage

### Web Interface

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser, upload a photo, and view the results.

### REST API

```bash
curl -X POST -F "file=@registration.jpg" http://127.0.0.1:5000/api/extract
```

Response:
```json
{
  "success": true,
  "filename": "registration.jpg",
  "fields": {
    "license_plate": "B 123 ABC",
    "vin": "WVWZZZ3CZWE123456",
    "vehicle_make": "VOLKSWAGEN",
    ...
  },
  "sheet_url": "https://docs.google.com/spreadsheets/d/..."
}
```

## Project Structure

```
car-registration-ocr/
├── app.py                  # Flask web server
├── ocr_engine.py           # Tesseract OCR + image preprocessing
├── sheets_integration.py   # Google Sheets read/write
├── templates/
│   ├── index.html          # Upload page
│   └── result.html         # Results page
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
└── README.md
```

## Tips for Better OCR Results

- Use high-resolution photos (at least 1000px wide)
- Ensure good lighting, avoid glare and shadows
- Keep the document flat and aligned
- Crop the image to just the registration document
- Avoid blurry or out-of-focus images

## License

MIT
