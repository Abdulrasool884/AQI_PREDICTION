"""Small Flask interface for the saved AQI prediction pipeline."""

from pathlib import Path
import math

import joblib
import pandas as pd
import requests
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "aqi_model.pkl"

FEATURES = [
    "City", "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2",
    "O3", "Benzene", "Toluene", "Xylene",
]
POLLUTANTS = FEATURES[1:]
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
AIR_QUALITY_VARIABLES = [
    "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone",
]

app = Flask(__name__)

# Load once when the web application starts, instead of once per form submission.
try:
    model = joblib.load(MODEL_PATH)
    model_load_error = None
except Exception as exc:
    model = None
    model_load_error = str(exc)

if model is not None:
    # Read the city categories already stored in the fitted encoder.
    city_encoder = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
    TRAINING_CITIES = set(city_encoder.categories_[0])
else:
    TRAINING_CITIES = set()


def aqi_category(aqi):
    """Return the India AQI band label for a numeric AQI value."""
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def predict_aqi(city, values):
    """Run the saved pipeline with all columns in the training feature order."""
    if model is None:
        raise RuntimeError(f"The saved model could not be loaded: {model_load_error}")

    # The model pipeline's imputer can handle fields that the API does not provide.
    input_row = {"City": city, **values}
    input_data = pd.DataFrame([input_row], columns=FEATURES)
    prediction = float(model.predict(input_data)[0])
    if not pd.notna(prediction):
        raise ValueError("The model returned a non-numeric prediction.")
    return max(0.0, prediction)


def get_open_meteo_data(city_query):
    """Search Indian cities, towns, and villages and return matching locations."""
    try:
        place_response = requests.get(
            GEOCODING_URL,
            params={"name": city_query, "count": 10, "language": "en", "countryCode": "IN"},
            timeout=10,
        )
        place_response.raise_for_status()
        results = place_response.json().get("results", [])
        for place in results:
            parts = [place.get("name"), place.get("admin1"), place.get("country")]
            place["display_name"] = ", ".join(part for part in parts if part)
        return results
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not contact Open-Meteo Geocoding: {exc}") from exc


def validate_coordinates(latitude_text, longitude_text):
    """Convert and validate manually entered latitude and longitude."""
    try:
        latitude = float(latitude_text)
        longitude = float(longitude_text)
    except (TypeError, ValueError) as exc:
        raise ValueError("Latitude and longitude must both be numbers.") from exc

    if not math.isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")
    if not math.isfinite(longitude) or not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180.")
    return latitude, longitude


def fetch_air_quality(latitude, longitude):
    """Fetch current pollutants and prepare inputs for the original pipeline."""
    try:
        air_response = requests.get(
            AIR_QUALITY_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": ",".join(AIR_QUALITY_VARIABLES),
                "timezone": "auto",
            },
            timeout=10,
        )
        air_response.raise_for_status()
        current = air_response.json().get("current")
        if not current or not current.get("time"):
            raise ValueError("The air-quality service did not return current conditions for this location.")

        # Preserve the original model inputs, order, and CO conversion.
        api_values = {
            "PM2.5": current.get("pm2_5"),
            "PM10": current.get("pm10"),
            "NO2": current.get("nitrogen_dioxide"),
            "SO2": current.get("sulphur_dioxide"),
            "CO": current.get("carbon_monoxide"),
            "O3": current.get("ozone"),
        }
        values = {name: float("nan") for name in POLLUTANTS}
        for feature_name, api_value in api_values.items():
            values[feature_name] = float(api_value) if api_value is not None else float("nan")
        if pd.notna(values["CO"]):
            values["CO"] = values["CO"] / 1000.0
        return current["time"], api_values, values
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not contact Open-Meteo Air Quality: {exc}") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html", pollutants=POLLUTANTS, error=None, location_results=None)

    city = request.form.get("City", "").strip()
    form_values = {name: request.form.get(name, "").strip() for name in POLLUTANTS}
    if not city:
        return render_template("index.html", pollutants=POLLUTANTS, error="Please select a city."), 400

    values = {}
    for name, raw_value in form_values.items():
        if not raw_value:
            return render_template(
                "index.html", pollutants=POLLUTANTS,
                error=f"Please enter a value for {name}.",
            ), 400
        try:
            number = float(raw_value)
        except ValueError:
            return render_template(
                "index.html", pollutants=POLLUTANTS,
                error=f"{name} must be a number.",
            ), 400
        if not pd.notna(number) or number < 0:
            return render_template(
                "index.html", pollutants=POLLUTANTS,
                error=f"{name} must be a valid, non-negative number.",
            ), 400
        values[name] = number

    try:
        prediction = predict_aqi(city, values)
    except Exception as exc:
        return render_template(
            "index.html", pollutants=POLLUTANTS,
            error=f"Prediction failed: {exc}",
        ), 500

    prediction = max(0.0, prediction)
    return render_template(
        "result.html", city=city, prediction=prediction,
        category=aqi_category(prediction), mode="manual", timestamp=None,
        location_label=city, api_pollutants=None,
    )


@app.route("/live-predict", methods=["POST"])
def live_predict():
    city_query = request.form.get("city_search", "").strip()
    latitude_text = request.form.get("latitude", "").strip()
    longitude_text = request.form.get("longitude", "").strip()
    location_choice = request.form.get("location_choice") == "selected"

    # If no coordinates were supplied, search by name and let the user choose a match.
    if not latitude_text and not longitude_text:
        if not city_query:
            return render_template(
                "index.html", pollutants=POLLUTANTS, location_results=None,
                error="Enter a city, town, village, or both coordinates.",
            ), 400
        try:
            locations = get_open_meteo_data(city_query)
        except Exception as exc:
            return render_template(
                "index.html", pollutants=POLLUTANTS, location_results=None,
                error=f"Location search failed: {exc}",
            ), 502
        if not locations:
            return render_template(
                "index.html", pollutants=POLLUTANTS, location_results=None,
                error=f"No Indian location matched '{city_query}'. Try another spelling or enter coordinates.",
            ), 404
        return render_template(
            "index.html", pollutants=POLLUTANTS, error=None,
            location_results=locations, search_query=city_query,
        )

    if not latitude_text or not longitude_text:
        return render_template(
            "index.html", pollutants=POLLUTANTS, location_results=None,
            error="Enter both latitude and longitude, or leave both blank to search by name.",
        ), 400

    try:
        latitude, longitude = validate_coordinates(latitude_text, longitude_text)
    except ValueError as exc:
        return render_template(
            "index.html", pollutants=POLLUTANTS, location_results=None,
            error=str(exc),
        ), 400

    if location_choice:
        city = request.form.get("selected_city", "").strip() or "Selected location"
        location_label = request.form.get("selected_label", "").strip() or city
    else:
        city = request.form.get("manual_location_name", "").strip() or "Manually entered location"
        location_label = f"Manually entered location: {city} ({latitude:.5f}, {longitude:.5f})"

    try:
        timestamp, api_pollutants, model_values = fetch_air_quality(latitude, longitude)
        prediction = predict_aqi(city, model_values)
    except Exception as exc:
        return render_template(
            "index.html", pollutants=POLLUTANTS, location_results=None,
            error=f"Live lookup or prediction failed: {exc}",
        ), 502

    api_received = [name for name, value in api_pollutants.items() if value is not None]
    imputed_pollutants = [name for name in POLLUTANTS if pd.isna(model_values[name])]
    return render_template(
        "result.html", city=city, prediction=prediction,
        category=aqi_category(prediction), mode="live", timestamp=timestamp,
        location_label=location_label, api_pollutants=api_pollutants,
        api_received=api_received, imputed_pollutants=imputed_pollutants,
        outside_training=city not in TRAINING_CITIES,
        latitude=latitude, longitude=longitude,
    )


if __name__ == "__main__":
    # debug=True is convenient for local learning; do not use it for deployment.
    app.run(debug=True)
