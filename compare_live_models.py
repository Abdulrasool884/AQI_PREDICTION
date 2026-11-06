"""Compare the original and six-pollutant AQI pipelines on one live API sample.

This script is diagnostic: it does not modify or retrain either saved model.
"""

import math
from pathlib import Path

import joblib
import pandas as pd
import requests


PROJECT_DIR = Path(__file__).resolve().parent
ORIGINAL_MODEL_PATH = PROJECT_DIR / "models" / "aqi_model.pkl"
LIVE_MODEL_PATH = PROJECT_DIR / "models" / "aqi_model_live.pkl"

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# These are the six measurements returned by Open-Meteo and used by both models.
API_TO_MODEL_FEATURES = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "nitrogen_dioxide": "NO2",
    "carbon_monoxide": "CO",
    "sulphur_dioxide": "SO2",
    "ozone": "O3",
}

# The original model has these additional features, which this API request does not provide.
ORIGINAL_ONLY_POLLUTANTS = ["NO", "NOx", "NH3", "Benzene", "Toluene", "Xylene"]


def read_coordinate(prompt, minimum, maximum):
    """Ask again until the user enters a finite coordinate in range."""
    while True:
        raw_value = input(prompt).strip()
        try:
            value = float(raw_value)
        except ValueError:
            print("Please enter a number.")
            continue

        if not math.isfinite(value) or not minimum <= value <= maximum:
            print(f"Enter a value from {minimum} to {maximum}.")
            continue
        return value


def get_live_data(city_query):
    """Find an Indian city or ask for coordinates, then fetch current pollutants."""
    geocoding_response = requests.get(
        GEOCODING_URL,
        params={
            "name": city_query,
            "count": 1,
            "language": "en",
            "countryCode": "IN",
        },
        timeout=15,
    )
    geocoding_response.raise_for_status()
    results = geocoding_response.json().get("results", [])
    if not results:
        print(f"No Indian location matched '{city_query}'.")
        print("Enter the location coordinates manually (decimal degrees).")
        latitude = read_coordinate("Latitude (-90 to 90): ", -90, 90)
        longitude = read_coordinate("Longitude (-180 to 180): ", -180, 180)
        city_name = city_query
        location_label = f"Manually entered location for '{city_query}'"
    else:
        # Like the website, use the first matching result. The printed location
        # and coordinates let you confirm which match was selected.
        place = results[0]
        latitude = place["latitude"]
        longitude = place["longitude"]
        city_name = place["name"]
        location_parts = [place.get("name"), place.get("admin1"), place.get("country")]
        location_label = ", ".join(part for part in location_parts if part)

    air_response = requests.get(
        AIR_QUALITY_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": ",".join(API_TO_MODEL_FEATURES.keys()),
            "timezone": "auto",
        },
        timeout=15,
    )
    air_response.raise_for_status()
    current = air_response.json().get("current")
    if not current or not current.get("time"):
        raise ValueError("Open-Meteo did not return current air-quality data for this location.")

    # Open-Meteo reports these concentrations in μg/m³. The training CSV's CO
    # feature is in mg/m³, so convert CO before passing it to either pipeline.
    pollutant_values = {}
    for api_name, feature_name in API_TO_MODEL_FEATURES.items():
        value = current.get(api_name)
        pollutant_values[feature_name] = float(value) if value is not None else float("nan")

    if pd.notna(pollutant_values["CO"]):
        pollutant_values["CO"] /= 1000.0

    return city_name, location_label, latitude, longitude, current["time"], pollutant_values


def print_imputed_medians(model, raw_input, model_label):
    """Show the fitted numeric medians for any NaN values in this input row."""
    preprocessor = model.named_steps["preprocessor"]
    replacements = []

    for _, transformer, columns in preprocessor.transformers_:
        if not hasattr(transformer, "named_steps"):
            continue
        imputer = transformer.named_steps.get("imputer")
        if imputer is None or not hasattr(imputer, "statistics_"):
            continue

        for column, median in zip(columns, imputer.statistics_):
            if pd.isna(raw_input.iloc[0][column]):
                replacements.append({"Feature": column, "Pipeline replacement": median})

    if replacements:
        print(f"\n{model_label} fitted imputation values for missing inputs:")
        print(pd.DataFrame(replacements).to_string(index=False))
    else:
        print(f"\n{model_label}: no missing numeric input values needed imputation.")


def main():
    """Fetch one sample, prepare both feature schemas, and print predictions."""
    for model_path in [ORIGINAL_MODEL_PATH, LIVE_MODEL_PATH]:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

    original_model = joblib.load(ORIGINAL_MODEL_PATH)
    live_model = joblib.load(LIVE_MODEL_PATH)

    print("AQI model comparison using one Open-Meteo sample")
    print("This is an experimental model comparison, not official AQI.")
    city_query = input("Enter a city or town in India: ").strip()
    if not city_query:
        raise ValueError("Please enter a location name.")

    city_name, location, latitude, longitude, timestamp, pollutants = get_live_data(city_query)

    print(f"\nSelected location: {location}")
    print(f"Coordinates used: latitude {latitude}, longitude {longitude}")
    print(f"Open-Meteo data timestamp: {timestamp}")
    print("\nAvailable API pollutant values used by both models:")
    print("Concentrations are μg/m³ except CO, converted to mg/m³ for model input.")
    print(pd.DataFrame([pollutants]).to_string(index=False))

    # The 13-feature pipeline receives the same six API readings plus NaN for
    # each of its six additional pollutant features.
    original_values = {"City": city_name}
    original_values.update(pollutants)
    original_values.update({name: float("nan") for name in ORIGINAL_ONLY_POLLUTANTS})
    original_features = list(original_model.feature_names_in_)
    original_input = pd.DataFrame([original_values], columns=original_features)

    # The seven-input pipeline gets City and only those same six pollutant values.
    live_values = {"City": city_name, **pollutants}
    live_features = list(live_model.feature_names_in_)
    live_input = pd.DataFrame([live_values], columns=live_features)

    print("\nRaw inputs passed to the original 13-feature model (NaN values are imputed by its pipeline):")
    print(original_input.to_string(index=False))
    print_imputed_medians(original_model, original_input, "Original 13-feature model")

    print("\nInputs passed to the new 7-input model (City plus six pollutants):")
    print(live_input.to_string(index=False))
    print_imputed_medians(live_model, live_input, "New 7-input model")

    original_prediction = float(original_model.predict(original_input)[0])
    live_prediction = float(live_model.predict(live_input)[0])

    print("\nPredictions (experimental ML estimates; neither is official AQI):")
    print(f"Original 13-feature model: {original_prediction:.2f}")
    print(f"New 7-input model:         {live_prediction:.2f}")
    print("\nThe original model imputes NO, NOx, NH3, Benzene, Toluene, and Xylene.")
    print("The new model uses City plus PM2.5, PM10, NO2, CO, SO2, and O3.")
    print("Different predictions are possible because the two models were trained with different feature sets.")
    print("These values are not guaranteed accurate and should not be treated as CPCB or Google AQI.")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"Open-Meteo request failed: {exc}")
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Could not complete comparison: {exc}")
