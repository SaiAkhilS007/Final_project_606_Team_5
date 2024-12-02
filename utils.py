import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.models import load_model
import requests
import googlemaps
from geopy.distance import geodesic

# Load Model and ResNet50 Feature Extractor
MODEL_PATH = "model/Resnet_Neural_Network_model.h5"
model = load_model(MODEL_PATH)
feature_extractor = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Initialize Google Maps API
gmaps = googlemaps.Client(key="AIzaSyAIP4GUoss0Z8bm6e9j7g4kaoWe0yu-tC8")

def preprocess_image(img):
    """
    Preprocess the image for prediction.
    """
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def predict_category(image, categories):
    """
    Predict the waste category.
    """
    features = feature_extractor.predict(image)
    predictions = model.predict(features)
    predicted_label = np.argmax(predictions, axis=1)[0]
    return categories[predicted_label]


def fetch_top_youtube_videos(waste_category, intent):
    youtube_api_key = "AIzaSyAqrbUiRO5WD800M8vnJLbPxKVd2gl6SzE"
    query = f"{waste_category} {intent} tutorial"
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=3&q={query}&type=video&key={youtube_api_key}"
    response = requests.get(url)
    results = response.json().get("items", [])
    return [
        {
            "title": item["snippet"]["title"],
            "link": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
            "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
        }
        for item in results
    ]


def fetch_nearby_locations(zip_code, category, intent, radius):
    try:
        geocode_result = gmaps.geocode(zip_code)
        if not geocode_result:
            raise ValueError("Invalid ZIP code or no data from geocoding API")
        
        user_location = geocode_result[0]['geometry']['location']
        keyword = f"{category} {intent} center"
        places = gmaps.places_nearby(location=user_location, radius=radius, keyword=keyword)
        
        results = places.get("results", [])
        if not results:
            return []  # Return empty list if no locations found

        # Return only the top 3 locations
        return [
            {
                "name": place["name"],
                "address": place["vicinity"],
                "distance": geodesic(
                    (user_location['lat'], user_location['lng']),
                    (place["geometry"]["location"]["lat"], place["geometry"]["location"]["lng"])
                ).miles,
            }
            for place in results[:3]  # Get only the top 3 results
        ]
    except Exception as e:
        print("Error fetching locations:", e)  # Debugging info
        raise ValueError("Could not fetch nearby locations. Please try again.")

