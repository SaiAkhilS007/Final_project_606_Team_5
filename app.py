from flask import Flask, render_template, request, redirect, session, url_for
from flask_session import Session
from utils import (
    preprocess_image, 
    predict_category, 
    fetch_top_youtube_videos, 
    fetch_nearby_locations
)
from PIL import Image
import os
import googlemaps



# Initialize Google Maps API
gmaps = googlemaps.Client(key="AIzaSyAIP4GUoss0Z8bm6e9j7g4kaoWe0yu-tC8")

# Initialize Flask app
app = Flask(__name__)

# Configure secret key and session
app.secret_key = "your_secret_key"
app.config["SESSION_TYPE"] = "filesystem"  # Store session data in the filesystem
Session(app)

categories = ['Cardboard', 'Plastic', 'Glass', 'Medical', 'Paper',
              'E-Waste', 'Organic Waste', 'Textiles', 'Metal', 'Wood']

import base64
from io import BytesIO

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_category = None
    error_message = None
    uploaded_image = None

    if request.method == 'POST':
        if 'image' not in request.files or request.files['image'].filename == '':
            error_message = "Please upload an image file."
            return render_template('index.html', predicted_category=None, error=error_message, uploaded_image=None)

        try:
            # Process the uploaded image
            image_file = request.files['image']
            image = Image.open(image_file)

            # Convert image to base64 for display
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            uploaded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Predict the category
            processed_image = preprocess_image(image)
            predicted_category = predict_category(processed_image, categories)
            session['category'] = predicted_category  # Save in session for actions
        except Exception as e:
            error_message = f"Error processing the image: {e}"

    return render_template('index.html', predicted_category=predicted_category, error=error_message, uploaded_image=uploaded_image)




@app.route('/action', methods=['GET'])
def action():
    category = session.get('category', None)
    if not category:
        return redirect(url_for('index'))

    action_type = request.args.get('action', None)
    if action_type == "reuse":
        return redirect(url_for('youtube_videos', intent='reuse'))
    elif action_type == "recycle":
        return redirect(url_for('youtube_videos', intent='recycle'))
    elif action_type == "disposal":
        return redirect(url_for('disposal_action'))
    else:
        return redirect(url_for('index'))


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or request.files['image'].filename == '':
        return {'error': 'No file uploaded'}, 400

    try:
        # Load and preprocess the image
        image_file = request.files['image']
        image = Image.open(image_file)
        processed_image = preprocess_image(image)

        # Predict the category
        predicted_category = predict_category(processed_image, categories)

        # Save the category in the session
        session['category'] = predicted_category

        # Return the predicted category as a JSON response
        return {'category': predicted_category}, 200
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_category = None
    error_message = None

    if request.method == 'POST':
        # Mock prediction for demonstration
        session['category'] = request.form.get('category', 'Plastic')  # Replace with your model's output
        return redirect(url_for('home'))

    predicted_category = session.get('category', None)
    return render_template('index.html', predicted_category=predicted_category)


@app.route('/reuse', methods=['GET'])
def reuse_action():
    return redirect(url_for('youtube_videos', intent='reuse'))

@app.route('/recycle', methods=['GET'])
def recycle_action():
    return redirect(url_for('youtube_videos', intent='recycle'))


@app.route('/disposal', methods=['GET', 'POST'])
def disposal_action():
    category = session.get('category', None)
    if not category:
        return redirect(url_for('home'))

    locations = None
    error_message = None

    if request.method == 'POST':
        zip_code = request.form.get('zip_code')
        radius = int(request.form.get('radius', 10)) * 1609  # Convert miles to meters

        def get_coordinates_from_zip(zip_code):
            geocode_result = gmaps.geocode(zip_code)
            if not geocode_result:
                raise ValueError("Invalid ZIP code or no result from Google Maps API")
            location = geocode_result[0]['geometry']['location']
            return location['lat'], location['lng']

        try:
            user_location = get_coordinates_from_zip(zip_code)
            locations = fetch_nearby_locations(zip_code, category, "disposal", radius)
        except ValueError as ve:
            error_message = f"Error: {ve}"
        except Exception as e:
            error_message = "Unexpected error occurred. Please try again."

    return render_template(
        'action.html',
        intent="disposal",
        locations=locations,
        category=category,
        error=error_message
    )

@app.route('/youtube', methods=['GET'])
def youtube_videos():
    intent = request.args.get('intent')
    category = session.get('category', None)
    intent = request.args.get('intent', None)

    if not category or not intent:
        return redirect(url_for('home'))

    videos = fetch_top_youtube_videos(category, intent)
    return render_template('youtube.html', videos=videos, category=category, intent=intent)

if __name__ == '__main__':
    app.run(debug=True)