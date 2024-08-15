from flask import Flask, request, jsonify
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model("image_classifier_model_simplified.h5")

# Define image dimensions
img_width, img_height = 150, 150

# Define class labels
class_labels = [
    'Anemone_flower', 'Asparagus', 'Banana', 'Chamomile', 'Chickweed', 'Daffodil_flower', 
    'Daisy Fleabane', 'Echinacea', 'Hyacinth_flower', 'Jonquil_flower', 'Lisianthus_flower', 
    'Madagascar_Periwinkle', 'Mini-Carnation_purple', 'Mustard', 'Pickerelweed_flower', 
    'Poinsettia_flower', 'Pompon_flower', 'Primrose_blue', 'Protea', 'Purple_Deadnettle_flower', 
    'Ranunculus_flower', 'Rose_hips', 'Trachelium_flower', 'Vervain_Mallow_flower', 'Waxflower', 
    'Wild Grape Vine', 'Wild Leek', 'aeonium', 'agapanthus', 'almond', 'aloe_vera', 
    'amaryllis_flower', 'aster', 'baby_breath_flower', 'black_rose_flower', 'blue_chicory', 
    'blue_vervain', 'bougainvillea_flower', 'bromeliad', 'buttercup_flower', 'calendula_flower', 
    'canna', 'cannabis_flower', 'carex', 'cattails', 'coconut_', 'cone_flower', 'coronation_gold', 
    'crimson_clover', 'crocus_blue', 'daisy', 'dandelion', 'datura_flower', 'delonix_regia_flower', 
    'downy_yellow_violet', 'elderberry_flower', 'fireweed_flower', 'forget_me_not', 
    'golden_champa_flower', 'harebell_flower', 'hibiscus_flower', 'iris_flower', 'jasmine_flower', 
    'joe_pye_weed', 'knapweed', 'larkspur_flower', 'lily_flower', 'lotus_flower', 'mallow_flower', 
    'marigold_flower', 'milk_thistle_flower', 'mullein_flower_yellow', 'mushroom', 
    'narcissistic_flower', 'oleander_flower', 'orchid_flower', 'palash_flower', 'parlor_palm', 
    'prickly_pear_cactus', 'queen_anne_s_lace_flower', 'red-hot_poker', 'red_clover', 'red_rose', 
    'saffron_flower', 'sedum_purple', 'solidago_flower', 'st_john', 'statice_flower', 'sunflower', 
    'teasel_flower', 'touch_me_not_flower', 'tuberose_flower', 'tulip_flower', 'viola_flower', 
    'white_yarrow', 'wild_bee_balm_flowers', 'yellow_sow_thistle', 'yucca', 'zinnia_flower_red'
]

def prepare_image_from_file(file):
    """Preprocess the image from the uploaded file."""
    try:
        img = load_img(file, target_size=(img_width, img_height))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image file: {e}")

def prepare_image_from_url(url):
    """Download and preprocess the image from the given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = load_img(BytesIO(response.content), target_size=(img_width, img_height))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error fetching image from URL: {e}")

def predict_image_class_from_file(file):
    """Predict the class of an uploaded image file."""
    prepared_image = prepare_image_from_file(file)
    predictions = model.predict(prepared_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_class_label, confidence

def predict_image_class_from_url(image_url):
    """Predict the class of an image given its URL."""
    prepared_image = prepare_image_from_url(image_url)
    predictions = model.predict(prepared_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_class_label, confidence

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_base64 = data['image']
        image_data = base64.b64decode(image_base64)
        img = load_img(BytesIO(image_data), target_size=(img_width, img_height))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_label = class_labels[predicted_class_index]
        confidence = predictions[0][predicted_class_index]
        
        return jsonify({
            "predicted_class": predicted_class_label,
            "confidence": float(confidence)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/status', methods=['GET'])
def status():
    """A simple endpoint to check the status of the server and predict image from URL."""
    image_url = "https://i.pinimg.com/originals/9e/59/18/9e59182ee4c635f3d77aacb517b87d17.jpg"
    if image_url:
        try:
            predicted_label, confidence = predict_image_class_from_url(image_url)
            return jsonify({
                "status": "server is running",
                "predicted_class": predicted_label,
                "confidence": float(confidence)  # Convert to standard Python float
            })
        except Exception as e:
            return jsonify({
                "status": "server is running",
                "error": str(e)
            }), 400
    else:
        return jsonify({"status": "server is running"})

# Run the Flask app
if __name__ == '__main__':
    # Note: Disable debug mode in production for security reasons
    app.run(debug=True)
