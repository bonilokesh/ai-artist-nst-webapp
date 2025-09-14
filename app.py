
NGROK_AUTH_TOKEN = "add your nrok auth token"

import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask import Flask, request, send_file
from pyngrok import ngrok
from flask_cors import CORS 



def load_img_from_bytes(image_bytes):
    """Loads an image from bytes and limits its maximum dimension to 512 pixels."""
    max_dim = 512
    img = tf.io.decode_image(image_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image_bytes(tensor):
    """Converts a tensor back to image bytes."""
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert len(tensor.shape) == 4
        tensor = tensor[0]
    pil_image = Image.fromarray(tensor)
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='JPEG')
    byte_arr.seek(0)
    return byte_arr

def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}

# --- Main Style Transfer Function ---
def perform_style_transfer(content_image_bytes, style_image_bytes):
    print("Loading images...")
    content_image = load_img_from_bytes(content_image_bytes)
    style_image = load_img_from_bytes(style_image_bytes)

    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    image_to_optimize = tf.Variable(content_image)
    optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    style_weight = 1e-2
    content_weight = 1e4
    
    epochs = 5
    steps_per_epoch = 50

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            style_outputs = outputs['style']
            content_outputs = outputs['content']
            style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) for name in style_outputs.keys()])
            style_loss *= style_weight / len(style_layers)
            content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs.keys()])
            content_loss *= content_weight / len(content_layers)
            loss = style_loss + content_loss
        grad = tape.gradient(loss, image)
        optimizer.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

    print("Starting optimization...")
    for n in range(epochs):
        for m in range(steps_per_epoch):
            train_step(image_to_optimize)
        print(f"Epoch {n+1}/{epochs} complete.")

    return tensor_to_image_bytes(image_to_optimize)


app = Flask(__name__)
CORS(app) 

@app.route('/')
def index():
    return "<h1>Style Transfer API</h1><p>POST a content and style image to the /stylize endpoint.</p>"

@app.route('/stylize', methods=['POST'])
def stylize():
    if 'content_file' not in request.files or 'style_file' not in request.files:
        return "Missing content_file or style_file", 400
    
    content_file = request.files['content_file']
    style_file = request.files['style_file']

    if content_file.filename == '' or style_file.filename == '':
        return "No selected file", 400

    content_bytes = content_file.read()
    style_bytes = style_file.read()
    
    print("Received images. Starting style transfer process...")
    result_bytes = perform_style_transfer(content_bytes, style_bytes)
    print("Process complete. Sending back image.")
    return send_file(result_bytes, mimetype='image/jpeg')

if __name__ == '__main__':
    if NGROK_AUTH_TOKEN == "YOUR_NGROK_AUTHTOKEN_HERE":
        print("ERROR: Please set your ngrok authtoken in the NGROK_AUTH_TOKEN variable.")
    else:
        # Set the authtoken for ngrok
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        
        # Start the ngrok tunnel
        port = 5000
        public_url = ngrok.connect(port)
        print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")
        print(f" * Use this URL for your frontend: {public_url}")
        
        print("Starting Flask server...")
        app.run(port=port)


