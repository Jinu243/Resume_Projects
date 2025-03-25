from flask import Flask
from dotenv import load_dotenv
import os

# Load environment variables from .env file (if present)
load_dotenv()

def create_app():
    app = Flask(__name__)
    # Configure upload folder and allowed file extensions
    app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
    app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt'}

    # Ensure the upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Register routes from the routes module
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
