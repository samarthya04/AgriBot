from flask import Flask, render_template, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_caching import Cache
from dotenv import load_dotenv
import structlog
import os

from config import DevelopmentConfig
from models import db, migrate
from api.plant_id import identify_plant
from api.plantnet import identify_species
from api.perenual import get_plant_info
from api.local_analysis import local_analysis_fallback
from api.chat import enhanced_chat
from api.upload import upload_image
from api.status import check_status

# Enhanced logging configuration
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

# Load environment variables
load_dotenv()

# Initialize Flask app
def create_app(config_class=DevelopmentConfig):
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    cache.init_app(app)
    limiter.init_app(app)
    CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])

    # Register blueprints
    app.register_blueprint(identify_plant)
    app.register_blueprint(identify_species)
    app.register_blueprint(get_plant_info)
    app.register_blueprint(enhanced_chat)
    app.register_blueprint(upload_image)
    app.register_blueprint(local_analysis_fallback)
    app.register_blueprint(check_status)

    # Routes
    @app.route("/")
    @limiter.limit("100 per minute")
    def index():
        return render_template("index.html")

    # Error handlers
    @app.errorhandler(429)
    def ratelimit_handler(e):
        logger.warning("Rate limit exceeded", remote_addr=get_remote_address())
        return jsonify({"error": "Rate limit exceeded. Please try again later.", "retry_after": str(e.retry_after)}), 429

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

    @app.errorhandler(500)
    def internal_error(e):
        logger.error("Internal server error", error=str(e))
        return jsonify({"error": "Internal server error occurred."}), 500

    # Health check
    @app.route("/health")
    def health_check():
        try:
            db.session.execute('SELECT 1')
            redis_client = cache.cache._client
            redis_client.ping()
            api_statuses = []
            for api_name in ["plant_id", "plantnet", "perenual"]:
                status = db.session.query(db.models.APIStatus).filter_by(api_name=api_name).first()
                api_statuses.append({
                    "name": api_name,
                    "status": status.status if status else False,
                    "last_checked": status.last_checked.isoformat() if status else None
                })
            return jsonify({
                "status": "healthy",
                "timestamp": db.func.now().isoformat(),
                "database": "connected",
                "cache": "connected",
                "apis": api_statuses
            })
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return jsonify({"status": "unhealthy", "error": str(e)}), 503

    # Database initialization
    @app.before_first_request
    def create_tables():
        db.create_all()
        logger.info("Database tables initialized")

    return app

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
cache = Cache()
limiter = Limiter(key_func=get_remote_address, default_limits=["1000 per hour", "100 per minute"])

# Create app
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)