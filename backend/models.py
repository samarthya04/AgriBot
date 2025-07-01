from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    region = db.Column(db.String(50), nullable=False)
    language = db.Column(db.String(20), default='english')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    queries = db.relationship('Query', backref='user', lazy=True)
    analyses = db.relationship('PlantAnalysis', backref='user', lazy=True)

class Query(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    region = db.Column(db.String(50), nullable=False)
    language = db.Column(db.String(20), nullable=False)
    response_time_ms = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PlantAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    image_hash = db.Column(db.String(64), nullable=False, index=True)
    plant_name = db.Column(db.String(200))
    scientific_name = db.Column(db.String(200))
    confidence_score = db.Column(db.Float)
    analysis_type = db.Column(db.String(50))
    diseases_detected = db.Column(db.JSON)
    treatment_recommendations = db.Column(db.Text)
    region = db.Column(db.String(50))
    api_source = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class APIStatus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    api_name = db.Column(db.String(50), nullable=False)
    status = db.Column(db.Boolean, default=True)
    last_checked = db.Column(db.DateTime, default=datetime.utcnow)
    response_time_ms = db.Column(db.Integer)
    error_message = db.Column(db.Text)