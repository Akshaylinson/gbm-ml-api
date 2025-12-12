# 🚀 Gradient Boosting ML API

A production-ready machine learning API built with FastAPI, featuring gradient boosting models for binary classification with comprehensive monitoring, security, and deployment capabilities.

## 📋 Features

### Core ML Capabilities
- **Gradient Boosting Model**: Scikit-learn based GBM for binary classification
- **Batch Predictions**: Process up to 100 rows per request
- **Model Versioning**: Support for multiple model versions
- **Feature Importance**: SHAP-based model interpretability
- **Confidence Intervals**: Statistical confidence measures

### Production Features
- **Authentication**: API key-based security
- **Rate Limiting**: 100 requests per minute per user
- **Caching**: In-memory caching with TTL
- **Monitoring**: Real-time metrics and drift detection
- **Health Checks**: Comprehensive health endpoints
- **CORS Support**: Cross-origin resource sharing

### Advanced Capabilities
- **A/B Testing**: Model comparison framework
- **Async Processing**: Celery-based background tasks
- **Dashboard**: Streamlit monitoring interface
- **Load Testing**: Locust-based performance testing
- **Security**: Enhanced security with encryption
- **Docker Support**: Full containerization

## 🏗️ Architecture

```
gbm_ml_api/
├── 📁 data/              # Sample datasets
├── 📁 models/            # Trained ML models
├── 📁 k8s/               # Kubernetes deployments
├── 📁 monitoring/        # Prometheus configs
├── 📁 .github/workflows/ # CI/CD pipelines
├── 🐍 app.py            # Main FastAPI application
├── 🐍 dashboard.py      # Streamlit dashboard
├── 🐍 train.py          # Model training script
├── 🐳 Dockerfile        # Container configuration
└── 📄 requirements.txt  # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker (optional)
- Redis (for caching)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd gbm_ml_api
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model** (optional - pre-trained model included)
```bash
python train.py
```

4. **Start the API**
```bash
python app.py
```

The API will be available at `http://localhost:8000`

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run individual containers
docker build -t gbm-api .
docker run -p 8000:8000 gbm-api
```

## 📖 API Usage

### Authentication
All endpoints require an API key in the Authorization header:
```bash
Authorization: Bearer demo-key-123
```

### Make Predictions

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "rows": [
    {
      "age": 35,
      "income": 55000,
      "balance": 1200,
      "city": "A",
      "has_credit_card": 1
    }
  ],
  "explain": true
}
```

**Response:**
```json
{
  "results": [
    {
      "prediction": 1,
      "probability": 0.742,
      "confidence_interval": [0.698, 0.786],
      "risk_score": "HIGH",
      "explanation": {
        "income": 0.35,
        "age": 0.28,
        "balance": 0.22
      }
    }
  ],
  "model_version": "v1",
  "processing_time_ms": 45.2,
  "request_id": "abc123"
}
```

### Other Endpoints

- `GET /health` - Health check
- `GET /metrics` - Performance metrics
- `GET /models` - Available models
- `GET /docs` - Interactive API documentation

## 📊 Monitoring Dashboard

Launch the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

Features:
- Real-time API metrics
- Prediction distribution
- Feature drift detection
- Interactive testing interface
- Performance monitoring

## 🧪 Testing

### Unit Tests
```bash
pytest test_api.py -v
```

### Load Testing
```bash
# Basic load test
python load_test.py

# Advanced load test with Locust
locust -f load_test_advanced.py --host=http://localhost:8000
```

### API Testing
```bash
# Test with sample data
python test_comprehensive.py
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file:
```env
API_KEYS=demo-key-123:demo-user,prod-key-456:prod-user
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost/dbname
MODEL_PATH=./models/gbm_pipeline.pkl
RATE_LIMIT=100
CACHE_TTL=300
```

### Model Configuration
- **Input Features**: age, income, balance, city, has_credit_card
- **Output**: Binary classification (0/1)
- **Model Type**: Gradient Boosting Classifier
- **Preprocessing**: StandardScaler + OneHotEncoder

## 🚀 Deployment

### Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
```

### Docker Compose
```bash
docker-compose -f docker-compose-ultimate.yml up
```

### Cloud Deployment
The API is ready for deployment on:
- AWS ECS/EKS
- Google Cloud Run/GKE
- Azure Container Instances/AKS
- Heroku

## 📈 Performance

- **Latency**: ~45ms average response time
- **Throughput**: 100+ requests/second
- **Scalability**: Horizontal scaling with load balancer
- **Caching**: 5-minute TTL for identical requests

## 🔒 Security Features

- API key authentication
- Rate limiting per user
- Input validation and sanitization
- CORS configuration
- Request size limits
- Error handling without data leakage

## 🔍 Monitoring & Observability

- **Metrics**: Request count, error rate, response time
- **Drift Detection**: Statistical feature drift monitoring
- **Health Checks**: Liveness and readiness probes
- **Logging**: Structured logging with correlation IDs
- **Prometheus**: Metrics export for monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

- **Documentation**: `/docs` endpoint for interactive API docs
- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions

## 🔄 Version History

- **v2.0**: Advanced features, monitoring, security
- **v1.0**: Basic ML API with FastAPI

---

**Built with ❤️ using FastAPI, Scikit-learn, and modern MLOps practices**
