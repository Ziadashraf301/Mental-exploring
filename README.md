# Mental-exploring

## Introduction
This repository contains a comprehensive mental health analysis system built as a graduation project. The project leverages machine learning and deep learning to analyze mental health through three core services: depression detection, sentiment analysis, and emotion detection. The system includes a complete MLOps pipeline with Apache Airflow for orchestration, MLflow for experiment tracking, and a FastAPI backend for serving predictions.

![Mental Health Analysis Architecture](https://github.com/Ziadashraf301/Mental-exploring/assets/111798631/bd3fd80e-b057-49f8-a61b-e4cc9dfe1cff)

## Goal

The Mental Health Exploring project is dedicated to promoting mental health awareness and providing individuals with a safe and accessible platform to explore their emotions and mental well-being. Our primary objective is to break down barriers to mental health care by creating a non-judgmental space where people can learn about their sentiments, mental disorders, and emotions without fear of stigmatization.

Our team has developed three machine learning and deep learning models:

1. **Depression Detection**: Analyzes text data to identify patterns and indicators of depression, enabling early intervention
2. **Sentiment Analysis**: Understands people's emotions and attitudes towards specific topics
3. **Emotion Detection**: Analyzes facial expressions in images to detect emotions

## Project Architecture

### System Overview
The project consists of three main components:

1. **ML Pipelines (Airflow)**: Orchestrates data loading, preprocessing, feature engineering, and model training
2. **API Layer (FastAPI)**: Serves predictions through RESTful endpoints
3. **MLOps Infrastructure**: Uses MLflow for experiment tracking and model registry, with AWS for deployment

### Architecture Diagrams
#### 1. API Application Architecture

![API Architecture](docs/Application_Architecture.png)

The API follows a layered architecture pattern:

**Presentation Layer (UI)**:
- Frontend built with Bootstrap, jQuery, HTML5, CSS3, and JavaScript
- Provides user interface for interacting with the mental health analysis services

**Application Layer**:
- **Web Server**: Uvicorn ASGI server handling HTTP requests
- **Routing Logic (REST API)**: FastAPI framework managing endpoints with GET and POST methods
  - Main application entry point: `main.py`
  - Route definitions: `routers/*.py`
  - Service integrations: `services/*.py`
- **Pydantic Models**: Data validation and serialization
  - Request/Response schemas: `schemas.py`
- **SQLAlchemy ORM**: Database abstraction layer
  - Models: User, Prediction tables
  - Session management and connection pooling
  - Database operations: `database.py`

**Database Layer**:
- PostgreSQL database for storing user data, predictions, and application state
- Handles persistent storage of all prediction results and user information

**Data Flow**:
1. Client sends POST request with data (text/image) to API endpoint
2. Request passes through middleware and reaches appropriate router
3. Router validates input using Pydantic models
4. Service layer loads trained model and performs inference
5. Prediction result is stored in PostgreSQL via SQLAlchemy ORM
6. Response is sent back to client with prediction results

#### 2. MLOps and Deployment Architecture

![MLOps Architecture](docs/MLOps_Deployment_Architecture.png)

The MLOps pipeline implements a complete CI/CD workflow:

**Development & Training (Local Environment)**:
- **Apache Airflow**: Orchestrates ML pipelines with dedicated DAGs for each model
  - Data loading → Preprocessing → Feature extraction → Model training
  - Automated scheduling and monitoring of pipeline tasks
  - Connects to PostgreSQL for metadata storage
- **MLflow**: Tracks experiments, models, and metrics
  - Logs model parameters, metrics, and artifacts
  - Stores model versions in model registry
  - Retrieves latest/best model versions for deployment

**Version Control & CI/CD**:
- **GitHub**: Source code repository
  - Developers push code/commits to trigger workflows
- **GitHub Actions**: Automated CI/CD pipeline
  - Builds Docker images on code push
  - Runs tests and validates builds
  - Pushes images to Amazon ECR

**AWS Cloud Infrastructure**:
- **Amazon ECR (Elastic Container Registry)**: 
  - Stores Docker container images
  - Provides secure, scalable image storage
- **Amazon S3**: 
  - Stores MLflow artifacts (models, plots, datasets)
  - Logs experiments and model metadata
- **Amazon RDS (PostgreSQL)**:
  - Application backend store for users and predictions
  - MLflow backend store for experiment tracking
  - Stores run metadata, parameters, and metrics
- **EC2 Instance (MLflow Server)**:
  - Hosts MLflow tracking server
  - Provides UI for experiment visualization
  - Manages model registry
- **EC2 Instance (App Deployment)**:
  - Runs FastAPI application containers
  - Pulls latest images from ECR
  - Executes SQL queries and writes data to RDS
  - Returns prediction results to clients

**Deployment Flow**:
1. Developer commits code to GitHub
2. GitHub Actions triggers automated build
3. Docker image is built and pushed to ECR
4. EC2 instance pulls new image from ECR
5. Container is deployed with updated models
6. Application connects to MLflow for model loading
7. Predictions are served to end users

### Technology Stack
- **Orchestration**: Apache Airflow
- **API Framework**: FastAPI, Uvicorn
- **ML/DL**: scikit-learn, PyTorch, TensorFlow/Keras, transformers (RoBERTa)
- **Experiment Tracking**: MLflow
- **Database**: PostgreSQL (RDS for production, local for development)
- **Containerization**: Docker, Docker Compose
- **Cloud**: AWS (EC2, RDS, S3, ECR)
- **CI/CD**: GitHub Actions

## Project Structure

```
Mental-exploring/
├── airflow/                          # Airflow orchestration
│   ├── dags/                         # DAG definitions
│   │   ├── Depression_detection_dags/
│   │   ├── Emotion_detection_dags/
│   │   └── Sentiment_analysis_dags/
│   ├── airflow_logs/                 # Execution logs
│   ├── docker/                       # Airflow Dockerfile
│   └── postgres_airflow/             # Airflow metadata DB
│
├── api/                              # FastAPI application
│   ├── main.py                       # Application entry point
│   ├── routers/                      # API route handlers
│   ├── services/                     # ML service integrations
│   │   ├── depression_service/
│   │   ├── emotion_service/
│   │   └── sentiment_service/
│   ├── config/                       # Configuration management
│   ├── middleware/                   # Request/response middleware
│   ├── database/                     # Database connections
│   ├── utils/                        # Utility functions
│   ├── logs/                         # API logs
│   ├── results/                      # Prediction results
│   └── docker/                       # API Dockerfile
│   └──docker-compose.test.yml        # Testing environment
│   └──.env.example                   # Environment template
│
├── Depression_detection/             # Depression detection module
│   ├── src/                          # Source code
│   │   ├── models/                   # Model definitions
│   │   ├── text/                     # Text preprocessing
│   │   ├── config/                   # Module configuration
│   │   └── utils/                    # Helper functions
│   ├── notebooks/                    # Jupyter notebooks
│   ├── models/                       # Trained models
│   ├── plots/                        # Visualization outputs
│   └── logs/                         # Training logs
│
├── Emotion_detection/                # Emotion detection module
│   ├── src/                          # Source code
│   │   ├── models/                   # Model definitions
│   │   ├── images/                   # Image preprocessing
│   │   ├── config/                   # Module configuration
│   │   └── utils/                    # Helper functions
│   ├── notebooks/                    # Jupyter notebooks
│   ├── models/                       # Trained models
│   ├── plots/                        # Visualization outputs
│   └── assets/                       # Static assets
│
├── Sentiment_analysis/               # Sentiment analysis module
│   ├── src/                          # Source code
│   │   ├── models/                   # Model definitions
│   │   ├── tweets/                   # Tweet preprocessing
│   │   ├── config/                   # Module configuration
│   │   └── utils/                    # Helper functions
│   ├── notebooks/                    # Jupyter notebooks
│   ├── models/                       # Trained models
│   └── plots/                        # Visualization outputs
│
├── Data_assets/                      # Dataset references
├── docker-compose.yml                # Airflow services
├── .env.example                      # Environment template
└── README.md                         # This file
```

## Models & Pipelines

### 1. Depression Detection

https://github.com/Ziadashraf301/Mental-exploring/assets/111798631/7de96f15-a505-4841-878d-3214c45b5ed4

**Traditional ML Pipeline:**
- [Data collection](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Data_assets/Data%20References.md)
- [Data preprocessing](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Depression_detection/notebooks/Preprocessing.ipynb)
- [Feature engineering](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Depression_detection/notebooks/Models_Dev.ipynb)
- [Machine learning modeling](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Depression_detection/notebooks/Models_Dev.ipynb)
- [Model evaluation](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Depression_detection/notebooks/Models_Dev.ipynb)
- [Statistical testing](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Depression_detection/notebooks/Test_Models_Statistically.ipynb)

**Deep Learning Pipeline:**
- [Fine-tuned RoBERTa with LoRA](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Depression_detection/notebooks/Depression_Detection_With_Bert.ipynb)

### 2. Sentiment Analysis

https://github.com/Ziadashraf301/Mental-exploring/assets/111798631/95c7c048-5ea8-46b9-919b-2d4a3ad85037

**Pipeline:**
- [Data collection](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Data_assets/Data%20References.md)
- [Data preprocessing](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Sentiment_analysis/notebooks/Models_Dev.ipynb)
- [Feature engineering & Modeling](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Sentiment_analysis/notebooks/Models_Dev.ipynb)
- [Statistical testing](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Sentiment_analysis/notebooks/test_models_statistically.ipynb)

### 3. Emotion Detection

https://github.com/Ziadashraf301/Mental-exploring/assets/111798631/2ebc6909-3733-480a-8bac-d1d949d29366

**Pipeline:**
- [Data collection](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Data_assets/Data%20References.md)
- [Data preprocessing](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Emotion_detection/notebooks/Models_Dev.ipynb)
- [Machine learning modeling and evaluation](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Emotion_detection/notebooks/Models_Dev.ipynb)
- [Statistical testing](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Emotion_detection/notebooks/test_models_statistically.ipynb)
- [Model test](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Emotion_detection/notebooks/Model_Tests.ipynb)

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.12
- AWS Account (for deployment)
- Git

### Environment Configuration

Both the Airflow pipeline and FastAPI service require environment variables. Copy and configure the `.env.example` file:

```bash
# In the root directory (for Airflow)
cp .env.example .env

# In the api/ directory (for FastAPI)
cp api/.env.example api/.env
```

#### Required Environment Variables

```dotenv
# Postgres for Airflow
POSTGRES_USER=airflow
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=airflow
POSTGRES_PORT=5432

# AWS Credentials
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Airflow Configuration
AIRFLOW_EXECUTOR=LocalExecutor
AIRFLOW_SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:your_password@postgres_airflow:5432/airflow
AIRFLOW_FERNET_KEY=your_fernet_key  # Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
AIRFLOW_DEFAULT_TIMEZONE=UTC
AIRFLOW_LOAD_EXAMPLES=False
AIRFLOW_USER=admin
AIRFLOW_PASSWORD=admin

# SMTP Configuration (for Airflow alerts)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_STARTTLS=True
SMTP_SSL=False
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_MAIL_FROM=your_email@gmail.com
```

### Local Development Setup
First, create an MLFlow server locally or in AWS (Set S3 bucket), and update the URI in each train_config.yaml (tracking_uri) and .env files (MLFLOW_TRACKING_URI)
#### 1. Running Airflow (Model Training Pipeline)

From the **root directory**:

```bash
# Start all Airflow services
docker compose up -d

# View logs
docker compose logs -f

# Access Airflow UI
# Open browser: http://localhost:8080
# Login with credentials from .env (default: admin/admin)
```

**Airflow DAGs** will automatically:
- Load datasets from configured sources
- Preprocess data
- Extract features
- Train models
- Log experiments to MLflow
- Store artifacts in S3

#### 2. Running FastAPI (Prediction Service)

From the **api/** directory:

```bash
cd api/

# Start FastAPI service
docker compose -f docker-compose.test.yml up -d

# View logs
docker compose -f docker-compose.test.yml logs -f

# Access API documentation
# Open browser: http://localhost:8000/docs
```

#### 3. Running Tests

```bash
# Run API tests
python api/test/test_api.py
```

## AWS Infrastructure Requirements (Before Deployment)

To deploy the Mental Exploring system into **production on AWS**, the following infrastructure components **must already be created and configured** in your AWS environment:

### 1. **Amazon S3 (Artifact Storage)**

Used for:

* Storing MLflow model artifacts
* Storing experiment outputs and related files

### 2. **Amazon RDS – PostgreSQL (Backend Store)**

Used for:

* MLflow backend tracking (parameters, metrics, run metadata)
* Application database (user data, prediction logs)

### 3. **Amazon EC2 Instances**

Two EC2 instances are required:

* **MLflow Tracking Server Instance**
* **FastAPI Application Server Instance (Dockerized)**
  Pulls images from ECR and loads the latest registered models.

### 4. **Amazon ECR (Elastic Container Registry)**

Used for storing:

* Docker images for the FastAPI application
* Versioned API containers pushed via GitHub Actions

### 5. **IAM Roles & Security Policies**

Required for:

* MLflow server access to S3 and RDS
* API server access to ECR and model storage
* GitHub Actions to push images and trigger deployments

### 6. **Networking & Security Groups**

You must define:

* Security group for API server (HTTP/HTTPS inbound)
* Security group for MLflow server (port 5000 inbound, restricted)
* Security group for RDS (PostgreSQL, allowed only from EC2 instances)
* Subnets & VPC routing rules

### 7. **GitHub Actions CI/CD Configuration**

The GitHub repository must include:

* AWS credentials and .env files stored as GitHub Secrets
* ECR repository URLs
* SSH access to the EC2 instance for deployment

---

### API Endpoints

Once the FastAPI service is running, open the interactive documentation at:
**`http://localhost:8000/docs`**

Below is the **full list of all endpoints** included in FastAPI service.

---

## **Core API**

| Method  | Endpoint  | Description                 |
| ------- | --------- | --------------------------- |
| **GET** | `/`       | Root API metadata           |
| **GET** | `/health` | Global service health check |

---

## **User Management**

| Method     | Endpoint                       | Description                    |
| ---------- | ------------------------------ | ------------------------------ |
| **POST**   | `/users`                       | Create new user                |
| **GET**    | `/users/{user_id}`             | Fetch user details             |
| **PATCH**  | `/users/{user_id}`             | Update user info               |
| **DELETE** | `/users/{user_id}`             | Delete user                    |
| **GET**    | `/users/{user_id}/stats`       | User statistics summary        |
| **GET**    | `/users/{user_id}/predictions` | Get all predictions for a user |

---

## **Depression Detection Service**

| Method   | Endpoint                 | Description                               |
| -------- | ------------------------ | ----------------------------------------- |
| **GET**  | `/depression/health`     | Depression model health check             |
| **GET**  | `/depression/model/info` | Model details & configuration             |
| **POST** | `/depression/predict`    | Predict depression levels from text input |

---

## **Sentiment Analysis Service**

| Method   | Endpoint                | Description                       |
| -------- | ----------------------- | --------------------------------- |
| **GET**  | `/sentiment/health`     | Sentiment model health check      |
| **GET**  | `/sentiment/model/info` | Model details & metadata          |
| **POST** | `/sentiment/predict`    | Predict sentiment from text input |

---

## **Emotion Detection Service (Image)**

| Method   | Endpoint              | Description                         |
| -------- | --------------------- | ----------------------------------- |
| **GET**  | `/emotion/health`     | Emotion model health check          |
| **GET**  | `/emotion/model/info` | Model details & configuration       |
| **POST** | `/emotion/predict`    | Predict emotion from uploaded image |

---

## **Analytics Endpoints**

| Method  | Endpoint                              | Description                           |
| ------- | ------------------------------------- | ------------------------------------- |
| **GET** | `/analytics`                          | Main analytics overview               |
| **GET** | `/analytics/summary`                  | Summary of predictions                |
| **GET** | `/analytics/realtime`                 | Real-time analytics                   |
| **GET** | `/analytics/trends`                   | Trends (supports `?days=`)            |
| **GET** | `/analytics/service/emotion`          | Emotion service analytics             |
| **GET** | `/analytics/performance`              | API performance metrics               |
| **GET** | `/analytics/predictions/distribution` | Distribution of predictions           |
| **GET** | `/analytics/export`                   | Export analytics (`?format=json/csv`) |

---

## Development Workflow

1. **Model Development**: Use Jupyter notebooks in each module directory
2. **Pipeline Development**: Define DAGs in `airflow/dags/`
3. **API Development**: Add endpoints in `api/routers/`
4. **Testing**: Run tests with `api/test/test_api.py`
5. **Deployment**: Push to GitHub, CI/CD automatically deploys to AWS

## Monitoring and Logs

- **Airflow Logs**: `airflow/airflow_logs/`
- **API Logs**: `api/logs/`
- **MLflow UI**: Access experiment tracking at your MLflow server URL
- **Model Performance**: View plots in each module's `plots/` directory

## Troubleshooting

### Airflow Issues
```bash
# Reset Airflow database
docker compose down -v
docker compose up -d

# Check Airflow logs
docker compose logs airflow-webserver
docker compose logs airflow-scheduler
```

### API Issues
```bash
# Check API logs
docker compose -f api/docker-compose.test.yml logs -f

# Rebuild containers
docker compose -f api/docker-compose.test.yml up --build
```

### MLflow Connection Issues
- Verify MLflow server is running: `curl http://your-mlflow-url:5000/health`
- Check AWS credentials in `.env`
- Verify S3 bucket permissions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is part of a graduation project. Please contact for usage rights.

## Team

Developed as a graduation project focused on mental health awareness and accessibility.

## Acknowledgments

- Dataset providers (see `Data_assets/Data References.md`)
- Open-source ML/DL libraries