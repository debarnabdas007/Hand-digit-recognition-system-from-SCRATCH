edge-vision-project/
├── backend/                  # The FastAPI Microservice
│   ├── app/
│   │   ├── api/              # routers.py (The "Receptionist" - URL endpoints)
│   │   ├── core/             # config.py, logger.py, exceptions.py (The rulebook)
│   │   ├── schemas/          # pydantic_models.py (The "Security Guard" validating JSON)
│   │   ├── services/         # ml_engine.py (The "Engine Room" - PyTorch inference)
│   │   └── main.py           # Ignites the FastAPI server
│   ├── models/               # edge_digit_vision_final.pth (The Model Vault)
│   ├── Dockerfile            # Container instructions for the backend
│   └── requirements.txt      # Backend dependencies
│
└── frontend/                 # The Streamlit Microservice
    ├── ui.py                 # The interactive canvas
    ├── Dockerfile            # Container instructions for the frontend
    └── requirements.txt      # Frontend dependencies