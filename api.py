from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging
import time

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar el modelo entrenado (modelo previamente guardado como model.pk)
model = joblib.load("modelo.pkl")

# Inicializar FastAPI
app = FastAPI()

# Middleware para CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las fuentes (puedes restringir esto en producción)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware para logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    idem = request.headers.get("X-Request-ID", "N/A")
    logger.info(f"Request {idem} - {request.method} {request.url}")
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Request {idem} completed in {process_time:.4f} seconds")
    
    return response

# Definir el esquema de entrada para las especificaciones del auto
class CarSpecifications(BaseModel):
    levy: float
    manufacturer: str
    model: str
    prod_year: int
    category: str
    leather_interior: bool
    fuel_type: str
    engine_volume: float
    mileage: float
    cylinders: int
    gear_box_type: str
    drive_wheels: str
    doors: int
    wheel: str
    color: str
    airbags: int

# Definir el esquema de entrada para la evaluación del precio
class PriceEvaluation(BaseModel):
    specifications: CarSpecifications
    proposed_price: float

# Definir el esquema de entrada para el presupuesto y las preferencias
class BudgetAndPreferences(BaseModel):
    budget: float
    manufacturer: str = None
    category: str = None
    fuel_type: str = None
    min_prod_year: int = None

# Lista de columnas categóricas
categorical = ['Manufacturer', 'Model', 'Category', 'Leather interior', 'Fuel type',
               'Gear box type', 'Drive wheels', 'Doors', 'Wheel', 'Color']

# Servicio 1: Predecir el precio óptimo basado en las especificaciones
@app.post("/predict_price/")
def predict_price(specs: CarSpecifications):
    # Crear un DataFrame con las características del auto
    car_features = pd.DataFrame([{
        'Levy': specs.levy,
        'Manufacturer': specs.manufacturer,
        'Model': specs.model,
        'Prod. year': specs.prod_year,
        'Category': specs.category,
        'Leather interior': specs.leather_interior,
        'Fuel type': specs.fuel_type,
        'Engine volume': specs.engine_volume,
        'Mileage': specs.mileage,
        'Cylinders': specs.cylinders,
        'Gear box type': specs.gear_box_type,
        'Drive wheels': specs.drive_wheels,
        'Doors': specs.doors,
        'Wheel': specs.wheel,
        'Color': specs.color,
        'Airbags': specs.airbags
    }])

    # Convertir las columnas categóricas en variables numéricas
    label_encoders = {}
    for col in categorical:
        le = LabelEncoder()
        car_features[col] = le.fit_transform(car_features[col])
        label_encoders[col] = le

    # Realizar la predicción con el modelo
    try:
        predicted_price = model.predict(car_features)[0]
        return {"optimal_price": predicted_price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Servicio 2: Evaluar si el precio propuesto por el vendedor es razonable
@app.post("/evaluate_price/")
def evaluate_price(evaluation: PriceEvaluation):
    car_features = pd.DataFrame([{
        'Levy': evaluation.specifications.levy,
        'Manufacturer': evaluation.specifications.manufacturer,
        'Model': evaluation.specifications.model,
        'Prod. year': evaluation.specifications.prod_year,
        'Category': evaluation.specifications.category,
        'Leather interior': evaluation.specifications.leather_interior,
        'Fuel type': evaluation.specifications.fuel_type,
        'Engine volume': evaluation.specifications.engine_volume,
        'Mileage': evaluation.specifications.mileage,
        'Cylinders': evaluation.specifications.cylinders,
        'Gear box type': evaluation.specifications.gear_box_type,
        'Drive wheels': evaluation.specifications.drive_wheels,
        'Doors': evaluation.specifications.doors,
        'Wheel': evaluation.specifications.wheel,
        'Color': evaluation.specifications.color,
        'Airbags': evaluation.specifications.airbags
    }])

    # Convertir las columnas categóricas en variables numéricas
    label_encoders = {}
    for col in categorical:
        le = LabelEncoder()
        car_features[col] = le.fit_transform(car_features[col])
        label_encoders[col] = le

    # Predecir el precio óptimo
    try:
        predicted_price = model.predict(car_features)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Evaluar si el precio propuesto es razonable
    price_diff = abs(predicted_price - evaluation.proposed_price)

    if price_diff / predicted_price < 0.1:  # Si la diferencia es menor al 10%
        return {"evaluation": "The proposed price is reasonable."}
    else:
        return {"evaluation": "The proposed price is not reasonable."}

# Servicio 3: Sugerir el mejor auto basado en presupuesto y preferencias
@app.post("/recommend_car/")
def recommend_car(preferences: BudgetAndPreferences):
    # Cargar el dataset original
    try:
        car_data = pd.read_csv("C:/Users/sapin/OneDrive/Documentos/Flare/prueba-tecnica/car_price_prediction.csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {e}")

    # Filtrar los autos según las preferencias del usuario
    filtered_cars = car_data[
        (car_data["Price"] <= preferences.budget) &
        (preferences.manufacturer is None or car_data["Manufacturer"] == preferences.manufacturer) &
        (preferences.category is None or car_data["Category"] == preferences.category) &
        (preferences.fuel_type is None or car_data["Fuel type"] == preferences.fuel_type) &
        (preferences.min_prod_year is None or car_data["Prod. year"] >= preferences.min_prod_year)
    ]

    if filtered_cars.empty:
        return {"recommendation": "No cars found matching the criteria."}

    # Sugerir el auto con las mejores características (se puede mejorar con modelos más complejos)
    recommended_car = filtered_cars.iloc[0].to_dict()  # Sugerimos el primer auto filtrado
    return {"recommended_car": recommended_car}

# Ejecutar la aplicación en localhost
# Inicia el servidor con `uvicorn main:app --reload`