from typing import Union

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pkg import data_preprocessing

import joblib
import uvicorn
from pipeline.DisasterPredictionPipeline import DisasterPredictionPipeline
import __main__

setattr(__main__, 'DisasterPredictionPipeline', DisasterPredictionPipeline)
state_emission_sums = {
    'Alabama': 32203132.0,
    'Arizona': 21286418.0,
    'Arkansas': 16515760.0,
    'California': 73194880.0,
    'Colorado': 21831952.0,
    'Connecticut': 8629382.0,
    'Delaware': 2781860.0,
    'District of Columbia': 1200219.375,
    'Florida': 50213768.0,
    'Georgia': 30042910.0,
    'Idaho': 5926557.0,
    'Illinois': 49257192.0,
    'Indiana': 40151328.0,
    'Iowa': 21242270.0,
    'Kansas': 16504667.0,
    'Kentucky': 25980814.0,
    'Louisiana': 31624816.0,
    'Maine': 4159026.0,
    'Maryland': 15023604.0,
    'Massachusetts': 14793342.0,
    'Michigan': 39783608.0,
    'Minnesota': 23890488.0,
    'Mississippi': 13527731.0,
    'Missouri': 27441352.0,
    'Montana': 5195487.0,
    'Nebraska': 12358290.0,
    'Nevada': 9590186.0,
    'New Hampshire': 3720648.5,
    'New Jersey': 22195524.0,
    'New Mexico': 10083089.0,
    'New York': 35945840.0,
    'North Carolina': 29293660.0,
    'North Dakota': 17414154.0,
    'Ohio': 46500944.0,
    'Oklahoma': 19011936.0,
    'Oregon': 11238526.0,
    'Pennsylvania': 50168752.0,
    'Rhode Island': 2615203.5,
    'South Carolina': 16354849.0,
    'South Dakota': 5452694.5,
    'Tennessee': 25447202.0,
    'Texas': 120057864.0,
    'Utah': 14572920.0,
    'Vermont': 1703126.625,
    'Virginia': 21978504.0,
    'Washington': 16859102.0,
    'West Virginia': 14949392.0,
    'Wisconsin': 25707530.0,
    'Wyoming': 17854828.0
}

model = joblib.load("disaster_prediction_pipeline.pkl")


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, change to specific origins for better security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{state_name}/{year}/{reduction_rate}")
async def read_item(state_name: str, year: int, reduction_rate: float):
    try:
        if state_name not in state_emission_sums:
            return {"error": "State not found"}
        # Load the model inside the function
        co2_emission = state_emission_sums[state_name]
        co2_emission = co2_emission * (1 - reduction_rate)
        predictions = model.predict([[co2_emission, data_preprocessing.year_encoding(year), data_preprocessing.map_state_to_encoding(state_name)]])
        print(predictions)
        return JSONResponse(content={"predictions": predictions.tolist()}, headers={"Access-Control-Allow-Origin": "*", "Content-Type": "application/json"})
    except KeyError as e:
        return {"error": f'Invalid state: {state_name}'}
    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
