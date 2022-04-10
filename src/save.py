from main import BikeRentalsPredictionService
import pickle
from pathlib import Path

source = Path(__file__).parent

brp = BikeRentalsPredictionService()

for i in ["casual", "registered"]:
    with open(f'{source}/model_{i}.pkl', 'rb') as f:
        model = pickle.load(f)

    brp.pack(f"model_{i}",  model)


saved_path = brp.save()
