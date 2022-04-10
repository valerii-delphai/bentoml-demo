from main import BikeRentalsPredictionService
import pickle
from pathlib import Path

source = Path(__file__).parent

# call predefined earlier bentoml object
brp = BikeRentalsPredictionService()

# for two models...
for i in ["casual", "registered"]:
    # unpickle
    with open(f'{source}/model_{i}.pkl', 'rb') as f:
        model = pickle.load(f)
    # pack
    brp.pack(f"model_{i}",  model)

# convert to service
saved_path = brp.save()
