LABEL_COLUMNS = ["location_traffic_convenience",
                 "location_distance_from_business_district",
                 "location_easy_to_find",
                 "service_wait_time",
                 "service_waiters_attitude",
                 "service_parking_convenience",
                 "service_serving_speed",
                 "price_level",
                 "price_cost_effective",
                 "price_discount",
                 "environment_decoration",
                 "environment_noise",
                 "environment_space",
                 "environment_cleaness",
                 "dish_portion",
                 "dish_taste",
                 "dish_look",
                 "dish_recommendation",
                 "others_overall_experience",
                 "others_willing_to_consume_again"]

LABEL_COLUMNS_ALL = []
for label in LABEL_COLUMNS:
    LABEL_COLUMNS_ALL += [label+"_-2", label+"_-1", label+"_0", label+"_1"]

BERT_MODEL_NAME = 'bert-base-chinese'
MAX_TOKEN_COUNT = 512
N_EPOCHS = 1
BATCH_SIZE = 16


RANDOM_SEED = 42
ROOT = "."
DATASET = "/dataset"
DATA_PATH = ROOT + DATASET

THRESHOLD = 0.5



def func(x):
    if x == -2:
        return [1, 0, 0, 0]
    if x == -1:
        return [0, 1, 0, 0]
    if x == 0:
        return [0, 0, 1, 0]
    if x == 1:
        return [0, 0, 0, 1]


