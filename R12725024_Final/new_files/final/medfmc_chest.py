import os
import pickle
import shutil
import pandas as pd

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD


@DATASET_REGISTRY.register()
class MedfmcChest(DatasetBase):

    dataset_dir = "medfmc/MedFMC"
    
    def __init__(self, cfg):

        TARGET = cfg.DATASET.TARGET
        SHOT = cfg.DATASET.NUM_SHOT
        
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "chest/images") # e.g. images
        self.split_path = os.path.join(self.dataset_dir, f"chest/split_tsai_medfmc_chest-{SHOT}shot-{TARGET}.json") # split_zhou_Food101.json
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "chest/split_fewshot") # split_fewshot
        mkdir_if_missing(self.split_fewshot_dir)

        # --------------------- By Arthur Tsai ---------------------
        # df_label = pd.read_csv(self.image_dir.replace("images", "chest_train.csv"))
        # df_label.drop(columns=['Unnamed: 0'], inplace=True)
        # df_label.set_index('img_id', inplace=True)
        # TARGET = df_label.columns[0]
        TARGET = cfg.DATASET.TARGET

        # self.neg_image_dir = os.path.join(self.image_dir, f"{TARGET} negative")
        # self.pos_image_dir = os.path.join(self.image_dir, f"{TARGET} positive")
        
        # os.makedirs(self.neg_image_dir, exist_ok=True)
        # os.makedirs(self.pos_image_dir, exist_ok=True)

        # for file in df_label[df_label[TARGET] == 0].index:
        #     shutil.move(os.path.join(self.image_dir, file), os.path.join(self.neg_image_dir, file))
        #     # shutil.move(f"images/{file}", f"images/{target} negative/{file}")
        # for file in df_label[df_label[TARGET] == 1].index:
        #     shutil.move(os.path.join(self.image_dir, file), os.path.join(self.pos_image_dir, file))
        #     # shutil.move(f"images/{file}", f"images/{target} positive/{file}")
        # ---------------------  ---------------------
        
        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)
