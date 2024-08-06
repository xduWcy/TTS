import os
from torch.utils.data import Dataset

class UtterDs(Dataset):
    def __init__(self, base_path):
        self.base_path = os.path.join(base_path, "APY210531004\data")
        self.available_ids = os.listdir(self.base_path)
    def __len__(self):
        return len(self.available_ids) * 3

    def __getitem__(self, index):
        file_id, utter_id = index // 3, index % 3
        file_code = self.available_ids[file_id]
        with open(os.path.join(self.base_path, file_code, "ProsodyLabeling", f"{file_code}.txt"), mode="r", encoding="utf-8") as f:
            meta = f.read().split("\n")[:-1]
        sound_id, text = meta[utter_id * 2].split("\t")
        words = text.split("#")
        words[1:] = [word[1:] for word in words[1:]]
        text = "".join(words)

        sound_path = os.path.join(self.base_path, file_code, "Wave", f"{sound_id}.wav")
        return sound_path, text