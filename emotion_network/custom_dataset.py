from torch.utils.data import Dataset


class EmotionSpeechDataset(Dataset):

    def __init__(self, annotations_file, audio_dir) -> None:
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)