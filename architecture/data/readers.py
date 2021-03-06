import copy
import json
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Set, Union

import h5py

from nltk.tokenize import word_tokenize
from tqdm import tqdm


class DialogsReader(object):

    def __init__(
        self,
        dialogs_jsonpath: str,
        num_examples: Optional[int] = None,
        num_workers: int = 1,
    ):
        with open(dialogs_jsonpath, "r") as visdial_file:
            visdial_data = json.load(visdial_file)
            self._split = visdial_data["split"]

            # Maintain questions and answers as a dict instead of list because
            # they are referenced by index in dialogs. We drop elements from
            # these in "overfit" mode to save time (tokenization is slow).
            self.questions = {
                i: question
                for i, question in enumerate(visdial_data["data"]["questions"])
            }
            self.answers = {
                i: answer for i, answer in enumerate(visdial_data["data"]["answers"])
            }

            # Add empty question, answer - useful for padding dialog rounds
            # for test split.
            self.questions[-1] = ""
            self.answers[-1] = ""

            # ``image_id``` serves as key for all three dicts here.
            self.captions: Dict[int, Any] = {}
            self.dialogs: Dict[int, Any] = {}
            self.num_rounds: Dict[int, Any] = {}

            all_dialogs = visdial_data["data"]["dialogs"]

            # Retain only first ``num_examples`` dialogs if specified.
            if num_examples is not None:
                all_dialogs = all_dialogs[:num_examples]

            for _dialog in all_dialogs:

                self.captions[_dialog["image_id"]] = _dialog["caption"]

                # Record original length of dialog, before padding.
                # 10 for train and val splits, 10 or less for test split.
                self.num_rounds[_dialog["image_id"]] = len(_dialog["dialog"])

                # Pad dialog at the end with empty question and answer pairs
                # (for test split).
                while len(_dialog["dialog"]) < 10:
                    _dialog["dialog"].append({"question": -1, "answer": -1})

                # Add empty answer (and answer options) if not provided
                # (for test split). We use "-1" as a key for empty questions
                # and answers.
                for i in range(len(_dialog["dialog"])):
                    if "answer" not in _dialog["dialog"][i]:
                        _dialog["dialog"][i]["answer"] = -1
                    if "answer_options" not in _dialog["dialog"][i]:
                        _dialog["dialog"][i]["answer_options"] = [-1] * 100

                self.dialogs[_dialog["image_id"]] = _dialog["dialog"]

            # If ``num_examples`` is specified, collect questions and answers
            # included in those examples, and drop the rest to save time while
            # tokenizing. Collecting these should be fast because num_examples
            # during debugging are generally small.
            if num_examples is not None:
                questions_included: Set[int] = set()
                answers_included: Set[int] = set()

                for _dialog in self.dialogs.values():
                    for _dialog_round in _dialog:
                        questions_included.add(_dialog_round["question"])
                        answers_included.add(_dialog_round["answer"])
                        for _answer_option in _dialog_round["answer_options"]:
                            answers_included.add(_answer_option)

                self.questions = {i: self.questions[i] for i in questions_included}
                self.answers = {i: self.answers[i] for i in answers_included}

            self._multiprocess_tokenize(num_workers)

    def _multiprocess_tokenize(self, num_workers: int):
        """
        Tokenize captions, questions and answers in parallel processes. This
        method uses multiprocessing module internally.
        """

        # While displaying progress bar through tqdm, specify total number of
        # sequences to tokenize, because tqdm won't know in case of pool.imap
        with mp.Pool(num_workers) as pool:
            print(f"[{self._split}] Tokenizing questions...")
            _question_tuples = self.questions.items()
            _question_indices = [t[0] for t in _question_tuples]
            _questions = list(
                tqdm(
                    pool.imap(word_tokenize, [t[1] for t in _question_tuples]),
                    total=len(self.questions),
                )
            )
            self.questions = {
                i: question + ["?"]
                for i, question in zip(_question_indices, _questions)
            }
            # Delete variables to free memory.
            del _question_tuples, _question_indices, _questions

            print(f"[{self._split}] Tokenizing answers...")
            _answer_tuples = self.answers.items()
            _answer_indices = [t[0] for t in _answer_tuples]
            _answers = list(
                tqdm(
                    pool.imap(word_tokenize, [t[1] for t in _answer_tuples]),
                    total=len(self.answers),
                )
            )
            self.answers = {
                i: answer + ["?"] for i, answer in zip(_answer_indices, _answers)
            }
            # Delete variables to free memory.
            del _answer_tuples, _answer_indices, _answers

            print(f"[{self._split}] Tokenizing captions...")
            # Convert dict to separate lists of image_ids and captions.
            _caption_tuples = self.captions.items()
            _image_ids = [t[0] for t in _caption_tuples]
            _captions = list(
                tqdm(
                    pool.imap(word_tokenize, [t[1] for t in _caption_tuples]),
                    total=(len(_caption_tuples)),
                )
            )
            # Convert tokenized captions back to a dict.
            self.captions = {i: c for i, c in zip(_image_ids, _captions)}

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, image_id: int) -> Dict[str, Union[int, str, List]]:
        caption_for_image = self.captions[image_id]
        dialog = copy.copy(self.dialogs[image_id])
        num_rounds = self.num_rounds[image_id]

        # Replace question and answer indices with actual word tokens.
        for i in range(len(dialog)):
            dialog[i]["question"] = self.questions[dialog[i]["question"]]
            dialog[i]["answer"] = self.answers[dialog[i]["answer"]]
            for j, answer_option in enumerate(dialog[i]["answer_options"]):
                dialog[i]["answer_options"][j] = self.answers[answer_option]

        return {
            "image_id": image_id,
            "caption": caption_for_image,
            "dialog": dialog,
            "num_rounds": num_rounds,
        }

    def keys(self) -> List[int]:
        return list(self.dialogs.keys())

    @property
    def split(self):
        return self._split


class DenseAnnotationsReader(object):

    def __init__(self, dense_annotations_jsonpath: str):
        with open(dense_annotations_jsonpath, "r") as visdial_file:
            self._visdial_data = json.load(visdial_file)
            self._image_ids = [entry["image_id"] for entry in self._visdial_data]

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, image_id: int) -> Dict[str, Union[int, List]]:
        index = self._image_ids.index(image_id)
        # keys: {"image_id", "round_id", "gt_relevance"}
        return self._visdial_data[index]

    @property
    def split(self):
        # always
        return "val"


class ImageFeaturesHdfReader(object):
    """
    A reader for HDF files containing pre-extracted image features. A typical
    HDF file is expected to have a column named "image_id", and another column
    named "features".
    """

    def __init__(self, features_hdfpath: str, in_memory: bool = False):
        self.features_hdfpath = features_hdfpath
        self._in_memory = in_memory

        with h5py.File(self.features_hdfpath, "r") as features_hdf:
            self._split = features_hdf.attrs["split"]
            self._image_id_list = list(features_hdf["image_id"])
            # "features" is List[np.ndarray] if the dataset is loaded in-memory
            # If not loaded in memory, then list of None.
            self.features = [None] * len(self._image_id_list)

    def __len__(self):
        return len(self._image_id_list)

    def __getitem__(self, image_id: int):
        index = self._image_id_list.index(image_id)
        if self._in_memory:
            # Load features during first epoch, all not loaded together as it
            # has a slow start.
            if self.features[index] is not None:
                image_id_features = self.features[index]
            else:
                with h5py.File(self.features_hdfpath, "r") as features_hdf:
                    image_id_features = features_hdf["features"][index]
                    self.features[index] = image_id_features
        else:
            # Read chunk from file everytime if not loaded in memory.
            with h5py.File(self.features_hdfpath, "r") as features_hdf:
                image_id_features = features_hdf["features"][index]

        return image_id_features

    def keys(self) -> List[int]:
        return self._image_id_list

    @property
    def split(self):
        return self._split


class CaptionReader(object):
    def __init__(self, caption_jsonpath: str):
        with open(caption_jsonpath, "r") as caption_file:
            caption_data = json.load(caption_file)
            self._split = caption_data["split"]
            self.dataall = caption_data["data"]
            self.caption = {}

            for item in self.dataall:
                self.caption[item["image_id"]] = item["captions"]

            print(f"[{self._split}] Tokenizing captions mul ...")
            for image_id in tqdm(self.caption.keys()):

                caption_token = []
                for each_caption in self.caption[image_id]:

                    caption_each = word_tokenize(each_caption)
                    caption_token.append(caption_each)
                self.caption[image_id] = caption_token

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, image_id: int) -> Dict[str, Union[int, str, List]]:

        caption_for_image = self.caption[image_id]

        caption_rounds = len(self.caption[image_id])

        return {
            "image_id": image_id,
            "captions": caption_for_image,
            "caption_rounds": caption_rounds,
        }

    def keys(self) -> List[int]:
        return list(self.caption.keys())

    @property
    def split(self):
        return self._split
