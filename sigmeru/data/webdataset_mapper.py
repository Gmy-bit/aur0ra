from __future__ import annotations

import copy
import glob
import random
from typing import Callable

import webdataset as wds
import wordsegment as ws
from loguru import logger
from torch.utils.data import IterableDataset
from torchvision import transforms as T

import sigmeru.utils.distributed as dist

ws.load()


def _sample_has_parent_field(key: str) -> bool:
    """Return True for Grounded/GRIT parent-box fields."""
    return key.startswith("parent") or key == "numparents.txt"


def _drop_parent_fields(sample: dict) -> dict:
    """
    Drop parent box images/text before PIL decoding.

    This is intentionally placed before ``wds.decode('pil')`` when boxes are not
    needed. Dropping after decode would still pay the JPEG/PIL decode cost.
    """
    return {key: value for key, value in sample.items() if not _sample_has_parent_field(key)}


class ImageTextWebDataset(IterableDataset):
    """
    Iterable dataset that serves instances from TAR file shards.

    When the mapper declares ``use_boxes=False``, parent box fields can be
    dropped before image decoding. This keeps SigMERU/MERU/CLIP no-box training
    mathematically identical while avoiding unused parent-image decode and
    transform work.
    """

    def __init__(
        self,
        tarfiles: str | list[str],
        mapper: Callable,
        buffer_size: int = 5000,
        infinite_stream: bool = True,
        seed: int = 0,
        drop_unused_box_fields: bool = True,
    ):
        """
        Args:
            tarfiles: Path(s) or glob-patterns for TAR files in WebDataset format.
            mapper: Callable to transform a single dataset dict.
            buffer_size: Internal shuffle buffer size. Set <= 1 to disable.
            infinite_stream: Yield indefinitely when True.
            seed: Random seed for deterministic shuffling.
            drop_unused_box_fields: If True and mapper.use_boxes is False, drop
                parent box fields before decode.
        """
        super().__init__()
        self.mapper = mapper
        self.buffer_size = buffer_size
        self.infinite_stream = infinite_stream
        self.seed = seed
        self.drop_unused_box_fields = drop_unused_box_fields

        if isinstance(tarfiles, str):
            tarfiles = [tarfiles]

        self.tarfiles = []
        for _path in tarfiles:
            for _single_glob in _path.split():
                self.tarfiles.extend(glob.glob(_single_glob))

        self.tarfiles = sorted(self.tarfiles)
        logger.info(f"{self.__class__.__name__} found {len(self.tarfiles)} TARs.")

        _rank, _world_size = dist.get_rank(), dist.get_world_size()
        self.tarfiles = self.tarfiles[_rank::_world_size]
        logger.info(f"RANK {_rank} will load {len(self.tarfiles)} TARs.")

    @property
    def _mapper_uses_boxes(self) -> bool:
        return bool(getattr(self.mapper, "use_boxes", True))

    def __iter__(self):
        rng = random.Random(self.seed)
        pipeline = wds.DataPipeline(
            wds.SimpleShardList(self.tarfiles, seed=self.seed),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
        )

        if self.buffer_size > 1:
            pipeline.append(
                wds.shuffle(self.buffer_size, initial=self.buffer_size, rng=rng),
            )

        if self.drop_unused_box_fields and not self._mapper_uses_boxes:
            pipeline.append(wds.map(_drop_parent_fields))

        pipeline.append(wds.decode("pil", handler=wds.warn_and_continue))
        pipeline.append(wds.map(self.mapper))

        if self.infinite_stream:
            while True:
                pipeline_copy = copy.deepcopy(pipeline)
                yield from pipeline_copy
        else:
            yield from pipeline


class GroundedDatasetTarMapper:
    """
    Mapper to pre-process image-text instances from Grounded/GRIT TAR files.
    """

    def __init__(
        self,
        image_transform: list[Callable] = [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
        ],
        use_boxes: bool = True,
    ):
        """
        Args:
            image_transform: List of image transformations from torchvision.
            use_boxes: Return parent box image/text fields when True. Set False
                for SigMERU/MERU/CLIP no-box training.
        """
        self.image_transform = T.Compose(image_transform)
        self.use_boxes = use_boxes

    def __call__(self, dataset_dict: dict):
        output = {
            "__key__": dataset_dict["__key__"],
            "image": self.image_transform(dataset_dict["child.jpg"]),
            "text": dataset_dict["child.txt"],
        }

        if self.use_boxes:
            num_boxes = int(dataset_dict["numparents.txt"])
            random_box = random.randrange(num_boxes)
            output.update(
                {
                    "box_image": self.image_transform(
                        dataset_dict[f"parent{random_box:03d}.jpg"]
                    ),
                    "box_text": dataset_dict[f"parent{random_box:03d}.txt"],
                }
            )

        return output
