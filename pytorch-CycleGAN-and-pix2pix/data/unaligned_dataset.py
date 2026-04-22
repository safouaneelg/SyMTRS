import os
import copy
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  # create a path '/path/to/data/trainB'

        if os.path.isdir(self.dir_A) and os.path.isdir(self.dir_B):
            self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
            self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        else:
            raw_a_dir, raw_b_dir = self._resolve_raw_dirs(opt)
            raw_a_paths = sorted(make_dataset(raw_a_dir, float("inf")))
            raw_b_paths = sorted(make_dataset(raw_b_dir, float("inf")))
            self.A_paths = self._split_paths(raw_a_paths, opt.phase, opt.train_ratio, opt.val_ratio, opt.split_seed)
            self.B_paths = self._split_paths(raw_b_paths, opt.phase, opt.train_ratio, opt.val_ratio, opt.split_seed + 1)

            if opt.max_dataset_size != float("inf"):
                self.A_paths = self.A_paths[: min(opt.max_dataset_size, len(self.A_paths))]
                self.B_paths = self.B_paths[: min(opt.max_dataset_size, len(self.B_paths))]

            if len(self.A_paths) == 0 or len(self.B_paths) == 0:
                raise RuntimeError(
                    "Auto-split produced an empty dataset. "
                    f"Check train_ratio/val_ratio and dataset sizes. "
                    f"A={len(self.A_paths)} B={len(self.B_paths)} phase={opt.phase}"
                )
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == "BtoA"
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image

        if self.opt.tile_size > 0:
            tile_opt = copy.deepcopy(self.opt)
            tile_opt.preprocess = "none"
            tile_opt.load_size = self.opt.tile_size
            tile_opt.crop_size = self.opt.tile_size
            self.transform_A = get_transform(tile_opt, grayscale=(input_nc == 1))
            self.transform_B = get_transform(tile_opt, grayscale=(output_nc == 1))
        else:
            self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
            self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--raw_A_dir", type=str, default="", help="optional absolute path to raw domain A images (no train/test split)")
        parser.add_argument("--raw_B_dir", type=str, default="", help="optional absolute path to raw domain B images (no train/test split)")
        parser.add_argument("--raw_A_subdir", type=str, default="hr", help="subdir under dataroot for raw domain A images when trainA/trainB are missing")
        parser.add_argument("--raw_B_subdir", type=str, default="night", help="subdir under dataroot for raw domain B images when trainA/trainB are missing")
        parser.add_argument("--train_ratio", type=float, default=0.9, help="fraction of images used for training when auto-splitting")
        parser.add_argument("--val_ratio", type=float, default=0.0, help="fraction of images used for validation when auto-splitting")
        parser.add_argument("--split_seed", type=int, default=0, help="random seed for deterministic auto-splitting")
        parser.add_argument("--tile_size", type=int, default=0, help="if > 0, randomly tile images to this size instead of resizing")
        parser.add_argument("--tile_mode", type=str, default="random", help="tile selection mode [random | center]")
        return parser

    @staticmethod
    def _split_paths(paths, phase, train_ratio, val_ratio, seed):
        if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio >= 1.0:
            raise ValueError("train_ratio and val_ratio must be >= 0 and sum to less than 1.0")

        if len(paths) == 0:
            return []

        rng = random.Random(seed)
        indices = list(range(len(paths)))
        rng.shuffle(indices)

        train_end = int(len(paths) * train_ratio)
        val_end = train_end + int(len(paths) * val_ratio)

        if phase == "train":
            selected = indices[:train_end]
        elif phase == "val":
            selected = indices[train_end:val_end]
        else:
            selected = indices[val_end:]

        return [paths[i] for i in selected]

    @staticmethod
    def _resolve_raw_dirs(opt):
        raw_a_dir = opt.raw_A_dir.strip()
        raw_b_dir = opt.raw_B_dir.strip()
        if raw_a_dir and raw_b_dir:
            if not os.path.isdir(raw_a_dir):
                raise FileNotFoundError(f"raw_A_dir not found: {raw_a_dir}")
            if not os.path.isdir(raw_b_dir):
                raise FileNotFoundError(f"raw_B_dir not found: {raw_b_dir}")
            return raw_a_dir, raw_b_dir

        raw_a_dir = os.path.join(opt.dataroot, opt.raw_A_subdir)
        raw_b_dir = os.path.join(opt.dataroot, opt.raw_B_subdir)
        if not os.path.isdir(raw_a_dir) or not os.path.isdir(raw_b_dir):
            raise FileNotFoundError(
                "Could not find trainA/trainB or raw directories for auto-splitting. "
                f"Checked: {raw_a_dir} and {raw_b_dir}. "
                "Provide --raw_A_dir and --raw_B_dir, or set --raw_A_subdir/--raw_B_subdir."
            )
        return raw_a_dir, raw_b_dir

    @staticmethod
    def _tile_image(img, tile_size, tile_mode):
        w, h = img.size
        if w < tile_size or h < tile_size:
            return img

        if tile_mode == "center":
            left = (w - tile_size) // 2
            top = (h - tile_size) // 2
        else:
            left = random.randint(0, w - tile_size)
            top = random.randint(0, h - tile_size)
        return img.crop((left, top, left + tile_size, top + tile_size))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")
        if self.opt.tile_size > 0:
            A_img = self._tile_image(A_img, self.opt.tile_size, self.opt.tile_mode)
            B_img = self._tile_image(B_img, self.opt.tile_size, self.opt.tile_mode)
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
