import os
import copy
import random
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = []
        if os.path.isdir(self.dir_AB):
            self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths

        if len(self.AB_paths) == 0:
            raw_a_dir, raw_b_dir = self._resolve_raw_dirs(opt)
            self.A_paths, self.B_paths = self._build_paired_paths(
                raw_a_dir, raw_b_dir, opt.phase, opt.train_ratio, opt.val_ratio, opt.split_seed, opt.max_dataset_size
            )
            if len(self.A_paths) == 0:
                raise RuntimeError("No paired images found for aligned dataset. Check filenames and split settings.")

        assert self.opt.load_size >= self.opt.crop_size  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == "BtoA" else self.opt.output_nc

        if self.opt.tile_size > 0:
            tile_opt = copy.deepcopy(self.opt)
            tile_opt.preprocess = "none"
            tile_opt.load_size = self.opt.tile_size
            tile_opt.crop_size = self.opt.tile_size
            self.A_transform = get_transform(tile_opt, grayscale=(self.input_nc == 1))
            self.B_transform = get_transform(tile_opt, grayscale=(self.output_nc == 1))
        else:
            self.A_transform = None
            self.B_transform = None

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--raw_A_dir", type=str, default="", help="optional absolute path to raw domain A images (paired)")
        parser.add_argument("--raw_B_dir", type=str, default="", help="optional absolute path to raw domain B images (paired)")
        parser.add_argument("--raw_A_subdir", type=str, default="hr", help="subdir under dataroot for raw domain A images when train/test folders are missing")
        parser.add_argument("--raw_B_subdir", type=str, default="night", help="subdir under dataroot for raw domain B images when train/test folders are missing")
        parser.add_argument("--train_ratio", type=float, default=0.9, help="fraction of images used for training when auto-splitting")
        parser.add_argument("--val_ratio", type=float, default=0.0, help="fraction of images used for validation when auto-splitting")
        parser.add_argument("--split_seed", type=int, default=0, help="random seed for deterministic auto-splitting")
        parser.add_argument("--tile_size", type=int, default=0, help="if > 0, tile paired images to this size instead of resizing")
        parser.add_argument("--tile_mode", type=str, default="random", help="tile selection mode [random | center]")
        parser.add_argument("--bad_image_max_retries", type=int, default=10, help="max retries to skip unreadable images")
        return parser

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
                "Could not find paired raw directories for aligned dataset. "
                f"Checked: {raw_a_dir} and {raw_b_dir}. "
                "Provide --raw_A_dir and --raw_B_dir, or set --raw_A_subdir/--raw_B_subdir."
            )
        return raw_a_dir, raw_b_dir

    @staticmethod
    def _build_paired_paths(raw_a_dir, raw_b_dir, phase, train_ratio, val_ratio, seed, max_dataset_size):
        if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio >= 1.0:
            raise ValueError("train_ratio and val_ratio must be >= 0 and sum to less than 1.0")

        a_paths = sorted(make_dataset(raw_a_dir, float("inf")))
        b_paths = sorted(make_dataset(raw_b_dir, float("inf")))
        a_map = {os.path.basename(p): p for p in a_paths}
        b_map = {os.path.basename(p): p for p in b_paths}

        common = sorted(set(a_map.keys()) & set(b_map.keys()))
        if len(common) == 0:
            return [], []

        rng = random.Random(seed)
        indices = list(range(len(common)))
        rng.shuffle(indices)

        train_end = int(len(common) * train_ratio)
        val_end = train_end + int(len(common) * val_ratio)

        if phase == "train":
            selected = indices[:train_end]
        elif phase == "val":
            selected = indices[train_end:val_end]
        else:
            selected = indices[val_end:]

        if max_dataset_size != float("inf"):
            selected = selected[: min(max_dataset_size, len(selected))]

        a_sel = [a_map[common[i]] for i in selected]
        b_sel = [b_map[common[i]] for i in selected]
        return a_sel, b_sel

    @staticmethod
    def _tile_pair(a_img, b_img, tile_size, tile_mode):
        w, h = a_img.size
        if w < tile_size or h < tile_size:
            return a_img, b_img

        if tile_mode == "center":
            left = (w - tile_size) // 2
            top = (h - tile_size) // 2
        else:
            left = random.randint(0, w - tile_size)
            top = random.randint(0, h - tile_size)
        box = (left, top, left + tile_size, top + tile_size)
        return a_img.crop(box), b_img.crop(box)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        attempts = 0
        while True:
            try:
                if len(self.AB_paths) > 0:
                    # read a image given a random integer index
                    AB_path = self.AB_paths[index]
                    AB = Image.open(AB_path).convert("RGB")
                    # split AB image into A and B
                    w, h = AB.size
                    w2 = int(w / 2)
                    A = AB.crop((0, 0, w2, h))
                    B = AB.crop((w2, 0, w, h))
                    A_path = AB_path
                    B_path = AB_path
                else:
                    A_path = self.A_paths[index]
                    B_path = self.B_paths[index]
                    A = Image.open(A_path).convert("RGB")
                    B = Image.open(B_path).convert("RGB")
                break
            except OSError:
                attempts += 1
                if attempts >= self.opt.bad_image_max_retries:
                    raise RuntimeError("Exceeded bad_image_max_retries while reading images") from None
                index = random.randint(0, self.__len__() - 1)

        if self.opt.tile_size > 0:
            A, B = self._tile_pair(A, B, self.opt.tile_size, self.opt.tile_mode)
            A = self.A_transform(A)
            B = self.B_transform(B)
        else:
            # apply the same transform to both A and B
            transform_params = get_params(self.opt, A.size)
            A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
            B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
            A = A_transform(A)
            B = B_transform(B)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        if len(self.AB_paths) > 0:
            return len(self.AB_paths)
        return len(self.A_paths)
