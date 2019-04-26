import os
import cv2
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset

from transforms import GroupMultiScaleCrop


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VideoDataset(Dataset):

    def __init__(self, root_path, list_file, transform, mode='train', T=3, tau=16, dense_sample=True, On_Video=True,
                 image_tmpl='img_{:05d}.jpg'):
        self.mode = mode
        self.list_file = list_file
        self.root_path = root_path
        self.T = T
        self.tau = tau
        self.dense_sample = dense_sample
        self.On_video = On_Video
        self.transform = transform
        self.image_tmpl = image_tmpl

        self._parse_list()

    def _parse_list(self):
        # related path of the video  /  number of frames  /  label
        self.video_list = [VideoRecord(x.strip().split()) for x in open(self.list_file)]

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == 'train':
            segment_indices = self._sample_indices(record)
        else:
            segment_indices = self._get_val_indices(record)
        return self._get(record, segment_indices)

    def _sample_indices(self, record):
        stride = self.tau // 8
        raw_total_frames = self.T * self.tau
        total_frames = self.T * self.tau // stride

        # the stride of the frames when load videos .
        # becase the sample way in the paper has some frame
        # do not use. we will not load them to memory

        if self.dense_sample:  # use i3d's way to get dense frame
            sample_pos = max(1, 1 + record.num_frames - raw_total_frames)
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos)
            offsets = [
                (start_idx + stride * idx) if (start_idx + stride * idx) < record.num_frames else record.num_frames - 1
                for idx in range(total_frames)]
        else:  # use tsn's way to get sparse frame
            average_duration = record.num_frames // raw_total_frames
            if average_duration >= 1:
                offsets = np.multiply(list(range(raw_total_frames)), average_duration) + np.random.randint(
                    average_duration, size=raw_total_frames)
                offsets = offsets[::, stride]
            else:
                average_duration = record.num_frames // total_frames
                if average_duration >= 1:
                    offsets = np.multiply(list(range(total_frames)), average_duration) + np.random.randint(
                        average_duration, size=total_frames)
                else:
                    need = total_frames - record.num_frames
                    offsets = list(range(record.num_frames)) + [record.num_frames - 1] * need
        offsets = np.array(offsets)
        return offsets if self.On_video else offsets + 1

    def _get_val_indices(self, record):

        # to run fast. We do not use the multi clips to val.
        # only to ensure the uniqueness of the clip

        stride = self.tau // 8
        raw_total_frames = self.T * self.tau
        total_frames = self.T * self.tau // stride

        if self.dense_sample:  # use i3d's way to get dense frame
            sample_pos = max(1, 1 + record.num_frames - raw_total_frames)
            start_idx = 0 if sample_pos == 1 else sample_pos // 2
            offsets = [
                (start_idx + stride * idx) if (start_idx + stride * idx) < record.num_frames else record.num_frames - 1
                for idx in range(total_frames)]
        else:  # use tsn's way to get sparse frame
            average_duration = record.num_frames // raw_total_frames
            if average_duration >= 1:
                offsets = np.multiply(list(range(raw_total_frames)), average_duration) + average_duration // 2
                offsets = offsets[::, stride]
            else:
                average_duration = record.num_frames // total_frames
                if average_duration >= 1:
                    offsets = np.multiply(list(range(total_frames)), average_duration) + average_duration // 2
                else:
                    need = total_frames - record.num_frames
                    offsets = list(range(record.num_frames)) + [record.num_frames - 1] * need
        offsets = np.array(offsets)
        return offsets if self.On_video else offsets + 1

    def _get(self, record, indices):
        images = list()
        if not self.On_video:
            for idx in indices:
                frame = self._load_image(record.path, idx)
                images.append(frame)
        else:
            if self.root_path.find('101') != -1:
                cap = cv2.VideoCapture(os.path.join(self.root_path, record.path + '.avi'))
            else:
                cap = cv2.VideoCapture(os.path.join(self.root_path, record.path))
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                res, frame = cap.read()
                try:
                    seg_imgs = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    images.append(seg_imgs)
                except:
                    print('Error in read video', os.path.join(self.root_path, record.path), idx, '/',
                          record.num_frames)

            cap.release()
        # processed_data = images
        processed_data = self.transform(images)
        return processed_data, record.label

    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]

    def __len__(self):
        return len(self.video_list)


def get_augmentation(mode, input_size):
    if mode == 'RGB':
        return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66])])
    elif mode == 'Flow':
        return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75])])
    elif mode == 'RGBDiff':
        return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75])])


if __name__ == '__main__':
    data = VideoDataset('/home/qinxin/project/data/sthsth/data', '/home/qinxin/project/data/sthsth/test.txt', None)
    for i in range(10):
        print(data[i][1], len(data[i][0]))
