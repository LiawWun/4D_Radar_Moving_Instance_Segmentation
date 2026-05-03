import numpy as np

import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, coord, feat, label, center):
        for t in self.transforms:
            coord, feat, label, center = t(coord, feat, label, center)
        return coord, feat, label, center


class ToTensor(object):
    def __call__(self, coord, feat, label):
        coord = torch.from_numpy(coord)
        if not isinstance(coord, torch.FloatTensor):
            coord = coord.float()
        feat = torch.from_numpy(feat)
        if not isinstance(feat, torch.FloatTensor):
            feat = feat.float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        return coord, feat, label


class RandomRotate(object):

    def __init__(self, angle=[0.00, 0.00, 0.05], prob = 0.5):
        self.angle = angle
        self.prob = prob

    def __call__(self, coord, feat, label, center):

        if np.random.uniform() <= self.prob:
            angle_x = np.random.uniform(-self.angle[0], self.angle[0]) * np.pi
            angle_y = np.random.uniform(-self.angle[1], self.angle[1]) * np.pi
            angle_z = np.random.uniform(-self.angle[2], self.angle[2]) * np.pi
            cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
            cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
            cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
            R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
            R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
            R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
            R = np.dot(R_z, np.dot(R_y, R_x))

            coord = np.dot(coord, np.transpose(R))
            center = np.dot(center, np.transpose(R))

            coord = coord.astype(np.float32)
            feat = feat.astype(np.float32)
            label = label.astype(np.float32)

        return coord, feat, label, center


class RandomScale(object):
    def __init__(self, scale=[0.9, 1.1], anisotropic=False, prob = 0.5):
        self.scale = scale
        self.prob = prob
        self.anisotropic = anisotropic

    def __call__(self, coord, feat, label, center):
        if np.random.uniform() < self.prob:
            scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
            coord *= scale
            center *= scale
        return coord, feat, label, center


class RandomShift(object):
    def __init__(self, shift=[0.2, 0.2, 0.1], prob = 0.5):
        self.shift = shift
        self.prob = prob
    def __call__(self, coord, feat, label, center):
        if np.random.uniform() < self.prob:
            shift_x = np.random.uniform(-self.shift[0], self.shift[0])
            shift_y = np.random.uniform(-self.shift[1], self.shift[1])
            shift_z = np.random.uniform(-self.shift[2], self.shift[2])
            coord += [shift_x, shift_y, shift_z]
            center += [shift_x, shift_y, shift_z]
        return coord, feat, label, center


class RandomFlip(object):
    def __init__(self, prob=0.5):
        self.p = prob

    def __call__(self, coord, feat, label, center):
        '''if np.random.rand() < self.p:
            coord[:, 0] = -coord[:, 0]'''
        if np.random.rand() < self.p:
            coord[:, 1] = -coord[:, 1]
            center[:, 1] = -center[:, 1]
        return coord, feat, label, center
    
class RandomJitterOnFeat(object):
    def __init__(self, prob = 0.5, portion = 0.2):
        self.prob = prob
        self.portion = portion
    
    def __call__(self, coord, feat, label, center):
        
        if np.random.uniform() < self.prob:
            num_points = coord.shape[0]
            feat_std = np.std(feat, axis = 0)

            random_num = int(num_points * self.portion)
            random_index = np.random.randint(0, num_points, random_num)
            feat_jitter = np.zeros(feat.shape)
            feat_jitter[random_index, :] = np.random.normal(loc = np.zeros(feat.shape[1]), scale = feat_std)
            feat += feat_jitter

        return coord, feat, label, center

class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05, prob = 0.5):
        self.sigma = sigma
        self.clip = clip
        self.prob = prob

    def __call__(self, coord, feat, label, center):
        assert (self.clip > 0)
        if np.random.uniform() < self.prob:
            jitter = np.clip(self.sigma * np.random.randn(coord.shape[0], 3), -1 * self.clip, self.clip)
            coord += jitter
            center += jitter
        return coord, feat, label, center


def test_argumentation():

    num_points = 10
    raw_pc_xyz = np.random.rand(num_points, 3)
    raw_pc_feat = np.random.rand(num_points, 2)
    raw_pc_label = np.random.rand(num_points, 1)

    argumentation = RandomJitterOnFeat(prob = 0.7, portion = 0.2)
    argumentation.__call__(raw_pc_xyz, raw_pc_feat, raw_pc_label)

    print("Point Cloud XYZ")
    print(raw_pc_xyz)
    print("Point Cloud Feat")
    print(raw_pc_feat)


if __name__ == "__main__":

    test_argumentation()