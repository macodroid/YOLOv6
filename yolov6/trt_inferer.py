import os
from collections import namedtuple, OrderedDict

import numpy as np

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import tensorrt as trt
import torch

from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression


class TrtInferer:
    def __init__(self, trt_model, image_size, device='0', stride=32, half=True):
        self.trt_model = trt_model
        self.image_size = image_size
        self.device = device
        cuda = self.device != 'cpu' and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{device}' if cuda else 'cpu')
        self.stride = stride
        self.half = half

        self.context, self.bindings, self.binding_addrs, self.trt_batch_size = self.init_trt_engine()

    def process_images_numpy(self, img_src, img_size, stride, half):
        '''Process image before image inference.'''
        image = letterbox(img_src, img_size, stride=stride)[0]
        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = np.ascontiguousarray(image)
        image = image.astype(float)
        image /= 255  # 0 - 255 to 0.0 - 1.0
        return image, img_src

    def process_image(self, img_src, img_size, stride, half):
        '''Process image before image inference.'''
        image = letterbox(img_src, img_size, stride=stride)[0]
        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src

    def process_batch_images_numpy(self, images, img_size, stride, half):
        processed_imgs = []
        imgs = []
        if len(images.shape) == 3:
            img, img_src = self.process_images_numpy(images, img_size, stride, half)
            return img, img_src
        for img in images:
            img, img_src = self.process_images_numpy(img, img_size, stride, half)
            processed_imgs.append(img)
            imgs.append(img_src)
        return np.stack(processed_imgs, axis=0), np.array(imgs)

    def process_batch_images(self, images, img_size, stride, half):
        processed_imgs = []
        imgs = []
        if len(images.shape) == 3:
            img, img_src = self.process_image(images, img_size, stride, half)
            return img, img_src
        for img in images:
            img, img_src = self.process_image(img, img_size, stride, half)
            processed_imgs.append(img)
            imgs.append(img_src)
        return torch.stack(processed_imgs, axis=0), np.array(imgs)

    def init_trt_engine(self):
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with open(self.trt_model, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        bindings = OrderedDict()
        for index in range(model.num_bindings):
            name = model.get_tensor_name(index)
            dtype = trt.nptype(model.get_tensor_dtype(name))
            shape = tuple(model.get_tensor_shape(name))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        context = model.create_execution_context()
        return context, bindings, binding_addrs, model.get_tensor_shape("images")[0]

    def rescale(self, ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes

    def infer(self, images):
        bboxs, fubs = [], []
        # Process images
        processed_imgs, imgs = self.process_batch_images(images, self.image_size, self.stride, self.half)
        # processed_imgs = torch.from_numpy(processed_imgs).contiguous().to(self.device)
        processed_imgs = processed_imgs.contiguous().to(self.device)
        # Copy to device
        self.binding_addrs['images'] = int(processed_imgs.data_ptr())
        # Inference
        self.context.execute_v2(list(self.binding_addrs.values()))
        # Copy to host
        output = [self.bindings[n].data for n in self.bindings.keys()]
        pred_results = (output[2], output[1])
        det, fub = non_max_suppression(pred_results, 0.65, 0.55, max_det=1000)

        for i in range(len(det)):
            if len(det[i]):
                det[i][:, :4] = self.rescale(processed_imgs.shape[2:], det[i][:, :4], imgs[i].shape).round()
            bboxs.append(det[i][:, :4].detach().cpu().numpy())
            fubs.append(fub[i].detach().cpu().numpy())
        return bboxs, fubs
