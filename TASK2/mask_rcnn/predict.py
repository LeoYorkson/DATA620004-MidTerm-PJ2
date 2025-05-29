import os
import time
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs, draw_comparison
# from draw_box_utils import draw_objs


def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(idx=None, comparison=False):
    num_classes = 20  # 不包含背景
    box_thresh = 0.5
    weights_path = "./save_weights/model_14.pth"
    # img_path = "./demo.jpg"
    # img_path = "./vis/3_beyond_data/white_cat&sofa.jpg"
    img_path = f'./vis/4_within_data/2007_000{idx}.jpg'
    label_json_path = './pascal_voc_indices.json'

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    # load train weights
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    # load image
    assert os.path.exists(img_path), f"{img_path} does not exits."
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions, proposals = model(img.to(device))
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))
        
        proposal_boxes = proposals[0].to("cpu").numpy()  # RPN proposals
        predict_boxes = predictions[0]["boxes"].to("cpu").numpy()
        predict_classes = predictions[0]["labels"].to("cpu").numpy()
        predict_scores = predictions[0]["scores"].to("cpu").numpy()
        predict_mask = predictions[0]["masks"].to("cpu").numpy()
        predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")
            return

        if comparison == True:
            result_image = draw_comparison(
                    original_img,
                    proposals=proposals[0].cpu().numpy(),
                    final_boxes=predict_boxes,
                    final_classes=predict_classes,
                    final_scores=predict_scores,
                    final_masks= predict_mask,
                    category_index=category_index
                )
        else:
            result_image = draw_objs(original_img,
                             boxes=predict_boxes,
                             classes=predict_classes,
                             scores=predict_scores,
                             masks=predict_mask,
                             category_index=category_index,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        # plt.imshow(plot_img)
        plt.imshow(result_image)
        plt.show()
        # 保存预测的图片结果
        # plot_img.save("test_result.jpg")
        # result_image.save("./vis/3_beyond_data/mask_rcnn/white_cat&sofa.jpg")
        result_image.save(f"./vis/4_within_data/mask_rcnn/2007_000{idx}.jpg")


if __name__ == '__main__':
    for idx in ['061','123','170','187','250','464']:
        
        main(idx, comparison=True)

