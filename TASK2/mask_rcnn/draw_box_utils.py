from PIL.Image import Image, fromarray
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from PIL import ImageColor
import numpy as np
from PIL import Image

# STANDARD_COLORS = [
#     'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
#     'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
#     'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
#     'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
#     'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
#     'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
#     'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
#     'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
#     'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
#     'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
#     'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
#     'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
#     'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
#     'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
#     'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
#     'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
#     'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
#     'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
#     'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
#     'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
#     'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
#     'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
#     'WhiteSmoke', 'Yellow', 'YellowGreen'
# ]

STANDARD_COLORS = [
    # 人、动物类（暖色调）
    'Gold',                  # 0: person (金色)
    'Tomato',                # 1: bird (番茄红)
    'DarkOrange',            # 2: cat (深橙)
    'Peru',                  # 3: cow (秘鲁棕)
    'SaddleBrown',           # 4: dog (鞍棕色)
    'Chocolate',             # 5: horse (巧克力色)
    'PaleVioletRed',         # 6: sheep (苍紫罗兰红)

    # 车辆类（冷色调）
    'DodgerBlue',            # 7: aeroplane (道奇蓝)
    'SteelBlue',             # 8: bicycle (钢蓝)
    'RoyalBlue',             # 9: boat (皇家蓝)
    'DeepSkyBlue',           # 10: bus (深天蓝)
    'CornflowerBlue',        # 11: car (矢车菊蓝)
    'LightSlateGray',        # 12: motorbike (亮石板灰)

    # 室内物体（中性/混合色）
    'MediumOrchid',          # 13: bottle (中兰花紫)
    'Plum',                  # 14: chair (李子色)
    'DarkMagenta',           # 15: dining table (深洋红)
    'DarkKhaki',             # 16: potted plant (深卡其)
    'LightSeaGreen',         # 17: sofa (浅海洋绿)
    'DarkCyan',              # 18: tv/monitor (深青色)
    'DarkSalmon',            # 19: train (深鲑红)
]


# def draw_text(draw,
#               box: list,
#               cls: int,
#               score: float,
#               category_index: dict,
#               color: str,
#               font: str = 'arial.ttf',
#               font_size: int = 24):
#     """
#     将目标边界框和类别信息绘制到图片上
#     """
#     try:
#         font = ImageFont.truetype(font, font_size)
#     except IOError:
#         font = ImageFont.load_default()

#     left, top, right, bottom = box
#     # If the total height of the display strings added to the top of the bounding
#     # box exceeds the top of the image, stack the strings below the bounding box
#     # instead of above.
#     display_str = f"{category_index[str(cls)]}: {int(100 * score)}%"
#     display_str_heights = [font.getsize(ds)[1] for ds in display_str]
#     # Each display_str has a top and bottom margin of 0.05x.
#     display_str_height = (1 + 2 * 0.05) * max(display_str_heights)

#     if top > display_str_height:
#         text_top = top - display_str_height
#         text_bottom = top
#     else:
#         text_top = bottom
#         text_bottom = bottom + display_str_height

#     for ds in display_str:
#         text_width, text_height = font.getsize(ds)
#         margin = np.ceil(0.05 * text_width)
#         draw.rectangle([(left, text_top),
#                         (left + text_width + 2 * margin, text_bottom)], fill=color)
#         draw.text((left + margin, text_top),
#                   ds,
#                   fill='black',
#                   font=font)
#         left += text_width


# def draw_masks(image, masks, colors, thresh: float = 0.7, alpha: float = 0.5):
#     np_image = np.array(image)
#     masks = np.where(masks > thresh, True, False)

#     # colors = np.array(colors)
#     img_to_draw = np.copy(np_image)
#     # TODO: There might be a way to vectorize this
#     for mask, color in zip(masks, colors):
#         img_to_draw[mask] = color

#     out = np_image * (1 - alpha) + img_to_draw * alpha
#     return fromarray(out.astype(np.uint8))

def draw_text(draw,
              box: list,
              cls: int,
              score: float,
              category_index: dict,
              color: str,
              font: str = 'arial.ttf',
              font_size: int = 24):
    """
    将目标边界框和类别信息绘制到图片上
    """
    try:
        font = ImageFont.truetype(font, font_size)
    except IOError:
        font = ImageFont.load_default()

    left, top, right, bottom = box
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str = f"{category_index[str(cls)]}: {int(100 * score)}%"
    # display_str_heights = [font.getsize(ds)[1] for ds in display_str]
    display_str_heights = [font.getbbox(ds)[3] - font.getbbox(ds)[1] for ds in display_str]
    # Each display_str has a top and bottom margin of 0.05x.
    display_str_height = (1 + 2 * 0.05) * max(display_str_heights)

    if top > display_str_height:
        text_top = top - display_str_height
        text_bottom = top
    else:
        text_top = bottom
        text_bottom = bottom + display_str_height

    # for ds in display_str:
    #     text_width, text_height = font.getsize(ds)
    #     margin = np.ceil(0.05 * text_width)
    #     draw.rectangle([(left, text_top),
    #                     (left + text_width + 2 * margin, text_bottom)], fill=color)
    #     draw.text((left + margin, text_top),
    #               ds,
    #               fill='black',
    #               font=font)
    #     left += text_width
    for ds in display_str:
        bbox = font.getbbox(ds)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        margin = np.ceil(0.05 * text_width)
        draw.rectangle([(left, text_top),
                        (left + text_width + 2 * margin, text_bottom)], fill=color)
        draw.text((left + margin, text_top),
                ds,
                fill='black',
                font=font)
        left += text_width + 2 * margin



def draw_masks(image, masks, colors, thresh: float = 0.7, alpha: float = 0.5):
    np_image = np.array(image)
    masks = np.where(masks > thresh, True, False)

    # colors = np.array(colors)
    img_to_draw = np.copy(np_image)
    # TODO: There might be a way to vectorize this
    for mask, color in zip(masks, colors):
        img_to_draw[mask] = color

    out = np_image * (1 - alpha) + img_to_draw * alpha
    return fromarray(out.astype(np.uint8))


# def draw_objs(image: Image,
#               boxes: np.ndarray = None,
#               classes: np.ndarray = None,
#               scores: np.ndarray = None,
#               masks: np.ndarray = None,
#               category_index: dict = None,
#               box_thresh: float = 0.1,
#               mask_thresh: float = 0.5,
#               line_thickness: int = 8,
#               font: str = 'arial.ttf',
#               font_size: int = 24,
#               draw_boxes_on_image: bool = True,
#               draw_masks_on_image: bool = True):
#     """
#     将目标边界框信息，类别信息，mask信息绘制在图片上
#     Args:
#         image: 需要绘制的图片
#         boxes: 目标边界框信息
#         classes: 目标类别信息
#         scores: 目标概率信息
#         masks: 目标mask信息
#         category_index: 类别与名称字典
#         box_thresh: 过滤的概率阈值
#         mask_thresh:
#         line_thickness: 边界框宽度
#         font: 字体类型
#         font_size: 字体大小
#         draw_boxes_on_image:
#         draw_masks_on_image:

#     Returns:

#     """

#     # 过滤掉低概率的目标
#     idxs = np.greater(scores, box_thresh)
#     boxes = boxes[idxs]
#     classes = classes[idxs]
#     scores = scores[idxs]
#     if masks is not None:
#         masks = masks[idxs]
#     if len(boxes) == 0:
#         return image

#     colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in classes]

#     if draw_boxes_on_image:
#         # Draw all boxes onto image.
#         draw = ImageDraw.Draw(image)
#         for box, cls, score, color in zip(boxes, classes, scores, colors):
#             left, top, right, bottom = box
#             # 绘制目标边界框
#             draw.line([(left, top), (left, bottom), (right, bottom),
#                        (right, top), (left, top)], width=line_thickness, fill=color)
#             # 绘制类别和概率信息
#             draw_text(draw, box.tolist(), int(cls), float(score), category_index, color, font, font_size)

#     if draw_masks_on_image and (masks is not None):
#         # Draw all mask onto image.
#         image = draw_masks(image, masks, colors, mask_thresh)

#     return image
def draw_objs(image: Image,
              boxes: np.ndarray = None,
              classes: np.ndarray = None,
              scores: np.ndarray = None,
              masks: np.ndarray = None,
              category_index: dict = None,
              box_thresh: float = 0.1,
              mask_thresh: float = 0.5,
              line_thickness: int = 4,
              font: str = 'arial.ttf',
              font_size: int = 24,
              draw_boxes_on_image: bool = True,
              draw_masks_on_image: bool = True):
    
    if boxes is None or len(boxes) == 0:
        return image

    if scores is not None:
        idxs = np.greater(scores, box_thresh)
        boxes = boxes[idxs]
        classes = classes[idxs] if classes is not None else None
        scores = scores[idxs]
        if masks is not None:
            masks = masks[idxs]

    if len(boxes) == 0:
        return image

    colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in classes] if classes is not None else [(255, 0, 0)] * len(boxes)

    if draw_boxes_on_image:
        draw = ImageDraw.Draw(image)
        for i, box in enumerate(boxes):
            left, top, right, bottom = box
            color = colors[i]
            draw.line([(left, top), (left, bottom), (right, bottom),
                       (right, top), (left, top)], width=line_thickness, fill=color)
            if classes is not None and scores is not None:
                draw_text(draw, box.tolist(), int(classes[i]), float(scores[i]), category_index, color, font, font_size)
    if draw_masks_on_image and masks is not None:
        # print(colors)
        image = draw_masks(image, masks, colors, mask_thresh)

    return image

def draw_comparison(image: Image,
                    proposals: np.ndarray,
                    final_boxes: np.ndarray,
                    final_classes: np.ndarray,
                    final_scores: np.ndarray,
                    final_masks: np.ndarray = None,
                    category_index: dict = None,
                    font: str = "arial.ttf",
                    font_size: int = 20):
    """
    显示 proposals（左图）和最终检测结果（右图）的对比。
    """

    image_left = image.copy()
    image_right = image.copy()

    # 左边：只画 proposal 框（不含类别信息）
    image_left = draw_objs(image_left,
                           boxes=proposals,
                           classes=None,
                           scores=None,
                           draw_boxes_on_image=True,
                           draw_masks_on_image=False)

    # 右边：画最终检测结果（框+类别+分数+mask）
    image_right = draw_objs(image_right,
                            boxes=final_boxes,
                            classes=final_classes,
                            scores=final_scores,
                            masks=final_masks,
                            category_index=category_index,
                            box_thresh=0.1,
                            font=font,
                            font_size=font_size,
                            draw_boxes_on_image=True,
                            draw_masks_on_image=True)

    # 拼接图片
    new_width = image_left.width + image_right.width
    new_image = Image.new("RGB", (new_width, image_left.height))
    new_image.paste(image_left, (0, 0))
    new_image.paste(image_right, (image_left.width, 0))

    return new_image