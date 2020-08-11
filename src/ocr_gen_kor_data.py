from pathlib import Path
import cv2
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import imgaug as ia
import imgaug.augmenters as iaa
from tqdm import tqdm

from base.label import Label, Shape, write_shapes
from base.util import show, draw_label

ocr_kor_path = Path('data/ocr-kor')

random_bg_img_paths = [str(f) for f in (ocr_kor_path / 'random_bg').glob('**/*.jpg')]

kor_font_names = [str(f) for f in (ocr_kor_path / 'font_kor').glob('**/*.ttf')]
kor_font_img_names = [str(ocr_kor_path / 'font_kor_img')]
eng_font_names = [str(f) for f in (ocr_kor_path / 'font_eng').glob('**/*.ttf')]
all_font_names = kor_font_names + kor_font_img_names + eng_font_names

kor_chars_img = '가나다라마거너더러머고노도로모구누두루무버서어저보소오조부수우주허하호바사아자배'
kor_chars_noimg = '울대광산경기충북전제인천세종강원남'
kor_chars = kor_chars_img + kor_chars_noimg
num_chars = '0123456789'
all_chars = kor_chars + num_chars

min_font_size = 34
max_font_size = 40

print("** len(kor_chars)=", len(kor_chars))
print("** len(num_chars)=", len(num_chars))

def get_random_hsv():
    h = np.random.randint(0, 256)
    s = np.random.randint(128, 256)
    v = np.random.randint(128, 256)
    return np.array([h, s, v], dtype=np.uint8)

def bgr_to_rgb(bgr):
    if len(bgr.shape) == 1: # color scalar
        return bgr[[2,1,0]]
    elif len(bgr.shape) == 3: # image
        return bgr[:, :, [2,1,0]]
    else:
        assert(False)

def hsv_to_bgr(hsv):
    hsv = hsv.reshape((1, 1, 3))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape((3,))

def get_random_kor_char():
    char_idx = np.random.randint(0, len(kor_chars))
    char = kor_chars[char_idx]
    return char, char_idx

def get_random_num_char():
    char_idx = np.random.randint(0, len(num_chars))
    char = num_chars[char_idx]
    # assume idx=[0, len(kor_chars) + len(num_chars)]
    return char, char_idx + len(kor_chars)

def get_random_kor_font(no_img_font=False):
    font_names = kor_font_names if no_img_font else kor_font_names + kor_font_img_names
    #font_names = kor_font_names
    #font_names = kor_font_img_names
    return font_names[np.random.randint(0, len(font_names))]

def get_random_num_font():
    font_names = kor_font_names + eng_font_names + kor_font_img_names
    #font_names = kor_font_names + eng_font_names
    #font_names = kor_font_img_names
    return font_names[np.random.randint(0, len(font_names))]

def get_random_font_size():
    return np.random.randint(min_font_size, max_font_size + 1)

def read_kor_font_img(filePath):
    if Path(filePath).exists() == False:
        return None
    with open(filePath.encode("utf-8") , "rb") as f:
        data = bytearray(f.read())
        npdata = np.asarray(data, dtype=np.uint8)
    return cv2.imdecode(npdata , cv2.IMREAD_UNCHANGED)

def gen_char_img_from_kor_font_img(char, font_name, font_size):
    char_image = read_kor_font_img(str(Path(font_name) / (str(char) + ".png")))
    if char_image is None:
        return None

    w, h = char_image.shape[1], char_image.shape[0]
    ratio = float(font_size) / float(h)
    w, h = int(w * ratio), int(h * ratio)
    char_image = cv2.resize(char_image, (w, h),
        interpolation=(cv2.INTER_AREA if ratio < 1. else cv2.INTER_LINEAR))
    
    image = np.zeros([font_size*2, font_size*2, 3], dtype=np.uint8)
    image.fill(255)

    cl = int((image.shape[1] - w) // 2)
    ct = int((image.shape[0] - h) // 2)
    image[ct:ct+h, cl:cl+w] = char_image
    image = cv2.bitwise_not(image)
    return image

def gen_char_img(char, font_name, fallback_font_name, font_size):
    if font_name in kor_font_img_names:
        image = gen_char_img_from_kor_font_img(char, font_name, font_size)
        if image is not None:
            return image
        else:
            font_name = fallback_font_name

    image = Image.new('RGB', (font_size*2, font_size*2), (0, 0, 0))
    font = ImageFont.truetype(font_name, size=font_size)
    draw = ImageDraw.Draw(image)
    draw.text((font_size/2, font_size/2), char, fill=(255, 255, 255), font=font)

    image = np.array(image)
    image = image[:, :, ::-1].copy()
    return image

# assume black bg & white text
def calc_bb(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    valid_rows = np.any(gray > 5, axis = 1)
    valid_cols = np.any(gray > 5, axis = 0)
    t = np.argmax(valid_rows) - 1
    l = np.argmax(valid_cols) - 1
    b = len(valid_rows) - np.argmax(np.flip(valid_rows)) + 1
    r = len(valid_cols) - np.argmax(np.flip(valid_cols)) + 1
    #print("** (l, t, r, b) =", (l, t, r, b))
    return (l, t, r, b)

def shrink_img(image):
    l, t, r, b = calc_bb(image)
    return image[t:b, l:r, :].copy()

def multiply_img(char, image, w_ratio, w_ratio_offset):
    # special character customizing
    if char == '1':
        w_ratio = w_ratio * 0.6
        w_ratio_offset = w_ratio_offset * 0.6

    w, h = float(image.shape[1]), float(image.shape[0])
    w_ratio = np.random.uniform(w_ratio - w_ratio_offset, w_ratio + w_ratio_offset)
    fx = (h * w_ratio) / w
    #print((w, h, w_ratio, fx))
    return cv2.resize(image, dsize=(0, 0), fx=fx, fy=1., interpolation=cv2.INTER_LINEAR)


def augement_img(image):
    images = np.expand_dims(image, axis=0)

    aug = iaa.SomeOf((1, 2),
        [
            iaa.OneOf(
                [
                    iaa.GaussianBlur((0, 1.0)),
                    iaa.AverageBlur(k=(1, 3)),
                    iaa.MotionBlur(k=3, angle=[-45, 45])
                ]),
            iaa.OneOf(
                [
                    iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
                    iaa.EdgeDetect(alpha=(0.0, 0.7)),
                ]),
            iaa.OneOf(
                [
                    #iaa.Clouds(),
                    #iaa.Fog(),
                    iaa.Snowflakes(flake_size=(0.05, 0.1), speed=(0.005, 0.03)),
                    iaa.Canny(alpha=(0.0, 0.2)),
                    # iaa.Dropout(0.04), # maybe included in darknet augmentation??
                    iaa.SaltAndPepper(p=(0.01, 0.04))
                ]
            ),
        ],
        random_order=True)

    images = aug(images=images)
    image = images[0]
    return image

def alpha_blend(bg, fg, alpha_channel, x, y):
    dst = bg.copy()
    w, h = fg.shape[1], fg.shape[0]

    # Convert uint8 to float
    bg = bg[y:y+h, x:x+w].astype(float)
    fg = fg.astype(float)
    alpha_channel = alpha_channel.astype(float) / 255.0 # assume fg.wh == alpah_channel.wh

    # extend dim for alpha_channel
    alpha_channel = [alpha_channel for _ in range(3)]
    alpha_channel = np.stack(alpha_channel, axis=-1)

    # blending
    # inv_alpha = 1.0 - alpha_channel
    # print("** inv_alpha=", inv_alpha.shape, ", bg=", bg.shape)
    bg = cv2.multiply(1.0 - alpha_channel, bg)
    fg = cv2.multiply(alpha_channel, fg)
    dst[y:y+h, x:x+w] = cv2.add(bg, fg)

    return dst

def overlay_char_img(img, char_img, x=0, y=0, bgr=(255., 255., 255.), opacity=1.0):
    alpha_channel = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    alpha_channel = cv2.multiply(opacity, alpha_channel)

    char_alpha_img = np.zeros_like(char_img)
    char_alpha_img[...] = bgr

    return alpha_blend(img, char_alpha_img, alpha_channel, x, y)

def overlay_random_bg(fg, alpha=0.12, idx=0):
    bg_path = random_bg_img_paths[np.random.randint(0, len(random_bg_img_paths))]
    bg = cv2.imread(bg_path)

    fg = fg.astype(float)
    h, w = fg.shape[:2]

    bh, bw = bg.shape[:2]
    bt, bl = np.random.randint(0, bh - h), np.random.randint(0, bw - w)
    bg = bg[bt:bt+h, bl:bl+w, :].astype(float)

    return (fg * (1.0 - alpha) + bg * alpha).astype(np.uint8)

def warp_affine_pts(mat, pts):
    def _pts2ptsh(pts):
        return np.matrix(np.concatenate((pts, np.ones((1, pts.shape[1]))), 0))
    ptsh = _pts2ptsh(pts) # (2, 4) -> (3, 4)
    return np.matmul(mat, ptsh) # (2, 3) * (3, 4)

# cx, cx_offset, cy, cy_offset, font_size, font_size_offset, isNumber, w_ratio
sync_template_0 = [ # 7자리 1줄 (12가3456)
    ( 48,        3, 48, 2, 40, 2, True,  0.5 ),
    ( 48 + 32*1, 3, 48, 2, 40, 2, True,  0.5 ),
    ( 48 + 32*2, 3, 48, 2, 40, 2, False, 0.5 ),
    ( 48 + 32*3, 3, 48, 2, 40, 2, True,  0.5 ),
    ( 48 + 32*4, 3, 48, 2, 40, 2, True,  0.5 ),
    ( 48 + 32*5, 3, 48, 2, 40, 2, True,  0.5 ),
    ( 48 + 32*6, 3, 48, 2, 40, 2, True,  0.5 ),
]

sync_template_1 = [ # 7자리 2줄 (12가 / 3456)
    ( 144 - 40,   3, 25,      2, 24, 2, True,  1.2 ),
    ( 144     ,   3, 25,      2, 24, 2, True,  1.2 ),
    ( 144 + 40,   3, 25,      2, 24, 2, False, 1.2 ),
    ( 144 - 26*3, 3, 48 + 12, 2, 40, 2, True,  1.0 ),
    ( 144 - 26*1, 3, 48 + 12, 2, 40, 2, True,  1.0 ),
    ( 144 + 26*1, 3, 48 + 12, 2, 40, 2, True,  1.0 ),
    ( 144 + 26*3, 3, 48 + 12, 2, 40, 2, True,  1.0 ),
]

sync_template_2 = [ # 택시 : 지역 2자리 + 7자리 2줄 (경기12가2345)
    ( 38 + 30*0, 0, 48 - 16, 0, 20, 0, False, 1.0 ),
    ( 38 + 30*0, 0, 48 + 16, 0, 20, 0, False, 1.0 ),
    ( 38 + 30*1, 3, 48     , 2, 44, 2, True,  0.42 ),
    ( 38 + 30*2, 3, 48     , 2, 44, 2, True,  0.42 ),
    ( 38 + 30*3, 3, 48     , 2, 44, 2, False, 0.42 ),
    ( 38 + 30*4, 3, 48     , 2, 44, 2, True,  0.42 ),
    ( 38 + 30*5, 3, 48     , 2, 44, 2, True,  0.42 ),
    ( 38 + 30*6, 3, 48     , 2, 44, 2, True,  0.42 ),
    ( 38 + 30*7, 3, 48     , 2, 44, 2, True,  0.42 ),
]

sync_template_3 = [ # 9자리 2줄 (전북86 / 사3456)
    ( 144 - 45,   1, 25,      1, 30, 2, False, 1.0 ),
    ( 144 - 15,   1, 25,      1, 30, 2, False, 1.0 ),
    ( 144 + 15,   1, 25,      1, 30, 2, True,  0.8 ),
    ( 144 + 45,   1, 25,      1, 30, 2, True,  0.8 ),
    ( 144 - 48*2, 3, 48 + 12, 2, 36, 2, False, 1.0 ),
    ( 144 - 48*1, 3, 48 + 12, 2, 36, 2, True,  0.9 ),
    ( 144 - 48*0, 3, 48 + 12, 2, 36, 2, True,  0.9 ),
    ( 144 + 48*1, 3, 48 + 12, 2, 36, 2, True,  0.9 ),
    ( 144 + 48*2, 3, 48 + 12, 2, 36, 2, True,  0.9 ),
]

sync_templates = [
    sync_template_0,
    sync_template_1,
    sync_template_2,
    sync_template_3,
]

def random_template():
    return sync_templates[np.random.randint(0, len(sync_templates))]

def check_h_diff(c0, c1, min_diff):
    min_h = min(int(c0[0]), int(c1[0]))
    max_h = max(int(c0[0]), int(c1[0]))
    return (max_h - min_h < min_diff) or (min_h + 256 - max_h < min_diff)

def gen_lp_image(template, idx=0):
    bg_color = get_random_hsv()
    font_color = get_random_hsv()
    while check_h_diff(bg_color, font_color, 96) == False:
        font_color = get_random_hsv()
    bg_color = hsv_to_bgr(bg_color)
    font_color = hsv_to_bgr(font_color)
    font_color_f = tuple([float(v) for v in font_color])

    w, h = 288, 96
    image = np.array(Image.new('RGB', (w, h), tuple(bgr_to_rgb(bg_color))))
    image = image[:, :, ::-1].copy()

    kor_font_name = get_random_kor_font()
    kor_font_fallback_name = get_random_kor_font(no_img_font=True)
    num_font_name = get_random_num_font()

    random_x_offset = np.random.randint(0, 30) - 15
    char_idx_list = []
    char_pts_list = []

    def _dice_offset(v):
        return np.random.randint(-v,  v + 1) if v > 0 else 0

    for tc in template:
        cx, cx_offset, cy, cy_offset, font_size, font_size_offset, isNumber, w_ratio = tc

        font_size = font_size + _dice_offset(font_size_offset)

        if isNumber:
            char, char_idx = get_random_num_char()
            char_image = gen_char_img(char, num_font_name, None, font_size)
        else:
            char, char_idx = get_random_kor_char()
            char_image = gen_char_img(char, kor_font_name, kor_font_fallback_name, font_size)

        char_image = shrink_img(char_image)
        char_image = multiply_img(char, char_image, w_ratio, 0.12)
        cw, ch = char_image.shape[1], char_image.shape[0]

        cx = cx + _dice_offset(cx_offset)
        cy = cy + _dice_offset(cy_offset)
        l, t = cx - cw // 2 + random_x_offset, cy - ch // 2
        image = overlay_char_img(image, char_image, l, t, font_color_f)

        char_idx_list.append(char_idx)
        char_pts = np.array([[l, l + cw, l + cw, l], [t, t, t + ch, t + ch]]).astype(np.float)
        char_pts_list.append(char_pts)

    # rotation
    rotation_degree_max = 8.
    rotation_degree = np.random.uniform(-rotation_degree_max, rotation_degree_max)
    rotation_mat = cv2.getRotationMatrix2D((w / 2, h / 2), rotation_degree, 1.0)
    rotation_borderValue = tuple([int(v) for v in bg_color])
    image = cv2.warpAffine(image, rotation_mat, (w, h), borderValue=rotation_borderValue)
    char_pts_list = [warp_affine_pts(rotation_mat, pts) for pts in char_pts_list]

    # overlay random bg image
    image = overlay_random_bg(image, idx=idx)

    # augment image
    image = augement_img(image)

    # convert to shapes
    shapes = []
    for pts, char_idx in zip(char_pts_list, char_idx_list):
        pts_prop = pts / np.array([float(w), float(h)]).reshape((2, 1))
        pts_prop = np.asarray(np.clip(pts_prop, 0.0, 1.0))
        shapes.append(Shape(pts=pts_prop, text=str(char_idx)))

    return image, shapes

def show_image_with_shapes(image, shapes):
    image_to_show = image.copy()
    for shape in shapes:
        draw_label(image_to_show, shape.pts)
    show(image_to_show)

def write_darknet_label(path, shapes):
    if len(shapes):
        with open(path, 'w') as fp:
            for shape in shapes:
                if shape.is_valid():
                    pts = shape.pts
                    l, t = np.amin(pts, 1)
                    r, b = np.amax(pts, 1)
                    #print(l, t, r, b)
                    cx, cy = (l + r) * 0.5, (t + b) * 0.5
                    w, h = r - l, b - t
                    fp.write(f'{shape.text} {cx} {cy} {w} {h}\n')

def gen_data(base_path, idx):
    #image, shapes = gen_lp_image_old(idx)
    image, shapes = gen_lp_image(random_template(), idx)
    base_path = Path(base_path)
    cv2.imwrite(str(base_path / (str(idx).zfill(5) + ".jpg")), image)
    write_shapes(str(base_path / (str(idx).zfill(5) + "_shapes.txt")), shapes)
    write_darknet_label(str(base_path / (str(idx).zfill(5) + ".txt")), shapes)

def gen_dataset(base_path, cnt=20):
    Path(base_path).mkdir(parents=True, exist_ok=True)
    for idx in tqdm(range(cnt)):
        gen_data(base_path, idx)

if __name__ == "__main__":
    #gen_dataset('tmp/sample/', 20)
    gen_dataset('_train_ocr/dataset/synth/train/', 12000)
    gen_dataset('_train_ocr/dataset/synth/val/', 3000)

