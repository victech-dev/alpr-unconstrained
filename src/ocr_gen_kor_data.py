from pathlib import Path
import cv2
import numpy as np
import PIL
from tqdm import tqdm
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug import parameters as iap
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

from base.label import Label, Shape, write_shapes, lwrite
import fontlib
import postprocess

# TODO swhich to argparse, fire or something
ocr_kor_path = Path('data/ocr-kor')
REAL_REGION_PROB = 1.0
MIN_COLOR_DIFF = 20 # 0~100
IMSHOW = True

random_bg_img_paths = [str(f) for f in (ocr_kor_path / 'random_bg').glob('**/*.jpg')]

kor_chars_cat = '가나다라마거너더러머고노도로모구누두루무버서어저보소오조부수우주허하호바사아자배'
kor_chars = kor_chars_cat + '울대광산경기충북전제인천세종강원남'
num_chars = '0123456789'
all_chars = kor_chars + num_chars
kor_regions = ['서울', '대구', '광주', '울산', '경기', '충북', '전북', '경북', '제주', '부산', '인천', '대전', '세종', '강원', '충남', '전남', '경남']

print("** len(kor_chars)=", len(kor_chars))
print("** len(num_chars)=", len(num_chars))

img_font = fontlib.ImgFont(ocr_kor_path/'font_kor_img', True)
kor_fonts = [fontlib.TrueTypeFont(str(f), [kor_chars, num_chars]) for f in (ocr_kor_path / 'font_kor').glob('**/*.*')]
eng_fonts = [fontlib.TrueTypeFont(str(f), [num_chars]) for f in (ocr_kor_path / 'font_eng').glob('**/*.*')]

def get_random_kor_font(no_img_font=False):
    fonts = kor_fonts + [] if no_img_font else [img_font]
    return np.random.choice(fonts)

def get_random_num_font():
    fonts = kor_fonts + eng_fonts + [img_font]
    return np.random.choice(fonts)

def get_random_image(width, height, scale=None):
    img = cv2.imread(np.random.choice(random_bg_img_paths))
    aug_list = [iaa.CropToFixedSize(width=width, height=height)]
    if scale is not None:
        aug_list.append(iaa.Affine(scale=scale))
    aug = aug_list[0] if len(aug_list) == 1 else iaa.Sequential(aug_list)
    return aug(image=img)

def random_overlay(scale=None):
    def _func_images(images, random_state, parents, hooks):
        return [get_random_image(img.shape[1], img.shape[0], scale=scale) for img in images]
    return iaa.Lambda(func_images=_func_images)

class ShadowMaskGen(iaa.IBatchwiseMaskGenerator):
    def __init__(self, mul=1.0, add=0.0):
        self.mul = iap.handle_continuous_param(
            mul, "mul", value_range=(0.0, 1.0),
            tuple_to_uniform=True, list_to_choice=True)
        self.add = iap.handle_continuous_param(
            add, "add", value_range=(-1.0, 1.0),
            tuple_to_uniform=True, list_to_choice=True)
    def draw_masks(self, batch, random_state=None):
        shapes = batch.get_rowwise_shapes()
        return [self._draw_mask(shape, random_state) for shape in shapes]
    def _draw_mask(self, shape, random_state):
        while True:
            src = get_random_image(shape[1], shape[0], scale=(4, 8))
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            gray = gray.astype(np.float32) / 255
            if gray.max() < 1e-2:
                continue
            gray /= gray.max()
            if np.var(gray) > 0.1:
                break
        mul = self.mul.draw_samples(1, random_state=random_state)
        add = self.add.draw_samples(1, random_state=random_state)
        return np.clip(gray * mul + add, 0, 1)

def augement_img(image, polygon):
    aug = iaa.Sequential([
        # background overlay
        iaa.BlendAlpha((0.75, 1.0), background=random_overlay()),
        # blur effect
        iaa.Sometimes(0.3, iaa.OneOf([
            iaa.GaussianBlur((0, 1.0)),
            iaa.AverageBlur(k=(1, 3)),
            iaa.MotionBlur(k=3, angle=[-45, 45]),
            iaa.imgcorruptlike.DefocusBlur(severity=1),
        ])),
        iaa.SomeOf((0, 2),
            [
                iaa.OneOf([
                    iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
                    iaa.EdgeDetect(alpha=(0.0, 0.3)),
                    iaa.Canny(alpha=(0.0, 0.3))
                ]),
                iaa.OneOf([
                    iaa.Identity(),
                    # iaa.Clouds(),
                    # iaa.imgcorruptlike.Fog(severity=1),
                    # iaa.imgcorruptlike.Frost(severity=1),
                    iaa.imgcorruptlike.Snow(severity=1),
                    # iaa.imgcorruptlike.Spatter(severity=(1, 5)),
                    # iaa.Dropout(0.04), # maybe included in darknet augmentation??
                    iaa.SaltAndPepper(p=(0.01, 0.04))
                ]),
                # elastic transform
                iaa.OneOf([
                    iaa.ElasticTransformation(alpha=(5,10), sigma=3),
                    iaa.ElasticTransformation(alpha=(10,20), sigma=5)
                ])
            ],
            random_order=True),
        # simplex noise
        iaa.Sometimes(0.3, iaa.BlendAlphaSimplexNoise(
            random_overlay(), size_px_max=(16,32), upscale_method='cubic',
            iterations=1, sigmoid=True, sigmoid_thresh=(4,8))),
        # shadow or large black blob
        iaa.Sometimes(0.3, iaa.BlendAlphaMask(ShadowMaskGen(mul=(0.5, 1), add=(-0.3, -0.1)), iaa.Multiply(0))),
        # brightness
        iaa.Sometimes(0.3, iaa.imgcorruptlike.Brightness(severity=(1, 3)))
    ])

    return aug(image=image, polygons=polygon)

def random_num_text(count):
    return ''.join([np.random.choice(list(num_chars)) for _ in range(count)])

# 1줄 (12가3456)
def gen_kor_lp_0(fonts, n=None):
    if n is None:
        n = np.random.choice([2, 3])
    text = random_num_text(n) + np.random.choice(list(kor_chars_cat)) + '__' + random_num_text(4)
    img, bbsoi = fontlib.hdraw_text(text, 100, fonts)
    return iaa.Resize(dict(width=(0.4, 0.8)))(image=img, bounding_boxes=bbsoi)

# 2줄 (12가 / 3456)
def gen_kor_lp_1(fonts):
    text1 = random_num_text(2) + '_' + np.random.choice(list(kor_chars_cat))
    text2 = random_num_text(4)
    img1, bbsoi1 = fontlib.hdraw_text(text1, 100, fonts)
    img1, bbsoi1 = iaa.Resize((0.4, 0.8))(image=img1, bounding_boxes=bbsoi1)
    img1, bbsoi1 = iaa.Resize(dict(width=(0.8, 1.0)))(image=img1, bounding_boxes=bbsoi1)
    img2, bbsoi2 = fontlib.hdraw_text(text2, 100, fonts)
    img2, bbsoi2 = iaa.Resize(dict(width=(0.6, 0.9)))(image=img2, bounding_boxes=bbsoi2)
    return fontlib.vstack_center_aligned(img1, bbsoi1, img2, bbsoi2, np.random.randint(0, 10))

# 2줄 (경기 / 12가2345)
def gen_kor_lp_2(fonts):
    text1 = np.random.choice(kor_regions)
    text1 = text1[0] + '___' + text1[1]
    text2 = np.random.choice(list(kor_chars_cat)) + '_' + random_num_text(4)
    img1, bbsoi1 = fontlib.hdraw_text(text1, 100, fonts)
    img1, bbsoi1 = iaa.Resize((0.8, 1.0))(image=img1, bounding_boxes=bbsoi1)
    img1, bbsoi1 = iaa.Resize(dict(width=(1.0, 1.3)))(image=img1, bounding_boxes=bbsoi1) 
    img2, bbsoi2 = fontlib.hdraw_text(text2, 100, fonts)
    img2, bbsoi2 = iaa.Resize(dict(width=(0.6, 0.9)))(image=img2, bounding_boxes=bbsoi2)
    return fontlib.vstack_center_aligned(img1, bbsoi1, img2, bbsoi2, np.random.randint(0, 10))

# 2줄 (경기12 / 가2345)
def gen_kor_lp_3(fonts, n=None):
    if n is None:
        n = np.random.choice([1,2])
    text1 = np.random.choice(kor_regions) + '_' + random_num_text(n)
    text2 = np.random.choice(list(kor_chars_cat)) + '_' + random_num_text(4)
    img1, bbsoi1 = fontlib.hdraw_text(text1, 100, fonts)
    img1, bbsoi1 = iaa.Resize((0.4, 0.8))(image=img1, bounding_boxes=bbsoi1)
    img1, bbsoi1 = iaa.Resize(dict(width=(0.8, 1.0)))(image=img1, bounding_boxes=bbsoi1)
    img2, bbsoi2 = fontlib.hdraw_text(text2, 100, fonts)
    img2, bbsoi2 = iaa.Resize(dict(width=(0.6, 0.9)))(image=img2, bounding_boxes=bbsoi2)
    return fontlib.vstack_center_aligned(img1, bbsoi1, img2, bbsoi2, np.random.randint(0, 10))

# 1줄 : 지역(세로) + 7자리 (경기12가2345)
def gen_kor_lp_4(fonts):
    img, bbsoi = gen_kor_lp_0(fonts, 2)
    img0, bbsoi0 = fontlib.vdraw_text(np.random.choice(kor_regions), 50, fonts)
    img0, bbsoi0 = iaa.Resize(dict(width=(0.9, 1.5), height=(0.9, 1.1)))(image=img0, bounding_boxes=bbsoi0)
    return fontlib.hstack_center_aligned(img0, bbsoi0, img, bbsoi, np.random.randint(0, 30))

def random_template():
    return np.random.choice([gen_kor_lp_0, gen_kor_lp_1, gen_kor_lp_2, gen_kor_lp_3, gen_kor_lp_4])

def random_crop_pad(image, psoi):
    dice = lambda low, high: iap.TruncatedNormal((low + high) / 2, (high - low) / 3, low, high)
    hdice, vdice = dice(-0.1, 0.2), dice(-0.1, 0.3)
    bbu = BoundingBox.from_point_soup(psoi.to_xy_array())
    while True:
        l2 = int(np.floor(bbu.x1 - hdice.draw_sample() * bbu.width))
        t2 = int(np.floor(bbu.y1 - vdice.draw_sample() * bbu.height))
        r2 = int(np.ceil(bbu.x2 + hdice.draw_sample() * bbu.width))
        b2 = int(np.ceil(bbu.y2 + vdice.draw_sample() * bbu.height))
        psoi2 = psoi.copy().shift(-l2, -t2)
        shape2 = (b2 - t2, r2 - l2)
        if max([p.compute_out_of_image_fraction(shape2) for p in psoi2]) < 0.1:
            break
    return iaa.CropAndPad(
        px=(-t2, r2-image.shape[1], b2-image.shape[0], -l2),
        keep_size=False)(image=image, polygons=psoi)

def gen_lp_image(template, idx=0):
    W, H = 288, 96
    bg_color, font_color = fontlib.get_rand_color2(MIN_COLOR_DIFF)
    kor_font = get_random_kor_font()
    kor_font_fallback = get_random_kor_font(no_img_font=True)
    num_font = get_random_num_font()
    fonts = dict(eng=num_font, kor=kor_font, fallback=kor_font_fallback)

    gen_func = random_template()
    image, bbsoi = gen_func(fonts)
    psoi = bbsoi.to_polygons_on_image()

    # handle affine/perspective transform with gray image first
    image, psoi = iaa.Sequential([
        iaa.Rotate(iap.TruncatedNormal(0, 5, -10, 10), fit_output=True),
        iaa.Sometimes(0.3, 
            iaa.PerspectiveTransform(scale=(0.0, 0.1), keep_size=False, fit_output=True))
    ])(image=image, polygons=psoi)
    image, psoi = random_crop_pad(image, psoi)
    image, psoi = iaa.Resize(dict(width=W, height=H))(image=image, polygons=psoi)
    image = fontlib.colorize(image, font_color, bg_color)

    # do augmentation
    image_a, psoi_a = augement_img(image, psoi)

    # convert to shapes
    classes = [all_chars.find(p.label) for p in psoi_a]
    assert np.all(np.array(classes) >= 0)
    scale = np.float32([W, H]).reshape(2, 1)
    shapes = [Shape(pts=np.clip(p.coords.T / scale, 0, 1), text=str(cl)) for p, cl in zip(psoi_a, classes)]

    return image_a, psoi_a, shapes

def gen_data(base_path, idx):
    image, psoi, shapes = gen_lp_image(random_template(), idx)
    labels = [Label(int(s.text), np.min(s.pts, 1), np.max(s.pts, 1)) for s in shapes]
    if IMSHOW:
        # draw shapes(polygons)
        image_to_show = psoi.draw_on_image(image, alpha_face=0)
        # test solve_to_text
        labels_perm = labels.copy()
        np.random.shuffle(labels_perm)
        print('\n', postprocess.solve_to_text(labels_perm, all_chars))
        # imshow
        cv2.namedWindow('img', 0) # make resizable
        cv2.imshow("img", image_to_show)
        if cv2.waitKey(0) == 27:
            exit()

    base_path = Path(base_path)
    cv2.imwrite(str(base_path / (str(idx).zfill(5) + ".jpg")), image)
    write_shapes(str(base_path / (str(idx).zfill(5) + "_shapes.txt")), shapes)
    lwrite(str(base_path / (str(idx).zfill(5) + ".txt")), labels, write_probs=False)

def gen_dataset(base_path, cnt=20):
    Path(base_path).mkdir(parents=True, exist_ok=True)
    for idx in tqdm(range(cnt)):
        gen_data(base_path, idx)

if __name__ == "__main__":
    gen_dataset('tmp/sample/', 32)
    # gen_dataset('_train_ocr/dataset/synth/train/', 12000)
    # gen_dataset('_train_ocr/dataset/synth/val/', 3000)
