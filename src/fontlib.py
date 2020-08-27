import cv2
import numpy as np
import re
import PIL
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

import imgaug.augmenters as iaa
from imgaug import parameters as iap
from imgaug.augmenters.blend import blend_alpha
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

def get_color_diff(c1, c2):
    # bgr to sRGB
    c1_rgb = sRGBColor(*c1[::-1], is_upscaled=True)
    c2_rgb = sRGBColor(*c2[::-1], is_upscaled=True)
    # rgb to LabColor
    c1_lab = convert_color(c1_rgb, LabColor)
    c2_lab = convert_color(c2_rgb, LabColor)
    # Find the color difference
    return delta_e_cie2000(c1_lab, c2_lab)

def get_rand_color2(min_diff):
    for _ in range(100):
        c1 = np.random.randint(256, size=3).astype(np.uint8)
        c2 = np.random.randint(256, size=3).astype(np.uint8)
        if get_color_diff(c1, c2) > min_diff:
            break
    return c1, c2

def warp_affine_bb(mat, bb):
    assert isinstance(bb, BoundingBox)
    ptsh = np.insert(bb.to_polygon().coords, 2, 1, axis=1) # convert to homogeneous
    bbw = BoundingBox.from_point_soup(ptsh @ mat.T)
    bbw.label = bb.label
    return bbw

class FontBase(object):
    def __init__(self):
        self.glyphs = {}

    def has_glyph(self, char):
        return char in self.glyphs

    def put_glyph(self, img, char, dst):
        glyph = self.glyphs[char]
        glyph_img = glyph['image']
        h0, w0 = glyph_img.shape[:2]
        dst.clip_out_of_image_(img)
        l, t, r, b = dst.x1_int, dst.y1_int, dst.x2_int, dst.y2_int        
        h1, w1 = b - t, r - l
        # copy src to dst
        interp = cv2.INTER_AREA if h1 * w1 < h0 * w0 else cv2.INTER_LINEAR
        glyph_img = cv2.resize(glyph_img, (w1, h1), interpolation=interp)
        img[t:b, l:r] = cv2.add(img[t:b, l:r], glyph_img)
        # transform bbox
        tfm = np.float32([[w1/w0, 0, l], [0, h1/h0, t]])
        bb = warp_affine_bb(tfm, glyph['bb'])
        return img, bb

class ImgFont(FontBase):
    def __init__(self, path, inverted):
        super(ImgFont, self).__init__()
        self._gen_glyphs(path, inverted)

    def _gen_glyphs(self, path, inverted):
        # collect all font images
        for x in path.iterdir():
            if x.is_file():
                img = np.fromfile(str(x), np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
                img = 255 - img if inverted else img
                self.glyphs[x.stem] = dict(image=img, bb=self._calc_bb(img, x.stem))
        first_glyph = next(iter(self.glyphs.values()))
        first_img = first_glyph['image']
        h, w = first_img.shape[:2]
        print('* image font loaded:', str(path), 'width:', w, 'height:', h)

    def _calc_bb(self, gray, label):
        h, w = gray.shape[:2]
        valid_rows = np.any(gray > 64, axis = 1)
        valid_cols = np.any(gray > 64, axis = 0)
        l = np.argmax(valid_cols) - 1
        t = np.argmax(valid_rows) - 1
        r = w - np.argmax(np.flip(valid_cols)) + 1
        b = h - np.argmax(np.flip(valid_rows)) + 1
        # print("** (l, t, r, b) =", (l, t, r, b))
        return BoundingBox(l, t, r, b, label=label)

class TrueTypeFont(FontBase):
    def __init__(self, font_name, groups):
        super(TrueTypeFont, self).__init__()
        self.font_name = font_name
        for chars in groups:
            self._gen_glyphs(list(chars))

    def _gen_glyph_single(self, font, img_size, char):
        # draw glyph
        image = Image.new('L', (img_size, img_size), 0) # gray image
        draw = ImageDraw.Draw(image)
        lt = (0.25 * img_size, 0.25 * img_size)
        draw.text(lt, char, fill=255, font=font)
        # calcuate bbox
        (offset_x, offset_y) = font.getoffset(char)
        bb = BoundingBox(*font.getmask(char).getbbox(), label=char)
        bb.shift_(lt[0] + offset_x, lt[1] + offset_y)
        return dict(image=np.array(image), bb=bb)

    def _gen_glyphs(self, chars):
        font_size = 128
        img_size = font_size * 2

        # collect all src glyphs
        font = ImageFont.truetype(self.font_name, size=font_size)
        glyphs = {}
        for char in chars:
            glyphs[char] = self._gen_glyph_single(font, img_size, char)
        
        # calcuate dst size: width = max width among glyphs, height = bb of bbs
        bb_list = [v['bb'] for v in glyphs.values()]
        top = np.min([bb.y1 for bb in bb_list])
        bottom = np.max([bb.y2 for bb in bb_list])
        width = int(np.ceil(np.max([bb.width for bb in bb_list])))
        height = int(np.ceil(bottom - top))
        print('* font loaded:', self.font_name, 'width:', width, 'height:', height)

        for char, glyph in glyphs.items():
            bb = glyph['bb']
            cx, cy = bb.center_x, 0.5 * (top + bottom) # src anchor
            tfm = np.float32([[1, 0, 0.5 * width - cx], [0, 1, 0.5 * height - cy]])
            dst = cv2.warpAffine(glyph['image'], tfm, (width, height))
            bb = warp_affine_bb(tfm, bb)
            self.glyphs[char] = dict(image=dst, bb=bb)
            # cv2.namedWindow('img', 0)
            # dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
            # dst = bb.draw_on_image(dst)          
            # cv2.imshow('img', dst)  
            # cv2.waitKey(0)

def _validate_layout(rect_list):
    bbsoi = BoundingBoxesOnImage(rect_list, shape=(1, 1))
    bbu = BoundingBox.from_point_soup(bbsoi.to_xy_array())
    bbsoi.shift_(-bbu.x1, -bbu.y1)
    bbsoi.shape = (int(round(bbu.height)), int(round(bbu.width)))
    return bbsoi

def _sel_font(fonts, char):
    if char in '0123456789':
        return fonts['eng']
    font_kor = fonts['kor']
    return font_kor if font_kor.has_glyph(char) else fonts['fallback']

def _tokenize_text(text, fonts):
    chars = ''.join(list(set([c for f in fonts.values() for c in f.glyphs.keys()])))
    return re.findall(f'[{chars}]|_+', text)

def _draw_text(text, sz, fonts, is_horz):
    rect_list = []
    pos = 0
    sz_dice = iap.Discretize(iap.Multiply(iap.TruncatedNormal(0.0, 0.07, -0.1, 0.1), sz))
    pos_dice = iap.Discretize(iap.Multiply(iap.TruncatedNormal(0.15, 0.07, 0.05, 0.25), sz))
    for token in _tokenize_text(text, fonts):
        if token.startswith('_'): # spacing
            spacing = np.random.randint(0, len(token) * round(sz * 0.5))
            pos += spacing
            continue
        rect = BoundingBox(0, 0, sz, sz, label=token)
        rect.shift_(pos if is_horz else 0, 0 if is_horz else pos)
        if is_horz:
            rect.extend_(0, sz_dice.draw_sample() // 2, sz_dice.draw_sample(), sz_dice.draw_sample() // 2, 0)
            pos = rect.x2_int + pos_dice.draw_sample()
        else:
            rect.extend_(0, 0, sz_dice.draw_sample() // 2, sz_dice.draw_sample(), sz_dice.draw_sample() // 2)
            pos = rect.y2_int + pos_dice.draw_sample()
        rect_list.append(rect)

    layout = _validate_layout(rect_list)
    output = np.zeros(layout.shape[:2], np.uint8)
    bboxes = []
    for dst in layout:
        f = _sel_font(fonts, dst.label)
        output, bb = f.put_glyph(output, dst.label, dst)
        bboxes.append(bb)
    bboxes = BoundingBoxesOnImage(bboxes, shape=output.shape)
    return output, bboxes

def hdraw_text(text, h, fonts):
    return _draw_text(text, h, fonts, True)

def vdraw_text(text, w, fonts):
    return _draw_text(text, w, fonts, False)

def _stack_center_aligned(img1, bbsoi1, img2, bbsoi2, spacing, offset, is_horz):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if is_horz: # hstack
        hmax = max(h1, h2)
        rect1 = BoundingBox(0, 0, w1, h1).shift_(0, (hmax - h1) // 2 + offset)
        rect2 = BoundingBox(0, 0, w2, h2).shift_(w1 + spacing, (hmax - h2) // 2)
    else: # vstack
        wmax = max(w1, w2)
        rect1 = BoundingBox(0, 0, w1, h1).shift_((wmax - w1) // 2 + offset, 0)
        rect2 = BoundingBox(0, 0, w2, h2).shift_((wmax - w2) // 2, h1 + spacing)
    layout = _validate_layout([rect1, rect2])
    l1, t1 = layout[0].x1_int, layout[0].y1_int
    l2, t2 = layout[1].x1_int, layout[1].y1_int

    output = np.zeros(layout.shape[:2], np.uint8)
    output[t1:t1+h1, l1:l1+w1] = img1
    bbsoi1.shift_(l1, t1)
    output[t2:t2+h2, l2:l2+w2] = img2
    bbsoi2.shift_(l2, t2)
    return output, BoundingBoxesOnImage(bbsoi1.bounding_boxes + bbsoi2.bounding_boxes, shape=layout.shape[:2])

def vstack_center_aligned(img1, bbsoi1, img2, bbsoi2, spacing=0, offset=0):
    return _stack_center_aligned(img1, bbsoi1, img2, bbsoi2, spacing, offset, False)

def hstack_center_aligned(img1, bbsoi1, img2, bbsoi2, spacing=0, offset=0):
    return _stack_center_aligned(img1, bbsoi1, img2, bbsoi2, spacing, offset, True)

def colorize(img, fg_color, bg_color):
    h, w = img.shape[:2]
    fg = np.full((h, w, 3), fg_color, np.uint8)
    bg = np.full((h, w, 3), bg_color, np.uint8)
    alpha = img.astype(np.float32) / 255
    return blend_alpha(fg, bg, alpha)

if __name__ == '__main__':
    font1 = TrueTypeFont('data\\ocr-kor\\font_kor\\NotoSansKR-Bold.otf', ['서울가나다라마바고노도로보조'])
    font2 = TrueTypeFont('data\\ocr-kor\\font_eng\\Staatliches-Regular.ttf', ['0123456789'])

    fonts = dict(eng=font2, kor=font1, fallback=font1)
    img, bbs = hdraw_text('03조6215', 100, fonts)
    img = colorize(img, (0, 255, 255), (64, 64, 64))
    img = bbs.draw_on_image(img)

    cv2.namedWindow('img', 0)
    cv2.imshow('img', img)            
    cv2.waitKey(0)
