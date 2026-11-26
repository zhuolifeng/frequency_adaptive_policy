import numpy as np
from ImplicitRDP.common.cv2_util import get_image_transform


def test_resized_image_shape():
    tf = get_image_transform((1280,720), (640,480), bgr_to_rgb=False)
    in_img = np.zeros((720,1280,3), dtype=np.uint8)
    out_img = tf(in_img)
    assert out_img.shape == (480,640,3)
