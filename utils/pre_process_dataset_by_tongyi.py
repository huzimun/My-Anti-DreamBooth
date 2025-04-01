from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
import requests
from dashscope import ImageSynthesis
import os

prompt = "人脸肖像"

# 上传参考图方式：url链接和本地路径二选一
# 若两者存在，ref_img参数优先级更高
# # 使用公网url链接
# ref_img = "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241031/rguyzt/girl.png"
# 使用本地文件路径
sketch_image_url = '/data1/humw/Datasets/CelebA-HQ/5/set_B/3645.jpg'

print('----sync call, please wait a moment----')
rsp = ImageSynthesis.call(api_key=os.getenv("DASHSCOPE_API_KEY"),
                          model=ImageSynthesis.Models.wanx_v1,
                          prompt=prompt,
                          n=1,
                          style='<photography>',
                          size='1024*1024',
                          ref_mode='repaint',
                          ref_strength=1.0,
                          sketch_image_url=sketch_image_url,
                        #   ref_img=ref_img
                          )
print(rsp)
if rsp.status_code == HTTPStatus.OK:
    print(rsp.output)
    # 保留图片到当前目录
    for result in rsp.output.results:
        file_name = PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
        with open('./%s' % file_name, 'wb+') as f:
            f.write(requests.get(result.url).content)
else:
    print('sync_call Failed, status_code: %s, code: %s, message: %s' %
          (rsp.status_code, rsp.code, rsp.message))