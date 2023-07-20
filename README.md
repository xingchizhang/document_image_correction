# document_image_correction
# 文档图像矫正
# 项目描述
对发生透视变换的文档图像进行几何校正处理，得到规范的文档图像。  
几何校正的目的是把发生了透视变换的目标变换为具有真实比例和角度的目标，如下图所示：  
左图中的地板砖经过透视变换之后，不再是正方形，经过校正之后，得到右侧图像，其中的地板砖被校正为正方形： 
<img src="https://github.com/xingchizhang/document_image_correction/blob/main/imgs/img1.jpg" height="340px">
<img src="https://github.com/xingchizhang/document_image_correction/blob/main/imgs/img2.jpg" height="340px">
# 矫正结果展示
共两行图片，每行图片中，第一层为输入，第二层为输出（矫正结果）  
<img src="https://github.com/xingchizhang/document_image_correction/blob/main/imgs/img3.jpg" height="350px"><img src="https://github.com/xingchizhang/document_image_correction/blob/main/imgs/img4.jpg" height="350px"><img src="https://github.com/xingchizhang/document_image_correction/blob/main/imgs/img5.jpg" height="350px">  
<img src="https://github.com/xingchizhang/document_image_correction/blob/main/imgs/img6.jpg" height="350px"><img src="https://github.com/xingchizhang/document_image_correction/blob/main/imgs/img7.jpg" height="350px"><img src="https://github.com/xingchizhang/document_image_correction/blob/main/imgs/img8.jpg" height="350px">
# 算法原理
## 根本原理：
通过Canny+Hough变换得到图像中的边缘直线。通过对这些直线的处理，获取到原始图像中的bounding box，进而得到四个角点。利用原始图像与目标图像角点之间的对应关系求出单应变换矩阵，最终实现图像的矫正。  
## 关键步骤：
通过对直线的处理，获取bounding box，并得到正确顺序的四个角点  
## 实现方法：  
### 对直线的处理：  
①清除噪点  
②清理短线  
③合并直线  
④清理外线  
⑤清理短线  
⑥清理内线  
⑦获取bounding box  
⑧求出角点并排序  
### 直线处理完成的后续操作：  
①求出目标图像的四个角点  
②获取单应变换矩阵  
③图像矫正
