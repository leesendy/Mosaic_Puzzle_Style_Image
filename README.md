# Mosaic_Puzzle_Style_Image

# 中文描述
整体描述：生成马赛克风格图像，图1所示著名海报《楚门的世界》(1998)
![Fig.1](https://user-images.githubusercontent.com/51181927/168855980-a8016d90-de20-490f-adf9-aebca3d2fbee.jpeg)

所需数据:
1.目标图像
![target_0](https://user-images.githubusercontent.com/51181927/169344625-a34561a8-87f7-4a45-aa7c-2c9794301ebc.jpeg)

2.匹配图像库 
cifar10 数据集
![cifar10](https://user-images.githubusercontent.com/51181927/169332343-02569254-2a12-4340-bb90-45ee3778bbdf.png)

flowers 数据集
![fig2](https://user-images.githubusercontent.com/51181927/169332407-765a8c70-cde8-435a-977e-2218805eb283.png)

输出结果：
由匹配图像库转化而成的马赛克风格目标图像 如图所示
![target_0_mosaic_2022-05-19-23-46](https://user-images.githubusercontent.com/51181927/169344652-577d44e2-e321-4447-aa72-03c5c16b921f.jpg)

# 使用
Step1 获取匹配图像库
使用flowers-数据库，可解压flowers数据库(./data_source)到./match_data_flowers中;
使用cifar10数据集，运行utils，即可获取图像至./match_data_cifar10中;

Step2 选取目标图像
将目标图像放置于./target_data 文件夹中

Step3 运行main.py
e.g. python main.py --target_img_path './target_data/target_0.jpeg' --file_format '.jpeg' --match_img_path './match_data_cifar10' 
--groups 10 --gain 2 --param_px 30
(推荐在main.py中直接修改， 115-123 line)

Step4 查看图像 调整参数
输出图像位于源图像位置，可不断调整放大比例gain 色素块 param_px 以实现更好效果

# 联系我
lidisen@126.com
