def get_attr():
    # 加载celeba数据集的人脸属性
    with open('./data/list_attr_celeba.txt', 'r') as f:
        attr = f.readlines()
    # 去掉第一二行
    attr = attr[2:]
    attr = [i.strip().split() for i in attr]
    img_name = [i[0] for i in attr]
    attr_label = [i[1:] for i in attr]
    # 将人脸属性转换为int类型
    attr_label = [[int(j) for j in i] for i in attr_label]
    return img_name, attr_label