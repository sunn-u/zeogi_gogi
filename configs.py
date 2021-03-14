
class Configs:
    # settings
    data_root = 'C:/Users/tjsnt/zeogi_gogi/data'

    # processing data
    image_gray = True
    normalize = (0.5,) if image_gray else (0.5, 0.5, 0.5)
    split = True
    train_percent = 0.8
